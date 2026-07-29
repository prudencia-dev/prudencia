from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from psycopg.rows import dict_row

from app.database import get_connection


router = APIRouter(prefix="/questionnaire-mvp", tags=["Questionnaire MVP"])

QUESTIONNAIRE_NAME = "Questionnaire PRUDENCIA MVP"
TECHNICAL_QUESTION_CODE = "ML_TYPE_IA"
DEFAULT_AI_TYPE = "scoring"


class Submission(BaseModel):
    project_name: str = Field(min_length=1, max_length=255)
    project_description: str | None = None
    respondent_id: str | None = None
    answers: dict[str, Any]


def normalize_options(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    return value if isinstance(value, list) else []


def answer_is_present(value: Any, answer_type: str) -> bool:
    if answer_type == "boolean":
        return value is not None
    if answer_type == "multiple_choice":
        return isinstance(value, list) and len(value) > 0
    if isinstance(value, str):
        return bool(value.strip())
    return value is not None


def insert_answer(
    cur: Any,
    response_id: str,
    question: dict[str, Any],
    value: Any,
    now: datetime,
) -> None:
    answer_type = str(question["answer_type"])

    if answer_type == "boolean":
        column, stored_value = "value_boolean", bool(value)
    elif answer_type == "integer":
        column, stored_value = "value_integer", int(value)
    elif answer_type == "decimal":
        column, stored_value = "value_decimal", float(value)
    elif answer_type in {"text", "single_choice"}:
        column, stored_value = "value_text", str(value)
    elif answer_type == "multiple_choice":
        column = "value_json"
        stored_value = json.dumps(value, ensure_ascii=False)
    else:
        raise HTTPException(
            status_code=422,
            detail=f"Type de réponse non pris en charge : {answer_type}",
        )

    cast = "::jsonb" if column == "value_json" else ""

    cur.execute(
        f'''
        INSERT INTO prudencia.answers (
            response_id, question_id, {column}, created_at, updated_at
        )
        VALUES (%s, %s, %s{cast}, %s, %s)
        ''',
        (response_id, question["id"], stored_value, now, now),
    )


def build_ml_features(answers: dict[str, Any]) -> dict[str, str]:
    required_codes = {"Q2", "Q9", "Q15"}
    missing = sorted(required_codes - set(answers))

    if missing:
        raise HTTPException(
            status_code=422,
            detail="Réponses nécessaires au modèle manquantes : " + ", ".join(missing),
        )

    sensitive_values = {
        "sante",
        "biometrie",
        "genetiques",
        "opinions_religion",
        "orientation_sexuelle",
    }

    q10 = answers.get("Q10") or []
    if not isinstance(q10, list):
        q10 = [q10]

    return {
        "secteur_grp": str(answers["Q2"]),
        "role": str(answers["Q15"]),
        "donnees_perso": "oui" if answers["Q9"] is True else "non",
        "donnees_sensibles": (
            "oui" if any(value in sensitive_values for value in q10) else "non"
        ),
        "type_ia_norm": DEFAULT_AI_TYPE,
    }


def get_or_create_project(
    cur: Any,
    project_name: str,
    project_description: str | None,
) -> str:
    clean_name = project_name.strip()

    if not clean_name:
        raise HTTPException(status_code=422, detail="Le nom du projet est obligatoire.")

    cur.execute(
        '''
        SELECT id
        FROM prudencia.projects
        WHERE LOWER(name) = LOWER(%s)
        ORDER BY created_at DESC
        LIMIT 1
        ''',
        (clean_name,),
    )

    existing = cur.fetchone()
    if existing:
        return str(existing["id"])

    cur.execute(
        '''
        INSERT INTO prudencia.projects (name, description)
        VALUES (%s, %s)
        RETURNING id
        ''',
        (
            clean_name,
            project_description.strip() if project_description else None,
        ),
    )

    return str(cur.fetchone()["id"])


@router.get("/active")
def active_questionnaire() -> dict[str, Any]:
    query = '''
        SELECT
            qn.id AS questionnaire_id,
            qn.name,
            qn.description AS questionnaire_description,
            qn.version,
            q.id AS question_id,
            q.code,
            q.label,
            q.description,
            q.category,
            q.answer_type,
            q.possible_values,
            q.display_order,
            q.is_required
        FROM prudencia.questionnaires qn
        JOIN prudencia.questions q
          ON q.questionnaire_id = qn.id
        WHERE qn.name = %s
          AND qn.is_active = TRUE
          AND q.is_active = TRUE
          AND q.code <> %s
        ORDER BY q.display_order
    '''

    try:
        with get_connection() as db:
            with db.cursor(row_factory=dict_row) as cur:
                cur.execute(query, (QUESTIONNAIRE_NAME, TECHNICAL_QUESTION_CODE))
                rows = cur.fetchall()
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur de lecture : {error}",
        ) from error

    if not rows:
        raise HTTPException(status_code=404, detail="Questionnaire actif introuvable.")

    first = rows[0]

    return {
        "success": True,
        "questionnaire": {
            "id": str(first["questionnaire_id"]),
            "name": first["name"],
            "description": first["questionnaire_description"],
            "version": first["version"],
            "questions": [
                {
                    "id": str(row["question_id"]),
                    "code": row["code"],
                    "label": row["label"],
                    "description": row["description"],
                    "category": row["category"],
                    "answer_type": row["answer_type"],
                    "options": normalize_options(row["possible_values"]),
                    "display_order": row["display_order"],
                    "is_required": row["is_required"],
                }
                for row in rows
            ],
        },
    }


@router.post("/submit")
def submit(payload: Submission) -> dict[str, Any]:
    try:
        with get_connection() as db:
            with db.cursor(row_factory=dict_row) as cur:
                project_id = get_or_create_project(
                    cur,
                    payload.project_name,
                    payload.project_description,
                )

                cur.execute(
                    '''
                    SELECT id
                    FROM prudencia.questionnaires
                    WHERE name = %s AND is_active = TRUE
                    LIMIT 1
                    ''',
                    (QUESTIONNAIRE_NAME,),
                )

                questionnaire = cur.fetchone()
                if questionnaire is None:
                    raise HTTPException(
                        status_code=404,
                        detail="Questionnaire actif introuvable.",
                    )

                questionnaire_id = str(questionnaire["id"])

                cur.execute(
                    '''
                    SELECT id, code, answer_type, is_required
                    FROM prudencia.questions
                    WHERE questionnaire_id = %s
                      AND is_active = TRUE
                      AND code <> %s
                    ORDER BY display_order
                    ''',
                    (questionnaire_id, TECHNICAL_QUESTION_CODE),
                )

                questions = cur.fetchall()
                by_code = {str(row["code"]): row for row in questions}

                missing = [
                    code
                    for code, row in by_code.items()
                    if row["is_required"]
                    and not answer_is_present(
                        payload.answers.get(code),
                        str(row["answer_type"]),
                    )
                ]

                if missing:
                    raise HTTPException(
                        status_code=422,
                        detail="Réponses obligatoires manquantes : "
                        + ", ".join(sorted(missing)),
                    )

                features = build_ml_features(payload.answers)
                now = datetime.now(timezone.utc)

                cur.execute(
                    '''
                    INSERT INTO prudencia.questionnaire_responses (
                        project_id,
                        questionnaire_id,
                        respondent_id,
                        status,
                        started_at,
                        completed_at,
                        created_at
                    )
                    VALUES (%s, %s, %s, 'completed', %s, %s, %s)
                    RETURNING id
                    ''',
                    (
                        project_id,
                        questionnaire_id,
                        payload.respondent_id,
                        now,
                        now,
                        now,
                    ),
                )

                response_id = str(cur.fetchone()["id"])

                for code, value in payload.answers.items():
                    question = by_code.get(code)
                    if question is None:
                        continue
                    if not answer_is_present(value, str(question["answer_type"])):
                        continue

                    insert_answer(cur, response_id, question, value, now)

            db.commit()

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur d'enregistrement : {error}",
        ) from error

    return {
        "success": True,
        "project_id": project_id,
        "response_id": response_id,
        "questionnaire_id": questionnaire_id,
        "features": features,
    }