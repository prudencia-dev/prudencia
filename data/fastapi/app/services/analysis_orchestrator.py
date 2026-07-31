from __future__ import annotations

import logging
import time
from typing import Any
from uuid import UUID

from app.services.analysis_history_service import AnalysisHistoryService
from app.services.report_builder import PrudenciaReportBuilder


logger = logging.getLogger(__name__)


class AnalysisOrchestrator:
    """
    Orchestre les différentes analyses de PRUDENCIA.

    Responsabilités :

    - centraliser les résultats ML, DL et RAG ;
    - construire le rapport final ;
    - comparer les prédictions brutes au rapport final ;
    - historiser l'analyse ;
    - historiser chaque exécution de modèle ;
    - enregistrer les éventuelles erreurs.

    L'historisation est automatique lorsque le projet contient
    un identifiant UUID valide.
    """

    def __init__(self) -> None:
        self.report_builder = PrudenciaReportBuilder()
        self.history_service = AnalysisHistoryService()

    def analyse(
        self,
        *,
        project: dict[str, Any],
        machine_learning_result: dict[str, Any] | None = None,
        deep_learning_result: dict[str, Any] | None = None,
        rag_result: dict[str, Any] | None = None,
        analysis_type: str = "documentaire",
        questionnaire_response_id: str | None = None,
        requested_by: str | None = None,
        source_name: str | None = None,
        input_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Construit le rapport PRUDENCIA et l'enregistre dans l'historique.

        Le fonctionnement reste rétrocompatible : si aucun ``project_id``
        valide n'est présent, le rapport est tout de même généré, mais
        l'historisation est ignorée.
        """

        ml_result = machine_learning_result or {}
        dl_result = deep_learning_result or {}
        rag_data = rag_result or {}

        project_id = self._extract_project_id(project)
        analysis_id: str | None = None
        started_at = time.perf_counter()

        history_input = input_data or {
            "project": project,
            "machine_learning_result": ml_result,
            "deep_learning_result": dl_result,
            "rag_result": rag_data,
        }

        try:
            if project_id is not None:
                analysis_id = self.history_service.start_analysis(
                    project_id=project_id,
                    analysis_type=analysis_type,
                    questionnaire_response_id=questionnaire_response_id,
                    requested_by=requested_by,
                    input_data=history_input,
                    source_name=(
                        source_name
                        or self._extract_source_name(project)
                    ),
                )

            report = self.report_builder.build(
                project=project,
                machine_learning_result=ml_result,
                deep_learning_result=dl_result,
                rag_result=rag_data,
            )

            raw_predictions = self._build_raw_predictions(
                machine_learning_result=ml_result,
                deep_learning_result=dl_result,
                rag_result=rag_data,
            )

            final_prediction = self._extract_final_prediction(report)

            confidence = self._extract_final_confidence(
                report=report,
                machine_learning_result=ml_result,
                deep_learning_result=dl_result,
            )

            consistency = self._check_consistency(
                raw_predictions=raw_predictions,
                final_prediction=final_prediction,
            )

            execution_time_ms = (
                self.history_service.measure_execution_time_ms(
                    started_at
                )
            )

            report = self._add_diagnostics_to_report(
                report=report,
                analysis_id=analysis_id,
                raw_predictions=raw_predictions,
                final_prediction=final_prediction,
                confidence=confidence,
                consistency=consistency,
                execution_time_ms=execution_time_ms,
                history_enabled=analysis_id is not None,
            )

            if analysis_id is not None:
                self._save_model_executions(
                    analysis_id=analysis_id,
                    machine_learning_result=ml_result,
                    deep_learning_result=dl_result,
                    rag_result=rag_data,
                    input_data=history_input,
                )

                self.history_service.complete_analysis(
                    analysis_id=analysis_id,
                    report=report,
                    final_prediction=final_prediction,
                    confidence=confidence,
                    overall_risk_score=self._confidence_to_risk_score(
                        confidence
                    ),
                    summary=self._extract_summary(report),
                    raw_predictions=raw_predictions,
                    consistency=consistency,
                    metadata={
                        "analysis_type": analysis_type,
                        "execution_time_ms": execution_time_ms,
                        "orchestrator": self.__class__.__name__,
                        "report_builder": (
                            self.report_builder.__class__.__name__
                        ),
                    },
                )

            return report

        except Exception as error:
            if analysis_id is not None:
                try:
                    self.history_service.fail_analysis(
                        analysis_id=analysis_id,
                        error_message=str(error),
                        error_context={
                            "analysis_type": analysis_type,
                            "project_id": project_id,
                            "source_name": source_name,
                        },
                    )
                except Exception:
                    logger.exception(
                        "Impossible d'enregistrer l'échec de "
                        "l'analyse %s.",
                        analysis_id,
                    )

            raise

    def _save_model_executions(
        self,
        *,
        analysis_id: str,
        machine_learning_result: dict[str, Any],
        deep_learning_result: dict[str, Any],
        rag_result: dict[str, Any],
        input_data: dict[str, Any],
    ) -> None:
        """
        Enregistre les résultats fournis par chaque moteur.

        Une absence de résultat n'entraîne aucun enregistrement.
        """

        if machine_learning_result:
            self.history_service.save_model_execution(
                analysis_id=analysis_id,
                model_type="machine_learning",
                model_name=self._extract_model_name(
                    machine_learning_result,
                    default="Random Forest",
                ),
                model_version=self._extract_model_version(
                    machine_learning_result
                ),
                task_name="classification_ai_act_questionnaire",
                input_data=self._extract_engine_input(
                    machine_learning_result,
                    fallback=input_data.get(
                        "machine_learning_input",
                        input_data,
                    ),
                ),
                output_data=machine_learning_result,
                execution_time_ms=self._extract_execution_time(
                    machine_learning_result
                ),
                success=self._extract_success(
                    machine_learning_result
                ),
                error_message=self._extract_error_message(
                    machine_learning_result
                ),
                execution_type="inference",
            )

        if deep_learning_result:
            self.history_service.save_model_execution(
                analysis_id=analysis_id,
                model_type="deep_learning",
                model_name=self._extract_model_name(
                    deep_learning_result,
                    default="JuriBERT",
                ),
                model_version=self._extract_model_version(
                    deep_learning_result
                ),
                task_name="classification_ai_act_documentaire",
                input_data=self._extract_engine_input(
                    deep_learning_result,
                    fallback=input_data.get(
                        "deep_learning_input",
                        input_data,
                    ),
                ),
                output_data=deep_learning_result,
                execution_time_ms=self._extract_execution_time(
                    deep_learning_result
                ),
                success=self._extract_success(
                    deep_learning_result
                ),
                error_message=self._extract_error_message(
                    deep_learning_result
                ),
                execution_type="inference",
            )

        if rag_result:
            self.history_service.save_model_execution(
                analysis_id=analysis_id,
                model_type="rag",
                model_name=self._extract_model_name(
                    rag_result,
                    default="PRUDENCIA RAG",
                ),
                model_version=self._extract_model_version(
                    rag_result
                ),
                task_name="recherche_documentaire_juridique",
                input_data=self._extract_engine_input(
                    rag_result,
                    fallback=input_data.get(
                        "rag_input",
                        input_data,
                    ),
                ),
                output_data=rag_result,
                execution_time_ms=self._extract_execution_time(
                    rag_result
                ),
                success=self._extract_success(rag_result),
                error_message=self._extract_error_message(
                    rag_result
                ),
                execution_type="inference",
            )

    def _build_raw_predictions(
        self,
        *,
        machine_learning_result: dict[str, Any],
        deep_learning_result: dict[str, Any],
        rag_result: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Conserve les prédictions brutes avant construction du rapport.
        """

        return {
            "machine_learning": {
                "prediction": self._extract_prediction(
                    machine_learning_result
                ),
                "confidence": self._extract_confidence(
                    machine_learning_result
                ),
                "probabilities": self._extract_probabilities(
                    machine_learning_result
                ),
                "model_name": self._extract_model_name(
                    machine_learning_result,
                    default=None,
                ),
            },
            "deep_learning": {
                "prediction": self._extract_prediction(
                    deep_learning_result
                ),
                "confidence": self._extract_confidence(
                    deep_learning_result
                ),
                "probabilities": self._extract_probabilities(
                    deep_learning_result
                ),
                "model_name": self._extract_model_name(
                    deep_learning_result,
                    default=None,
                ),
            },
            "rag": {
                "classification": self._extract_prediction(
                    rag_result
                ),
                "confidence": self._extract_confidence(rag_result),
                "model_name": self._extract_model_name(
                    rag_result,
                    default=None,
                ),
            },
        }

    def _check_consistency(
        self,
        *,
        raw_predictions: dict[str, Any],
        final_prediction: str | None,
    ) -> dict[str, Any]:
        """
        Compare le résultat final avec les prédictions ML et DL.

        Le RAG n'est pas considéré comme un classifieur principal.
        """

        normalized_final = self._normalize_prediction(
            final_prediction
        )

        ml_prediction = self._normalize_prediction(
            raw_predictions
            .get("machine_learning", {})
            .get("prediction")
        )

        dl_prediction = self._normalize_prediction(
            raw_predictions
            .get("deep_learning", {})
            .get("prediction")
        )

        available_predictions = {
            key: value
            for key, value in {
                "machine_learning": ml_prediction,
                "deep_learning": dl_prediction,
            }.items()
            if value
        }

        if not normalized_final:
            return {
                "status": "warning",
                "is_consistent": False,
                "message": (
                    "La classification finale du rapport "
                    "n'a pas été trouvée."
                ),
                "final_prediction": None,
                "raw_predictions": available_predictions,
            }

        if not available_predictions:
            return {
                "status": "not_checked",
                "is_consistent": None,
                "message": (
                    "Aucune prédiction brute ML ou DL n'est "
                    "disponible pour effectuer la comparaison."
                ),
                "final_prediction": normalized_final,
                "raw_predictions": {},
            }

        matching_engines = [
            engine
            for engine, prediction in available_predictions.items()
            if prediction == normalized_final
        ]

        disagreeing_engines = [
            engine
            for engine, prediction in available_predictions.items()
            if prediction != normalized_final
        ]

        if matching_engines and not disagreeing_engines:
            return {
                "status": "coherent",
                "is_consistent": True,
                "message": (
                    "La classification finale correspond aux "
                    "prédictions brutes disponibles."
                ),
                "final_prediction": normalized_final,
                "raw_predictions": available_predictions,
                "matching_engines": matching_engines,
                "disagreeing_engines": [],
            }

        if matching_engines and disagreeing_engines:
            return {
                "status": "warning",
                "is_consistent": False,
                "message": (
                    "Les moteurs ne produisent pas tous la même "
                    "classification."
                ),
                "final_prediction": normalized_final,
                "raw_predictions": available_predictions,
                "matching_engines": matching_engines,
                "disagreeing_engines": disagreeing_engines,
            }

        return {
            "status": "incoherent",
            "is_consistent": False,
            "message": (
                "La classification finale ne correspond à aucune "
                "prédiction brute disponible."
            ),
            "final_prediction": normalized_final,
            "raw_predictions": available_predictions,
            "matching_engines": [],
            "disagreeing_engines": list(
                available_predictions.keys()
            ),
        }

    @staticmethod
    def _add_diagnostics_to_report(
        *,
        report: dict[str, Any],
        analysis_id: str | None,
        raw_predictions: dict[str, Any],
        final_prediction: str | None,
        confidence: float | None,
        consistency: dict[str, Any],
        execution_time_ms: int,
        history_enabled: bool,
    ) -> dict[str, Any]:
        """
        Ajoute les informations techniques sans supprimer le rapport métier.
        """

        enriched_report = dict(report)

        enriched_report["analysis_history"] = {
            "analysis_id": analysis_id,
            "saved": history_enabled,
        }

        enriched_report["analysis_diagnostics"] = {
            "final_prediction": final_prediction,
            "confidence": confidence,
            "raw_predictions": raw_predictions,
            "consistency": consistency,
            "execution_time_ms": execution_time_ms,
        }

        return enriched_report

    def _extract_final_prediction(
        self,
        report: dict[str, Any],
    ) -> str | None:
        """
        Recherche la classification finale dans plusieurs structures
        possibles du rapport.
        """

        candidates = [
            report.get("final_prediction"),
            report.get("prediction"),
            report.get("classification"),
            report.get("risk_level"),
            report.get("niveau_risque"),
            self._nested_get(
                report,
                "ai_act",
                "classification",
            ),
            self._nested_get(
                report,
                "ai_act",
                "risk_level",
            ),
            self._nested_get(
                report,
                "result",
                "prediction",
            ),
            self._nested_get(
                report,
                "report",
                "classification",
            ),
            self._nested_get(
                report,
                "report",
                "risk_level",
            ),
        ]

        for candidate in candidates:
            if candidate is not None and str(candidate).strip():
                return self._normalize_prediction(str(candidate))

        return None

    def _extract_final_confidence(
        self,
        *,
        report: dict[str, Any],
        machine_learning_result: dict[str, Any],
        deep_learning_result: dict[str, Any],
    ) -> float | None:
        report_confidence = self._extract_confidence(report)

        if report_confidence is not None:
            return report_confidence

        dl_confidence = self._extract_confidence(
            deep_learning_result
        )

        if dl_confidence is not None:
            return dl_confidence

        return self._extract_confidence(
            machine_learning_result
        )

    @classmethod
    def _extract_prediction(
        cls,
        result: dict[str, Any] | None,
    ) -> str | None:
        if not result:
            return None

        candidates = [
            result.get("prediction"),
            result.get("classification"),
            result.get("risk_level"),
            result.get("predicted_class"),
            result.get("label"),
            result.get("niveau_risque"),
            cls._nested_get(result, "result", "prediction"),
            cls._nested_get(result, "result", "classification"),
            cls._nested_get(result, "ai_act", "classification"),
        ]

        for candidate in candidates:
            if candidate is not None and str(candidate).strip():
                return cls._normalize_prediction(str(candidate))

        return None

    @staticmethod
    def _extract_confidence(
        result: dict[str, Any] | None,
    ) -> float | None:
        if not result:
            return None

        candidates = [
            result.get("confidence"),
            result.get("confidence_score"),
            result.get("score"),
            AnalysisOrchestrator._nested_get(
                result,
                "result",
                "confidence",
            ),
        ]

        for candidate in candidates:
            if candidate is None:
                continue

            try:
                confidence = float(candidate)

                if confidence > 1.0 and confidence <= 100.0:
                    confidence /= 100.0

                return min(max(confidence, 0.0), 1.0)
            except (TypeError, ValueError):
                continue

        return None

    @staticmethod
    def _extract_probabilities(
        result: dict[str, Any] | None,
    ) -> dict[str, float]:
        if not result:
            return {}

        probabilities = (
            result.get("probabilities")
            or result.get("class_probabilities")
            or result.get("scores")
            or {}
        )

        if not isinstance(probabilities, dict):
            return {}

        normalized: dict[str, float] = {}

        for key, value in probabilities.items():
            try:
                normalized[str(key)] = float(value)
            except (TypeError, ValueError):
                continue

        return normalized

    @staticmethod
    def _extract_project_id(
        project: dict[str, Any],
    ) -> str | None:
        candidates = [
            project.get("id"),
            project.get("project_id"),
            project.get("uuid"),
        ]

        for candidate in candidates:
            if candidate in {None, ""}:
                continue

            try:
                return str(UUID(str(candidate)))
            except (TypeError, ValueError):
                continue

        return None

    @staticmethod
    def _extract_source_name(
        project: dict[str, Any],
    ) -> str | None:
        candidates = [
            project.get("source_name"),
            project.get("filename"),
            project.get("file_name"),
            project.get("document_name"),
            project.get("name"),
            project.get("project_name"),
        ]

        for candidate in candidates:
            if candidate is not None and str(candidate).strip():
                return str(candidate).strip()

        return None

    @staticmethod
    def _extract_model_name(
        result: dict[str, Any] | None,
        default: str | None,
    ) -> str | None:
        if not result:
            return default

        value = (
            result.get("model_name")
            or result.get("model")
            or result.get("classifier")
        )

        if value is None:
            return default

        return str(value)

    @staticmethod
    def _extract_model_version(
        result: dict[str, Any] | None,
    ) -> str | None:
        if not result:
            return None

        value = (
            result.get("model_version")
            or result.get("version")
            or result.get("checkpoint")
        )

        if value is None:
            return None

        return str(value)

    @staticmethod
    def _extract_engine_input(
        result: dict[str, Any],
        fallback: Any,
    ) -> dict[str, Any]:
        value = (
            result.get("input_data")
            or result.get("features")
            or result.get("input")
            or fallback
        )

        if isinstance(value, dict):
            return value

        return {
            "value": value,
        }

    @staticmethod
    def _extract_execution_time(
        result: dict[str, Any],
    ) -> int | None:
        value = (
            result.get("execution_time_ms")
            or result.get("duration_ms")
            or result.get("elapsed_ms")
        )

        if value is None:
            return None

        try:
            return max(0, int(float(value)))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_success(
        result: dict[str, Any],
    ) -> bool:
        if "success" not in result:
            return True

        return bool(result.get("success"))

    @staticmethod
    def _extract_error_message(
        result: dict[str, Any],
    ) -> str | None:
        value = (
            result.get("error_message")
            or result.get("error")
        )

        if value is None:
            return None

        return str(value)

    @staticmethod
    def _extract_summary(
        report: dict[str, Any],
    ) -> str | None:
        candidates = [
            report.get("summary"),
            report.get("conclusion"),
            AnalysisOrchestrator._nested_get(
                report,
                "report",
                "summary",
            ),
            AnalysisOrchestrator._nested_get(
                report,
                "report",
                "conclusion",
            ),
        ]

        for candidate in candidates:
            if candidate is not None and str(candidate).strip():
                return str(candidate).strip()[:5000]

        return None

    @staticmethod
    def _confidence_to_risk_score(
        confidence: float | None,
    ) -> float | None:
        """
        Valeur temporaire pour alimenter le champ existant.

        Ce score représente ici la confiance exprimée sur 100.
        Il ne remplace pas un véritable score métier de risque.
        """

        if confidence is None:
            return None

        return round(confidence * 100.0, 2)

    @staticmethod
    def _normalize_prediction(
        prediction: Any,
    ) -> str | None:
        if prediction is None:
            return None

        normalized = (
            str(prediction)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("é", "e")
            .replace("è", "e")
            .replace("ê", "e")
            .replace("à", "a")
        )

        aliases = {
            "prohibited": "interdit",
            "pratique_interdite": "interdit",
            "high_risk": "haut_risque",
            "risque_eleve": "haut_risque",
            "limited_risk": "risque_limite",
            "limite": "risque_limite",
            "minimal_risk": "risque_minimal",
            "minimal": "risque_minimal",
            "hors_champ": "hors_perimetre",
            "out_of_scope": "hors_perimetre",
            "hors_perimetre": "hors_perimetre",
        }

        return aliases.get(normalized, normalized)

    @staticmethod
    def _nested_get(
        data: dict[str, Any],
        *keys: str,
    ) -> Any:
        current: Any = data

        for key in keys:
            if not isinstance(current, dict):
                return None

            current = current.get(key)

        return current