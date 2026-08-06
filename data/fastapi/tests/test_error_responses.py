import logging

import pytest
from app.api import reports
from app.api.error_responses import raise_api_error
from app.request_context import bind_request_id, reset_request_id
from fastapi import HTTPException


def test_internal_error_is_logged_but_not_exposed(caplog) -> None:
    secret = "postgresql://user:private-password@database/prudencia"
    request_id = "8de15f01-7c7e-4a65-9492-a64117551fbd"
    token = bind_request_id(request_id)

    try:
        with caplog.at_level(logging.ERROR, logger="prudencia.api"):
            with pytest.raises(HTTPException) as captured:
                try:
                    raise RuntimeError(secret)
                except RuntimeError as error:
                    raise_api_error(
                        operation="tests.failure",
                        error=error,
                        detail="Une erreur interne est survenue.",
                    )
    finally:
        reset_request_id(token)

    assert captured.value.status_code == 500
    assert captured.value.detail == "Une erreur interne est survenue."
    assert secret not in captured.value.detail
    assert caplog.records[-1].operation == "tests.failure"
    assert caplog.records[-1].request_id == request_id
    assert secret in caplog.text


def test_custom_status_code_is_preserved() -> None:
    with pytest.raises(HTTPException) as captured:
        try:
            raise ConnectionError("database unavailable")
        except ConnectionError as error:
            raise_api_error(
                operation="tests.health",
                error=error,
                detail="Service indisponible.",
                status_code=503,
            )

    assert captured.value.status_code == 503
    assert captured.value.detail == "Service indisponible."


def test_report_route_hides_orchestrator_exception(monkeypatch) -> None:
    secret = "internal filesystem path: /private/models/juribert"

    class FailingOrchestrator:
        def analyse(self, **_kwargs):
            raise RuntimeError(secret)

    monkeypatch.setattr(reports, "AnalysisOrchestrator", FailingOrchestrator)

    request = reports.ReportGenerationRequest(project={"title": "Test"})

    with pytest.raises(HTTPException) as captured:
        reports.generate_report(request)

    assert captured.value.status_code == 500
    assert captured.value.detail == "Impossible de générer le rapport PRUDENCIA."
    assert secret not in captured.value.detail
