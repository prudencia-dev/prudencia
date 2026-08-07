import io
import json
import logging

from app.logging_config import JsonFormatter, configure_logging
from app.request_context import bind_request_id, reset_request_id


def test_json_formatter_exposes_structured_request_fields() -> None:
    record = logging.LogRecord(
        name="prudencia.requests",
        level=logging.INFO,
        pathname=__file__,
        lineno=12,
        msg="Requête HTTP terminée",
        args=(),
        exc_info=None,
    )
    record.request_id = "8de15f01-7c7e-4a65-9492-a64117551fbd"
    record.method = "GET"
    record.path = "/health"
    record.status_code = 200
    record.duration_ms = 4.2

    event = json.loads(JsonFormatter().format(record))

    assert event["message"] == "Requête HTTP terminée"
    assert event["request_id"] == record.request_id
    assert event["method"] == "GET"
    assert event["path"] == "/health"
    assert event["status_code"] == 200
    assert event["duration_ms"] == 4.2


def test_configured_logger_uses_request_context() -> None:
    output = io.StringIO()
    logger = configure_logging(output_format="json", stream=output)
    token = bind_request_id("8de15f01-7c7e-4a65-9492-a64117551fbd")

    try:
        logger.getChild("tests").info("Événement de test")
    finally:
        reset_request_id(token)

    event = json.loads(output.getvalue())
    assert event["request_id"] == "8de15f01-7c7e-4a65-9492-a64117551fbd"
    assert event["logger"] == "prudencia.tests"


def test_invalid_format_falls_back_to_json() -> None:
    output = io.StringIO()
    logger = configure_logging(output_format="unsupported", stream=output)

    logger.info("Format de repli")

    assert json.loads(output.getvalue())["message"] == "Format de repli"
