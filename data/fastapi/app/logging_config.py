"""Configuration centralisée des journaux de l'API PRUDENCIA."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from typing import TextIO

from app.request_context import get_request_id

LOGGER_NAME = "prudencia"
SUPPORTED_FORMATS = {"json", "text"}
_STRUCTURED_FIELDS = (
    "request_id",
    "method",
    "path",
    "status_code",
    "duration_ms",
    "operation",
)


class RequestContextFilter(logging.Filter):
    """Ajoute l'identifiant de requête aux événements qui n'en ont pas."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = get_request_id()
        return True


class JsonFormatter(logging.Formatter):
    """Sérialise un événement par ligne, dans un JSON lisible par les outils de logs."""

    def format(self, record: logging.LogRecord) -> str:
        event: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(
                record.created,
                tz=UTC,
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for field in _STRUCTURED_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                event[field] = value

        if record.exc_info:
            event["exception"] = self.formatException(record.exc_info)

        return json.dumps(event, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    """Produit une sortie compacte destinée au développement local."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)s %(name)s "
            "[request_id=%(request_id)s] %(message)s"
        )


def configure_logging(
    *,
    level: str | None = None,
    output_format: str | None = None,
    stream: TextIO | None = None,
) -> logging.Logger:
    """Configure une seule fois le logger applicatif et retourne sa racine.

    ``LOG_LEVEL`` contrôle le seuil et ``LOG_FORMAT`` accepte ``json`` ou
    ``text``. Les valeurs invalides reviennent à des valeurs sûres afin qu'une
    erreur de configuration ne bloque pas le démarrage du service.
    """

    selected_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    numeric_level = getattr(logging, selected_level, logging.INFO)
    selected_format = (output_format or os.getenv("LOG_FORMAT", "json")).lower()
    if selected_format not in SUPPORTED_FORMATS:
        selected_format = "json"

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(numeric_level)
    logger.propagate = False

    # Le remplacement rend la fonction idempotente lors des rechargements Uvicorn.
    for handler in list(logger.handlers):
        if getattr(handler, "prudencia_managed", False):
            logger.removeHandler(handler)

    handler = logging.StreamHandler(stream or sys.stdout)
    handler.prudencia_managed = True  # type: ignore[attr-defined]
    handler.addFilter(RequestContextFilter())
    handler.setFormatter(
        JsonFormatter() if selected_format == "json" else TextFormatter()
    )
    logger.addHandler(handler)
    return logger
