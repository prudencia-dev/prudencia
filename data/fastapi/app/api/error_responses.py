from __future__ import annotations

import logging
from typing import NoReturn

from app.request_context import get_request_id
from fastapi import HTTPException, status

LOGGER = logging.getLogger("prudencia.api")


def raise_api_error(
    *,
    operation: str,
    error: Exception,
    detail: str,
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
) -> NoReturn:
    """Journalise une erreur interne et renvoie un message public stable.

    Le traceback est volontairement conservé dans les logs serveur, mais
    l'exception originale n'est jamais incluse dans la réponse HTTP.
    """

    LOGGER.exception(
        "Échec d'une opération API",
        extra={
            "operation": operation,
            "request_id": get_request_id(),
        },
    )
    raise HTTPException(
        status_code=status_code,
        detail=detail,
    ) from error
