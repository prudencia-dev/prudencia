from __future__ import annotations

import logging
from contextvars import ContextVar, Token
from time import perf_counter
from uuid import UUID, uuid4

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

REQUEST_ID_HEADER = "X-Request-ID"
REQUEST_LOGGER = logging.getLogger("prudencia.requests")
_REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="-")


def normalize_request_id(candidate: str | None) -> str:
    """Accepte uniquement un UUID valide ou génère un nouvel identifiant.

    Cette validation empêche qu'une valeur libre fournie par un client soit
    injectée telle quelle dans les journaux du serveur.
    """

    if candidate:
        try:
            return str(UUID(candidate))
        except ValueError:
            pass
    return str(uuid4())


def get_request_id() -> str:
    """Retourne l'identifiant lié à la requête en cours."""

    return _REQUEST_ID.get()


def bind_request_id(request_id: str) -> Token[str]:
    """Associe un identifiant au contexte asynchrone courant."""

    return _REQUEST_ID.set(request_id)


def reset_request_id(token: Token[str]) -> None:
    """Restaure le contexte précédent après traitement de la requête."""

    _REQUEST_ID.reset(token)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Ajoute un identifiant de corrélation et journalise chaque requête."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = normalize_request_id(request.headers.get(REQUEST_ID_HEADER))
        token = bind_request_id(request_id)
        started_at = perf_counter()

        try:
            response = await call_next(request)
            response.headers[REQUEST_ID_HEADER] = request_id

            REQUEST_LOGGER.info(
                "Requête HTTP terminée",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": round(
                        (perf_counter() - started_at) * 1000,
                        2,
                    ),
                },
            )
            return response
        finally:
            # Le reset évite qu'un identifiant soit réutilisé par une autre
            # requête traitée dans le même worker asynchrone.
            reset_request_id(token)
