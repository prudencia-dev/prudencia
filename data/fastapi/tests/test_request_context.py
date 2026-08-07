from uuid import UUID

from app.request_context import (
    REQUEST_ID_HEADER,
    RequestContextMiddleware,
    bind_request_id,
    get_request_id,
    normalize_request_id,
    reset_request_id,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient

app = FastAPI()
app.add_middleware(RequestContextMiddleware)


@app.get("/test")
def sample_endpoint() -> dict[str, str]:
    return {"status": "ok"}


client = TestClient(app)


def test_valid_request_id_is_normalized() -> None:
    request_id = "A4CE9F9C-5946-4A02-86AC-4D6F925EDC70"

    assert normalize_request_id(request_id) == request_id.lower()


def test_invalid_request_id_is_replaced_with_uuid() -> None:
    generated = normalize_request_id("invalid\nlog-entry")

    assert str(UUID(generated)) == generated


def test_request_id_context_is_restored() -> None:
    initial = get_request_id()
    token = bind_request_id("8de15f01-7c7e-4a65-9492-a64117551fbd")

    try:
        assert get_request_id() == "8de15f01-7c7e-4a65-9492-a64117551fbd"
    finally:
        reset_request_id(token)

    assert get_request_id() == initial


def test_middleware_returns_request_id_header() -> None:
    request_id = "8de15f01-7c7e-4a65-9492-a64117551fbd"

    response = client.get("/test", headers={REQUEST_ID_HEADER: request_id})

    assert response.status_code == 200
    assert response.headers[REQUEST_ID_HEADER] == request_id


def test_middleware_replaces_invalid_header() -> None:
    response = client.get(
        "/test",
        headers={REQUEST_ID_HEADER: "not-a-uuid"},
    )

    generated = response.headers[REQUEST_ID_HEADER]
    assert str(UUID(generated)) == generated
