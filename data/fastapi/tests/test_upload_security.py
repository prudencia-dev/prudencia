from io import BytesIO
from pathlib import Path

import pytest
from app.services.upload_security import (
    build_stored_filename,
    confined_path,
    copy_limited_upload,
    read_limited_file,
    sanitized_filename,
)
from fastapi import UploadFile


def make_upload(filename: str, content: bytes) -> UploadFile:
    return UploadFile(filename=filename, file=BytesIO(content))


def test_client_path_is_reduced_to_its_basename() -> None:
    assert sanitized_filename("../../contrat.pdf", ".pdf") == "contrat.pdf"


def test_stored_filename_is_unique_and_keeps_only_expected_suffix() -> None:
    first = build_stored_filename("contrat.pdf", ".pdf")
    second = build_stored_filename("contrat.pdf", ".pdf")

    assert first != second
    assert Path(first).suffix == ".pdf"
    assert Path(first).stem.isalnum()


def test_confined_path_never_uses_client_directories(tmp_path: Path) -> None:
    assert confined_path(tmp_path, "../../outside.pdf").parent == tmp_path.resolve()


def test_pdf_signature_is_required_and_invalid_file_is_removed(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "document.pdf"

    with pytest.raises(ValueError, match="PDF valide"):
        copy_limited_upload(
            make_upload("document.pdf", b"not-a-pdf"),
            destination,
            max_bytes=100,
            require_pdf_signature=True,
        )

    assert not destination.exists()


def test_oversized_file_is_rejected_and_removed(tmp_path: Path) -> None:
    destination = tmp_path / "dataset.csv"

    with pytest.raises(ValueError, match="taille maximale"):
        copy_limited_upload(
            make_upload("dataset.csv", b"123456"),
            destination,
            max_bytes=5,
        )

    assert not destination.exists()


def test_limited_reader_accepts_content_at_limit() -> None:
    assert read_limited_file(BytesIO(b"12345"), max_bytes=5) == b"12345"
