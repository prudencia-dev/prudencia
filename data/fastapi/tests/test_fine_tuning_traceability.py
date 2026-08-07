from pathlib import Path

from app.api.fine_tuning import (
    _benchmark_signature,
    _sha256_file,
    _source_code_sha256,
)


def test_sha256_file_is_stable(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.csv"
    dataset.write_bytes(b"text,label\nexemple,conforme\n")

    assert _sha256_file(dataset) == _sha256_file(dataset)
    assert len(_sha256_file(dataset)) == 64


def test_benchmark_signature_ignores_dictionary_order() -> None:
    first = {
        "dataset_sha256": "abc",
        "epochs": 10,
        "seed": 42,
    }
    reordered = {
        "seed": 42,
        "epochs": 10,
        "dataset_sha256": "abc",
    }

    assert _benchmark_signature(first) == _benchmark_signature(reordered)


def test_benchmark_signature_changes_with_configuration() -> None:
    baseline = {
        "dataset_sha256": "abc",
        "epochs": 10,
        "seed": 42,
    }
    changed = baseline | {"seed": 7}

    assert _benchmark_signature(baseline) != _benchmark_signature(changed)


def test_source_code_sha256_changes_with_source(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    source.write_text("VALUE = 1\n")
    first = _source_code_sha256(tmp_path)

    source.write_text("VALUE = 2\n")

    assert first != _source_code_sha256(tmp_path)
