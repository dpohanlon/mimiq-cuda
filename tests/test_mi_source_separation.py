from pathlib import Path


def test_cuda_source_is_unconditionally_mi_only():
    cpp = Path(__file__).resolve().parents[1] / "cpp"
    backend = cpp / "mi_cuda_backend.cu"
    assert backend.is_file()
    assert not (cpp / "cuda_backend.cu").exists()

    source = backend.read_text()
    assert "#ifdef" not in source
    assert "#endif" not in source
    assert "pidc" not in source.lower()
    assert "run_cuda_histogram_mi" in source
