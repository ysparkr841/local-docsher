from __future__ import annotations

from pathlib import Path
import zipfile


SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample_docs_hwp"


def _read_zip_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        return "\n".join(
            archive.read(name).decode("utf-8")
            for name in archive.namelist()
            if name.lower().endswith(".xml")
        )


def test_committed_hwpx_sample_fixtures_are_valid_zip_xml_with_unique_tokens() -> None:
    simple = SAMPLE_DIR / "simple.hwpx"
    table = SAMPLE_DIR / "table.hwpx"

    assert simple.is_file()
    assert table.is_file()
    assert zipfile.is_zipfile(simple)
    assert zipfile.is_zipfile(table)
    assert "문서셔HWPX단순토큰" in _read_zip_text(simple)
    assert "문서셔HWPX표토큰" in _read_zip_text(table)


def test_hwp_binary_sample_has_manifest_instead_of_unproven_binary() -> None:
    manifest = SAMPLE_DIR / "simple.hwp.manifest.md"

    assert not (SAMPLE_DIR / "simple.hwp").exists()
    assert manifest.is_file()
    manifest_text = manifest.read_text(encoding="utf-8")
    assert "문서셔HWP단순토큰" in manifest_text
    assert "redistribution" in manifest_text
    assert "SHA256" in manifest_text
