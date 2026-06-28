"""Office document parsers for PDF/DOCX/PPTX/XLSX files.

The MVP implementation intentionally avoids external runtime dependencies by
parsing the small subset of text-layer structures needed for local indexing:
OOXML files are ZIP containers with XML payloads, and PDFs are scanned for text
showing operators in content streams.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

SUPPORTED_OFFICE_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".docx", ".pptx", ".xlsx"})
PARSER_NAME = "office"


class OfficeParserError(Exception):
    """Base class for office parser failures."""


class UnsupportedOfficeExtensionError(OfficeParserError):
    """Raised when an unsupported extension is parsed by the office parser."""


class OfficeParseError(OfficeParserError):
    """Raised when a supported office document cannot be parsed."""


@dataclass(frozen=True)
class ParsedOfficeSegment:
    """A location-preserving parsed text segment."""

    text: str
    page_number: int | None = None
    slide_number: int | None = None
    sheet_name: str | None = None
    section_title: str | None = None


@dataclass(frozen=True)
class ParsedOfficeDocument:
    """Parsed office document payload."""

    segments: tuple[ParsedOfficeSegment, ...]
    parser_name: str = PARSER_NAME


_XML_TEXT_NAMES = {"t"}
_SPREADSHEET_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
_OFFICE_REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def is_office_document(path: str | Path) -> bool:
    """Return true when ``path`` has an office-parser-supported extension."""

    return Path(path).suffix.lower() in SUPPORTED_OFFICE_EXTENSIONS


def parse_office_document(path: str | Path) -> ParsedOfficeDocument:
    """Parse supported PDF/DOCX/PPTX/XLSX documents into text segments."""

    document_path = Path(path)
    extension = document_path.suffix.lower()
    try:
        if extension == ".pdf":
            return ParsedOfficeDocument(_parse_pdf(document_path))
        if extension == ".docx":
            return ParsedOfficeDocument(_parse_docx(document_path))
        if extension == ".pptx":
            return ParsedOfficeDocument(_parse_pptx(document_path))
        if extension == ".xlsx":
            return ParsedOfficeDocument(_parse_xlsx(document_path))
    except OfficeParserError:
        raise
    except (OSError, KeyError, zipfile.BadZipFile, ET.ParseError, UnicodeDecodeError) as exc:
        raise OfficeParseError(f"Could not parse {document_path.name}: {exc}") from exc
    raise UnsupportedOfficeExtensionError(f"Unsupported office document extension: {extension}")


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _xml_text_from_bytes(payload: bytes) -> str:
    root = ET.fromstring(payload)
    values: list[str] = []
    for element in root.iter():
        if _local_name(element.tag) in _XML_TEXT_NAMES and element.text:
            values.append(element.text)
    return "\n".join(value.strip() for value in values if value.strip())


def _parse_docx(path: Path) -> tuple[ParsedOfficeSegment, ...]:
    with zipfile.ZipFile(path) as archive:
        text = _xml_text_from_bytes(archive.read("word/document.xml"))
    return (ParsedOfficeSegment(text=text),) if text else ()


def _slide_number_from_name(name: str) -> int:
    match = re.search(r"slide(\d+)\.xml$", name)
    if not match:
        return 0
    return int(match.group(1))


def _parse_pptx(path: Path) -> tuple[ParsedOfficeSegment, ...]:
    segments: list[ParsedOfficeSegment] = []
    with zipfile.ZipFile(path) as archive:
        slide_names = sorted(
            (name for name in archive.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", name)),
            key=_slide_number_from_name,
        )
        for slide_name in slide_names:
            text = _xml_text_from_bytes(archive.read(slide_name))
            if text:
                segments.append(
                    ParsedOfficeSegment(
                        text=text,
                        slide_number=_slide_number_from_name(slide_name),
                    )
                )
    return tuple(segments)


def _read_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        payload = archive.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(payload)
    strings: list[str] = []
    for si in root.findall(f"{_SPREADSHEET_NS}si"):
        parts = [node.text or "" for node in si.iter(f"{_SPREADSHEET_NS}t")]
        strings.append("".join(parts))
    return strings


def _read_workbook_sheet_names(archive: zipfile.ZipFile) -> dict[str, str]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels: dict[str, str] = {}
    try:
        rel_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        for rel in rel_root:
            rel_id = rel.attrib.get("Id")
            target = rel.attrib.get("Target", "")
            if rel_id and target:
                rels[rel_id] = str(PurePosixPath("xl") / target.lstrip("/"))
    except KeyError:
        pass

    names: dict[str, str] = {}
    for sheet in workbook.iter(f"{_SPREADSHEET_NS}sheet"):
        name = sheet.attrib.get("name", "Sheet")
        rel_id = sheet.attrib.get(f"{_OFFICE_REL_NS}id")
        if rel_id and rel_id in rels:
            names[rels[rel_id]] = name
        else:
            sheet_id = sheet.attrib.get("sheetId")
            if sheet_id:
                names[f"xl/worksheets/sheet{sheet_id}.xml"] = name
    return names


def _cell_text(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.iter(f"{_SPREADSHEET_NS}t")).strip()
    value = cell.find(f"{_SPREADSHEET_NS}v")
    if value is None or value.text is None:
        return ""
    if cell_type == "s":
        try:
            return shared_strings[int(value.text)].strip()
        except (ValueError, IndexError):
            return ""
    return value.text.strip()


def _parse_xlsx(path: Path) -> tuple[ParsedOfficeSegment, ...]:
    segments: list[ParsedOfficeSegment] = []
    with zipfile.ZipFile(path) as archive:
        shared_strings = _read_shared_strings(archive)
        sheet_names = _read_workbook_sheet_names(archive)
        worksheet_paths = sorted(
            name for name in archive.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml$", name)
        )
        for worksheet_path in worksheet_paths:
            root = ET.fromstring(archive.read(worksheet_path))
            values: list[str] = []
            for cell in root.iter(f"{_SPREADSHEET_NS}c"):
                text = _cell_text(cell, shared_strings)
                if text:
                    values.append(text)
            if values:
                segments.append(
                    ParsedOfficeSegment(
                        text="\n".join(values),
                        sheet_name=sheet_names.get(worksheet_path, Path(worksheet_path).stem),
                    )
                )
    return tuple(segments)


def _decode_pdf_literal(raw: bytes) -> str:
    out: list[str] = []
    index = 0
    while index < len(raw):
        char = raw[index]
        if char == 0x5C and index + 1 < len(raw):  # backslash
            index += 1
            escaped = raw[index]
            escapes = {
                ord("n"): "\n",
                ord("r"): "\r",
                ord("t"): "\t",
                ord("b"): "\b",
                ord("f"): "\f",
                ord("("): "(",
                ord(")"): ")",
                ord("\\"): "\\",
            }
            if escaped in escapes:
                out.append(escapes[escaped])
            elif 48 <= escaped <= 55:
                octal = bytes([escaped])
                consumed = 0
                while index + 1 < len(raw) and consumed < 2 and 48 <= raw[index + 1] <= 55:
                    index += 1
                    consumed += 1
                    octal += bytes([raw[index]])
                out.append(chr(int(octal, 8)))
            else:
                out.append(chr(escaped))
        else:
            out.append(chr(char))
        index += 1
    return "".join(out)


def _extract_pdf_literals(content: bytes) -> list[str]:
    literals: list[str] = []
    index = 0
    while index < len(content):
        if content[index] != ord("("):
            index += 1
            continue
        index += 1
        depth = 1
        literal = bytearray()
        while index < len(content) and depth > 0:
            byte = content[index]
            if byte == 0x5C and index + 1 < len(content):
                literal.append(byte)
                index += 1
                literal.append(content[index])
            elif byte == ord("("):
                depth += 1
                literal.append(byte)
            elif byte == ord(")"):
                depth -= 1
                if depth > 0:
                    literal.append(byte)
            else:
                literal.append(byte)
            index += 1
        if literal:
            text = _decode_pdf_literal(bytes(literal)).strip()
            if text:
                literals.append(text)
    return literals


def _parse_pdf_objects(data: bytes) -> dict[int, bytes]:
    return {
        int(match.group(1)): match.group(2)
        for match in re.finditer(rb"(\d+)\s+0\s+obj\b(.*?)\bendobj", data, flags=re.S)
    }


def _pdf_content_refs(page_object: bytes) -> list[int]:
    contents = re.search(rb"/Contents\s+(\[(.*?)\]|(\d+)\s+0\s+R)", page_object, flags=re.S)
    if not contents:
        return []
    if contents.group(2) is not None:
        return [int(ref) for ref in re.findall(rb"(\d+)\s+0\s+R", contents.group(2))]
    if contents.group(3) is not None:
        return [int(contents.group(3))]
    return []


def _pdf_page_refs_in_tree(objects: dict[int, bytes]) -> list[int]:
    catalog_ref = next(
        (
            object_id
            for object_id, body in objects.items()
            if re.search(rb"/Type\s*/Catalog\b", body)
        ),
        None,
    )
    if catalog_ref is None:
        return []
    catalog = objects[catalog_ref]
    pages_match = re.search(rb"/Pages\s+(\d+)\s+0\s+R", catalog)
    if not pages_match:
        return []

    seen: set[int] = set()

    def walk(ref: int) -> list[int]:
        if ref in seen:
            return []
        seen.add(ref)
        body = objects.get(ref, b"")
        if re.search(rb"/Type\s*/Page\b", body) and not re.search(rb"/Type\s*/Pages\b", body):
            return [ref]
        kids_match = re.search(rb"/Kids\s*\[(.*?)\]", body, flags=re.S)
        if not kids_match:
            return []
        ordered: list[int] = []
        for child_ref in re.findall(rb"(\d+)\s+0\s+R", kids_match.group(1)):
            ordered.extend(walk(int(child_ref)))
        return ordered

    return walk(int(pages_match.group(1)))


def _pdf_stream_payload(object_body: bytes) -> bytes:
    stream_match = re.search(rb"stream\r?\n(.*?)\r?\nendstream", object_body, flags=re.S)
    if not stream_match:
        return b""
    payload = stream_match.group(1)
    if re.search(rb"/Filter\s*/FlateDecode\b", object_body):
        try:
            return zlib.decompress(payload)
        except zlib.error as exc:
            raise OfficeParseError(f"Could not decompress PDF FlateDecode stream: {exc}") from exc
    return payload


def count_pdf_pages(path: str | Path) -> int:
    """Return a best-effort page count for a PDF without external dependencies."""

    document_path = Path(path)
    data = document_path.read_bytes()
    if not data.startswith(b"%PDF"):
        raise OfficeParseError("Not a PDF file")
    objects = _parse_pdf_objects(data)
    page_refs = _pdf_page_refs_in_tree(objects)
    if page_refs:
        return len(page_refs)
    return sum(
        1
        for body in objects.values()
        if re.search(rb"/Type\s*/Page\b", body) and not re.search(rb"/Type\s*/Pages\b", body)
    )


def _parse_pdf(path: Path) -> tuple[ParsedOfficeSegment, ...]:
    data = path.read_bytes()
    if not data.startswith(b"%PDF"):
        raise OfficeParseError("Not a PDF file")

    objects = _parse_pdf_objects(data)
    page_refs = _pdf_page_refs_in_tree(objects)
    if not page_refs:
        page_refs = [
            object_id
            for object_id, body in objects.items()
            if re.search(rb"/Type\s*/Page\b", body) and not re.search(rb"/Type\s*/Pages\b", body)
        ]
    segments: list[ParsedOfficeSegment] = []
    for page_number, page_ref in enumerate(page_refs, start=1):
        page_body = objects.get(page_ref, b"")
        page_texts: list[str] = []
        for content_ref in _pdf_content_refs(page_body):
            content_body = objects.get(content_ref, b"")
            payload = _pdf_stream_payload(content_body)
            page_texts.extend(_extract_pdf_literals(payload))
        text = "\n".join(page_texts).strip()
        if text:
            segments.append(ParsedOfficeSegment(text=text, page_number=page_number))

    if segments:
        return tuple(segments)

    # Fallback for tiny/non-standard fixtures that contain text in raw streams
    # but omit a parseable page tree. This is intentionally secondary so real
    # page objects drive page_number preservation.
    for page_number, stream in enumerate(re.findall(rb"stream\r?\n(.*?)\r?\nendstream", data, flags=re.S), start=1):
        text = "\n".join(_extract_pdf_literals(stream)).strip()
        if text:
            segments.append(ParsedOfficeSegment(text=text, page_number=page_number))
    return tuple(segments)
