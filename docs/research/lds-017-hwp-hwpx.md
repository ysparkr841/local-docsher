# LDS-017 HWP/HWPX parser adapter and conversion tool research

LDS-017 records the HWP/HWPX extraction decision and prepares an isolated adapter contract. It deliberately does not integrate HWP/HWPX parsing into the scanner or indexer; that work is deferred to LDS-018 and later implementation issues.

## Goals and acceptance coverage

- Research candidate tools for `.hwp` and `.hwpx` handling.
- Identify candidates usable offline on Windows.
- Record license and deployment risk.
- Decide how the parser adapter interface connects to future implementation.
- Prepare the minimum sample-document policy.

## Candidate matrix

| Candidate | Formats | Windows offline usability | License/deployment notes | Assessment |
| --- | --- | --- | --- | --- |
| Built-in Python stdlib HWPX ZIP/XML baseline | `.hwpx` only | Strong. HWPX is ZIP/XML and can be read with `zipfile` and `xml.etree.ElementTree` without new dependencies. | Lowest deployment risk; no new runtime license. Limited to simple text extraction and must tolerate HWPX schema variation. | Selected for LDS-018 baseline simple HWPX extraction. |
| Node sidecar using `@ssabrojs/hwpxjs` | `.hwp` HWP5 and `.hwpx` | Good if Node package is vendored/installed by user and can run without network after installation. Works as an external sidecar, keeping Python base dependencies unchanged. | Need to track npm package license and transitive dependencies before bundling. Operational risk: Node runtime, sidecar process, JSON contract, timeout/error handling. | Selected as primary optional external backend candidate. |
| `kordoc` | HWP-oriented conversion/extraction, useful for older/difficult files | Potentially usable offline on Windows if packaged or user-installed, but requires validation against target files. | License and binary/runtime packaging must be reviewed before redistribution. Treat as optional fallback, not a default dependency. | Optional fallback candidate for HWP3 or difficult files. |
| `pyhwp` | `.hwp` | Offline possible, but project age and Python compatibility are concerns. | AGPLv3+ creates strong copyleft obligations unsuitable as a default embedded parser for this proprietary project. Old Python support increases maintenance risk. | Rejected as default backend. Could only be considered as a user-configured external tool with explicit license warning. |
| LibreOffice conversion | Varies by installed filters; may support HWP/HWPX depending on build | User-installed desktop/server package can be used offline. Windows command-line automation is possible but fragile. | Large install, version-dependent results, filter availability varies, deployment and support burden. License generally acceptable for user-installed optional tooling, but not bundled. | User-configured optional converter only. |
| Hancom Office / Hanword automation | Native HWP/HWPX | Best fidelity when installed by user; offline desktop use possible. | Proprietary commercial software; automation interfaces and redistribution are not suitable for bundled support. Requires user license and local configuration. | User-configured optional converter only. |

## Decision

Use a layered strategy:

1. Implement a built-in stdlib `.hwpx` ZIP/XML baseline for simple text in LDS-018.
2. Define an optional external Node sidecar backend using `@ssabrojs/hwpxjs` as the primary external candidate for `.hwp` HWP5 and `.hwpx`.
3. Keep `kordoc` as an optional fallback for HWP3 or files that fail the primary backend, pending hands-on validation and license review.
4. Do not use `pyhwp` as a default because of AGPLv3+ licensing and old Python-support concerns.
5. Treat LibreOffice and Hancom Office/Hanword as user-configured optional conversion tools only, not bundled dependencies.

The LDS-017 code artifact is therefore only an adapter contract in `docsher.parsers_hwp`, not scanner/indexer integration.

## Adapter interface connection approach

`src/docsher/parsers_hwp.py` defines the public contract:

- supported extensions: `.hwp`, `.hwpx`;
- shared parser name: `hwp`;
- typed parsed document and segment dataclasses;
- common exceptions for unsupported extension, unavailable tooling, and parse failures;
- `HwpTextExtractor` protocol with `name`, `is_available()`, and `extract(path)`;
- `parse_hwp_document(path, extractor=None)` as the future integration point.

Future extractor implementations should live behind this protocol. The scanner/indexer should only be wired after a real extractor is available and tested. External tools should communicate through a small, deterministic contract such as JSON records containing text segments and optional metadata, with explicit timeout, stderr, and exit-code handling.

## LDS-018 implementation plan

Recommended next steps for LDS-018:

1. Generate synthetic HWPX ZIP/XML fixtures in tests.
2. Implement a stdlib-only HWPX extractor for simple paragraph/table text.
3. Return `ParsedHwpDocument` and `ParsedHwpSegment` values through the existing adapter contract.
4. Add parser-level tests for simple text, table text, malformed ZIP/XML, and unsupported schema gaps.
5. Only after parser behavior is stable, plan a separate scanner/indexer wiring change.

## Sample policy

The minimum sample set is recorded in `sample_docs_hwp/README.md` and includes provenance-safe committed HWPX fixtures:

- `simple.hwpx` with unique Korean token `문서셔HWPX단순토큰`;
- `table.hwpx` with unique Korean token `문서셔HWPX표토큰`;
- `simple.hwp.manifest.md`, documenting the required future/user-provided HWP5 sample with unique Korean token `문서셔HWP단순토큰`.

The committed `.hwpx` files are synthetic ZIP/XML fixtures generated for this project with Python stdlib `zipfile` and contain only synthetic text. A real binary `simple.hwp` is not bundled in LDS-017 because redistribution provenance has not been established. Before committing binary documents, record provenance, content ownership, redistribution permission, generator tool/version, checksum, and confirmation that no private or third-party content is included.
