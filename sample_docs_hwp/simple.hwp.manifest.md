# simple.hwp external sample manifest

`simple.hwp` is intentionally **not committed** in LDS-017 because redistributable HWP5 binary sample provenance has not been established.

For local/manual validation, create or provide a synthetic HWP5 file at this path:

```text
sample_docs_hwp/simple.hwp
```

Required contents:

```text
문서셔HWP단순토큰
```

Before committing a real `.hwp` binary, record:

- generator tool and version;
- owner/author;
- redistribution license or permission;
- confirmation that the file contains only synthetic text and no personal, proprietary, or third-party content;
- SHA256 checksum.

LDS-018 may use this manifest to skip binary HWP integration tests unless a local `simple.hwp` is present. The committed HWPX fixtures cover the redistributable baseline sample set.
