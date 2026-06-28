# Local Docsher

Local Docsher is a local-first document indexing and search project.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m docsher --help
```

## OCR backend

Local Docsher uses a pluggable OCR backend interface. The default production-oriented backend decision for MVP-B is **PaddleOCR** because it has Korean language support and can run offline when its model files are already present on disk. PaddleOCR ONNX remains a future optimization candidate after the baseline backend is proven.

Install the OCR extra when you want to run the real PaddleOCR backend:

```bash
python -m pip install -e ".[ocr]"
```

The OCR extra installs both `paddleocr` and `paddlepaddle`. On machines where `paddleocr` is installed without `paddlepaddle`, Docsher reports a clear backend-unavailable error.

For offline deployments, pre-download/copy PaddleOCR model directories and point Docsher at those local paths in the config:

```json
{
  "ocr": {
    "enabled": true,
    "backend": "paddle",
    "paddle": {
      "lang": "korean",
      "det_model_dir": "/models/paddleocr/korean/det",
      "rec_model_dir": "/models/paddleocr/korean/rec",
      "cls_model_dir": "/models/paddleocr/korean/cls",
      "use_angle_cls": true
    }
  }
}
```

Smoke-test an image with the configured or explicit PaddleOCR model paths:

```bash
python -m docsher ocr-test ./sample_docs/korean_image.png --backend paddle --paddle-lang korean \
  --paddle-det-model-dir /models/paddleocr/korean/det \
  --paddle-rec-model-dir /models/paddleocr/korean/rec \
  --paddle-cls-model-dir /models/paddleocr/korean/cls
```

If `paddleocr` is not installed, the command exits with a clear `PaddleOCR backend unavailable` error instead of failing with a Python traceback.
