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

## Experimental Unlimited-OCR backend

`baidu/Unlimited-OCR` is exposed as an **experimental** backend through its documented SGLang OpenAI-compatible server mode. It is not the default backend because the upstream README targets Python 3.12, CUDA 12.9, recent PyTorch/Transformers, long context generation, and a separate local server process. On small/offline machines, PaddleOCR remains the baseline OCR backend.

If you already have Unlimited-OCR running locally, Docsher can call it with:

```bash
python -m docsher ocr-test ./sample_docs/korean_image.png --backend unlimited \
  --unlimited-endpoint http://127.0.0.1:10000/v1/chat/completions \
  --unlimited-model Unlimited-OCR
```

Equivalent config shape:

```json
{
  "ocr": {
    "backend": "unlimited",
    "unlimited": {
      "endpoint": "http://127.0.0.1:10000/v1/chat/completions",
      "model": "Unlimited-OCR",
      "prompt": "document parsing.",
      "timeout_seconds": 1200
    }
  }
}
```

If the local SGLang server is not running, Docsher returns a clear `Unlimited-OCR backend unavailable` error.
