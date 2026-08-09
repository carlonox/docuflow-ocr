# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial public release of the framework extracted from a private,
  battle-tested document processing system.
- Cascade OCR pipeline (fast → accurate → LLM correction → validation).
- Self-learning engine with per-field parameter tuning.
- Advanced learning from manual corrections.
- Precision monitor with sliding-window drift alerts.
- Resumable state machine with checkpoints.
- Providers: Surya OCR 2, RapidOCR (ONNX), Google Cloud Vision, Ollama GLM,
  Tesseract.
- Synthetic fixture generator and unit tests.
- **Docker support**: multi-stage Dockerfile with build-arg-controlled provider
  installation, Docker Compose for development, and comprehensive deployment
  guide (docs/DEPLOY.md).
