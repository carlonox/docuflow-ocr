# Security

This document exists because the framework's origin project made every
mistake on this page — including committing a Google Cloud service-account
key to a repository. The key was disabled by Google, the project was
scheduled for deletion, and the lessons are documented here so the next
person doesn't repeat them.

## Credentials

### The golden rule

**Never commit credentials.** No service-account JSON, no API keys, no
`.env` files, no secrets of any kind.

### The right way

1. Store the service account key outside the repository.
2. Point to it with an environment variable:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/home/user/keys/vision-service-account.json
```

3. Use `.env.example` (committed) for *placeholders*, and a real `.env`
   (gitignored) for actual values.

### Rotation and response

If a key ever leaks:

1. **Rotate immediately**: disable/delete the key in the provider console.
   Every second it stays live it can be abused.
2. **Audit usage**: check API logs for calls you did not make (Cloud
   Console → APIs → Metrics or the provider's equivalent).
3. **Set budget alerts**: configure billing alerts at a low threshold so a
   compromised key triggers an alarm before it racks up a bill.
4. **Remove it from history**: a deleted key in git history is still a
   liability — rewrite history or disable the key; disabling is the only
   real fix.
5. **Document the incident**: write down what happened and how to prevent it,
   like this file.

### Local OCR as a security feature

DocuFlow's local providers (Surya, RapidOCR, Tesseract, Ollama) never send
document data anywhere. For sensitive documents — medical records, legal
files, personal IDs — the cascade can run entirely offline. This is a
feature: some documents should never leave the machine.

## Document data

- **Never commit real documents** to the repository: no scans, no photos, no
  `temp_*.png` from processing runs.
- **Synthetic fixtures only**: use `scripts/generate_sample_docs.py` to
  create test documents. Real names, real ID numbers, real anything — no.
- **Learned state can leak information**: `ocr_config_learned.json`,
  `error_patterns.json` and friends may contain patterns derived from real
  documents (e.g. actual digit sequences seen in the wild). Before publishing
  or sharing, regenerate them on synthetic data or scrub them.

## Repository hygiene

- `.gitignore` covers credentials, learned state, real images and temporary
  files (see the committed `.gitignore`).
- `git-secrets` (or a pre-commit hook) blocks accidental credential commits:

```bash
pip install git-secrets
git secrets --install
git secrets --register-aws  # plus your own patterns
```

- Before any push, run the scan:

```bash
grep -rniE "api[_-]?key|password|secret|token|credential" . --exclude-dir=.git | grep -v example
```

- Check for oversized files before committing: `find . -size +1M` — a stray
  scan can hide in there.

## Environment variables used by DocuFlow

| Variable | Purpose |
|---|---|
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to the service account JSON (Vision, Sheets) |
| `SHEET_ID` / `SHEET_NAME` | Google Sheets target (optional) |
| `OLLAMA_HOST` / `OLLAMA_MODEL` | Local Ollama endpoint and model |
| `LLM_CORRECTOR_MODEL` | Local correction model (OpenAI-compatible) |
| `LOG_LEVEL` | Logging verbosity |

## Reporting a vulnerability

Open an issue with the tag `security` (or contact the maintainer directly
for sensitive findings). Please include: affected version, impact, and a
minimal reproduction.
