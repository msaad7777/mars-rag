# mars-rag: rules for agents

A RAG assistant that answers questions about human Mars missions from 82 NASA technical reports, cites the
report and page for every answer, and declines when the reports don't hold the answer. Free local models only.

## Check command

```bash
uv sync --frozen && uv run pytest -q
```

This is what CI runs (`.github/workflows/ci.yml`). The tests need no PDFs, index or model. The full evaluation
(`uv run python eval/run_eval.py`) needs the downloaded reports, the index and Ollama; see the README.

- Report results only from files the eval wrote. The retrieval-gate threshold was set before the eval; don't
  retune it on the test set.
- `data/` downloads come only from NASA NTRS (free, public). Don't commit the PDFs or the index.
- This project has no AWS code. If any is added, the AWS rule below applies.

## Rules

- **Free services only.** Never call real AWS or any paid API, and never add a paid service, key or
  subscription. Models run locally (Ollama or open weights) and tests use fakes.
- **AWS goes to a local mock.** For anything AWS-backed, use the free local
  [AWS Emulator](https://github.com/aws/aws-emulator) or [Moto](https://github.com/getmoto/moto), with
  `AWS_ENDPOINT_URL` pointing at `127.0.0.1`. Never use real AWS credentials or profiles.
- **No IAM-enforcement claims.** The emulator doesn't evaluate IAM policies (and Moto doesn't by default), so a
  call that succeeds locally proves nothing about permissions. Don't write "least privilege enforced" or
  "access denied as expected" unless it's checked another way, and say so.
- **No secrets or personal data.** No keys, tokens, `.env` files, real names, emails, addresses or account
  numbers in code, data, docs or commit messages. Fixtures and demo data are fictional.
- **Report what was run.** Numbers in the README come from committed result files. If you didn't run
  something, say so instead of guessing.
- Keep changes small and reviewable, one branch and pull request per change, and don't merge your own PR.
