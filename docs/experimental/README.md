# Experimental and secondary tracks

These documents support work that is intentionally outside Heirloom's primary tensor/autograd → CUDA → memory-transformer review path.

- `qb/` covers governed-corpus, tokenizer, pretraining-readiness, and cloud infrastructure experiments.
- `padawan/` covers the Padawan agent/verifier workflow.

They are retained to preserve engineering provenance. Some commands require external data, cloud accounts, or private infrastructure. The default validation gate includes local readiness and Padawan fixture checks in [`scripts/validate.sh`](../../scripts/validate.sh#L19-L23); passing those checks does not reproduce the external experiments.
