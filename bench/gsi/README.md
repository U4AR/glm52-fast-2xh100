# GSI fidelity/performance matrix

Use separate baseline and GSI servers so CUDA graphs never contain an on/off
branch. Run the harness once with MTP enabled and once with MTP disabled; each
run exercises both `-top2` and `-top8`.

```bash
python bench/gsi/run_matrix.py \
  --baseline-url http://baseline:8000 \
  --gsi-url http://candidate:8000 \
  --runs 30 \
  --out gsi_reports/mtp-on.json
```

The command fails unless deterministic answer content is identical. The report
retains reasoning equality, latency, throughput, and any server metadata for the
MTP-acceptance analysis.

On a TP2 host where baseline and candidate cannot run simultaneously, record
each fixed-seed launch with `capture_endpoint.py`, then compare the JSON reports
with `compare_reports.py`.
