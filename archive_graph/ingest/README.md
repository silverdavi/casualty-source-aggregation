# Event-graph ingest (build `f76bd2fb`)

Download and verify the Archive Genocide **event graph** only. Do not fetch media torrents.

```bash
bash archive_graph/ingest/download.sh
python3 archive_graph/ingest/test_schema_filters.py
python3 archive_graph/ingest/schema_audit.py
```

Raw CSVs land in `archive_graph/raw/` and are gitignored (names, captions, clip metadata). Public output is count-only JSON under `archive_graph/aggregates/`. Never publish a sum of `killed_min`. See `DATASET_CONSTRAINTS.md`.
