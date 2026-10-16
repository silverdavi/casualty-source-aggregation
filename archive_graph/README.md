# archive_graph — Archive Genocide event-graph companion (frozen-POPHM)

A self-contained companion module that ingests the
[Archive Genocide "Gaza Node Map / lab"](https://archivegenocide.com/lab/)
**event graph** (build `f76bd2fb`, pinned in `BUILD_ID` and
`ingest/PINNED_SOURCE.json`) and produces incident-level spatial,
protected-site, name-overlap, corroboration, and incident-size analyses that
independently ground the POPHM Gaza claims **without changing them**. The
POPHM manuscript, the `gaza_sim/` posterior, and
`data/per_war/israel_gaza_war_2023.json` are frozen; the bridge from these
analyses to the paper's claims is `report.md` in this directory.

## Ethics and anti-misuse rules

- **Event-graph CSVs only; no media downloaded.** The ~2 TB footage torrents
  are never fetched; the clip table is metadata.
- **No personal names in committed artefacts.** Raw CSVs (names, captions,
  clip metadata) live in `raw/` and are gitignored. Public outputs are
  count-only JSON aggregates and figures.
- **Not a death toll.** An event is a cluster of footage — an upper bound on
  distinct incidents. `killed_min`/`killed_max` are caption-stated cluster
  figures; summing them is forbidden and a unit test enforces it.
- Coordinates are gazetteer place anchors, not strike locations.
  `verification_tier` is corroboration inside this archive, not truth. Most
  dates are publication dates; analyses use stated dates, with a separate
  publication-date sensitivity. See `ingest/DATASET_CONSTRAINTS.md`.

## Licence and credit

Data © [Archive Genocide](https://archivegenocide.com/lab/), licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This module's code
follows the repository's CC-BY-4.0 licence. Build pinned to `f76bd2fb`;
ingest verifies the GPG signature (key `9B4D575499AFA53C`, fingerprint
`C24E C92B 12D6 670A 2516 065F 9B4D 5754 99AF A53C`) and all SHA256 sums.

## Reproduce

```bash
# 1. Download + verify (GPG + SHA256); raw files land in archive_graph/raw/ (gitignored)
bash archive_graph/ingest/download.sh

# 2. Schema audit → aggregates/schema_audit.json
python3 archive_graph/ingest/schema_audit.py

# 3. Analyses → aggregates/*.json + figures/*.png
python3 archive_graph/analysis/01_spatial.py
python3 archive_graph/analysis/02_protected_sites.py
python3 archive_graph/analysis/03_name_overlap.py
python3 archive_graph/analysis/04_corroboration.py
python3 archive_graph/analysis/05_incident_size.py

# 4. Tests (filters, sum-guard, aggregate invariants)
python3 -m pytest -q archive_graph/analysis archive_graph/ingest/test_schema_filters.py
```

## Layout

```
archive_graph/
├── README.md            this file
├── report.md            implications memo (claim-by-claim bridge to POPHM)
├── BUILD_ID             pinned dataset build (f76bd2fb)
├── ingest/
│   ├── download.sh          fetch + GPG/SHA256 verify (event graph only)
│   ├── schema_audit.py      row counts, field audits, filter tallies
│   ├── filters.py           analysis-ready filters (stated dates, mapped,
│   │                        non-cumulative killed; sum of killed_min forbidden)
│   ├── test_schema_filters.py
│   ├── PINNED_SOURCE.json   URLs, build id, key fingerprint
│   └── DATASET_CONSTRAINTS.md
├── analysis/
│   ├── common.py            shared loading/filtering
│   ├── 01_spatial.py        governorate × month vs PCBS population null
│   ├── 02_protected_sites.py facility strikes + victim-group mentions
│   ├── 03_name_overlap.py   MoH register overlap (counts only)
│   ├── 04_corroboration.py  tiers, archives × footage, relation grades
│   ├── 05_incident_size.py  cluster-size distribution (never a toll)
│   └── test_aggregates.py
├── aggregates/*.json    committed count-only outputs
├── figures/*.png        committed figures
└── raw/                 gitignored raw CSVs (names/captions stay local)
```

## Later, if POPHM asks (punch-list only — nothing here is implemented)

- Replace the simulator's homogeneous cells with governorate-month
  documentation weights as a **sensitivity** in a fork of `gaza_sim/` (never
  the frozen original).
- Add the name-overlap result as a qualitative paragraph in a revision, if
  requested.
- Second-paper thesis: "primary-footage event graphs as a finding aid for
  casualty identification" — not a new combatant-share estimate.
