# Event-graph method constraints (build `f76bd2fb`)

Source: Archive Genocide lab README (`https://archivegenocide.com/lab/data/README.txt`), CC BY 4.0.
Credit Archive Genocide. These rules are method constraints, not optional footnotes.

1. An event is a **cluster of footage**, not a verified incident. The event count is an **upper bound** on distinct incidents.
2. Coordinates are **gazetteer place anchors**, not surveyed strike points. Inferred locations are absent from GeoJSON.
3. Most dates are publication dates. Time series use `date_source == stated`. Do not mix `post_proxy` silently.
4. `killed_min` / `killed_max` are caption-stated figures for that cluster. **Never sum `killed_min` as a death toll** (duplicates + over-split footage). War-to-date totals live in `cumulative_min` / `cumulative_max`.
5. `verification_tier` is corroboration **inside this archive**, not truth.
6. `people.in_victims_list` is an unambiguous match to the ministry identified-dead register; `victim_ambiguous` is a lead, not a confirmation.
7. The lab README describes link `grade=established|possible`. In build `f76bd2fb`, `event_relations.csv` carries `kind` and `why` instead; treat `kind=possible_same_event` as the possible-link analogue.

Public artefacts from this companion may include counts and histograms. They must not include victim-name dumps, clip captions, or any published sum of `killed_min`.
