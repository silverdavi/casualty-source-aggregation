# JPR submission package

**Target:** *Journal of Peace Research* (Oxford University Press / PRIO; online-only since Jan 2026).

## Files

| File | Purpose |
|---|---|
| `main.tex` | Manuscript (built on shared `../../content/*.tex` + `../../content/refs.bib`). |
| `cover_letter.tex` | Cover letter. |
| `README.md` | This file. |

## Build

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
pdflatex cover_letter
```

## Submission checklist (JPR / OUP)

1. **Class file.** Replace `article` with the OUP authoring template:
   - Download from [OUP author guidance](https://academic.oup.com/journals/pages/authoring/journals_in_two_columns/preparing_your_manuscript).
   - Use the single-column variant (JPR is single-column-typeset post-OUP transition).
2. **Length.** JPR Regular Article: typically up to ~10,000 words *including* references, footnotes, and supplementary material captions; supplements free. Current main text is ~7,500 words.
3. **Submission portal.** ScholarOne Manuscripts. JPR transitioned from SAGE to OUP in January 2026; the portal moved with it.
4. **Anonymous submission.** Author details on a separate title page; main file is fully anonymous.
5. **Replication data policy.** JPR requires replication code/data deposit at acceptance (see `cover_letter.tex` for the commitment).

## Key citations to highlight in cover letter

- Sundberg & Melander 2013 (UCDP-GED) and Raleigh, Linke, Hegre, Karlsen 2010 (ACLED).
- Eck & Hultman 2007 JPR (one-sided violence dataset).
- Lacina & Gleditsch 2005 (battle-deaths data).
- Vesco et al. 2026 JCR (concurrent UCDP underreporting reassessment).
- VIEWS Prediction Challenge papers, JPR vol. 62 (2025).

## Possible reviewers (no conflict)

Michael Spagat, Patrick Ball, Lisa Hultman, Diego Alburez-Gutierrez, Erik Melander, Håvard Hegre.
