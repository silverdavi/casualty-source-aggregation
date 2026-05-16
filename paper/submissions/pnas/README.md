# PNAS submission package

**Target:** *Proceedings of the National Academy of Sciences*, Direct Submission, Social Sciences track.

## Files

| File | Purpose |
|---|---|
| `main.tex` | Manuscript (built on shared `../../content/*.tex` + `../../content/refs.bib`). |
| `cover_letter.tex` | Cover letter to the editorial board. |
| `README.md` | This file. |

## Build

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
pdflatex cover_letter
```

## Submission checklist (PNAS)

1. **Class file.** This draft uses `article` to keep the master tree portable. Before submitting:
   - Download `pnas-new.cls`, `pnas.bst`, and the `pnas-sample.tex` skeleton from <https://www.pnas.org/page/authors/format>.
   - Replace the preamble of `main.tex` with the `pnas-sample.tex` preamble (using `\documentclass{pnas-new}`); keep the `\input{...}` calls and `\bibliography{...}` line unchanged.
   - Switch `\bibliographystyle` to `pnas`.
2. **Length.** Standard Research Report is 6 pages (~4,000 words, 50 references, 4 medium graphical elements); maximum 12 pages. The current main text is ~4,500 words and one figure; trim Section 7 (Empirical overview) and move it to SI if needed.
3. **Significance Statement.** Already in `main.tex` (120 words, on page 1).
4. **Data deposit.** All Gaza-application inputs and the spatial Bayesian simulator code must be deposited in a public repository before submission; replace the placeholder URL in `cover_letter.tex` and `08_discussion.tex`.
5. **Suggested editors.** Cover letter names: Charles Manski (Northwestern), Gary King (Harvard), James Berger (Duke), Håvard Hegre (Uppsala).
6. **Suggested reviewers (non-conflict).** Michael Spagat, Patrick Ball, Megan Price, Diego Alburez-Gutierrez.
7. **Acceptance rate ≈ 14%.** First decision in 2–4 weeks.

## Key citations to highlight in cover letter

- Radford et al. 2023 PNAS (closest precedent, Russia–Ukraine).
- Alburez-Gutierrez et al. 2024 Science Advances (family bereavement, demographic methodology).
- Gómez-Ugarte et al. 2025 Pop Health Metrics (Gaza Bayesian, complementary).
- Spagat et al. 2026 Lancet Global Health (Gaza field survey).
