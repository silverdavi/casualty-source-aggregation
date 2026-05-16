# Population Health Metrics submission package

**Target:** *Population Health Metrics* (BioMed Central / Springer Nature, open access).

## Files

| File | Purpose |
|---|---|
| `main.tex` | Manuscript (built on shared `../../content/*.tex` + `../../content/refs.bib`). Includes BMC structured abstract. |
| `cover_letter.tex` | Cover letter, framing the work as a complement to Gómez-Ugarte et al. 2025 in the same journal. |
| `README.md` | This file. |

## Build

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
pdflatex cover_letter
```

## Submission checklist (Pop Health Metrics / BMC)

1. **Class file.** Replace `article` with the BMC `bmcart` class:
   - Download `bmcart.cls` and `biomed_central_bib.bst` from [BMC author resources](https://www.biomedcentral.com/getpublished/writing-resources/instructions-for-authors-bmc-journals).
   - Use `\documentclass{bmcart}` per the BMC skeleton.
   - Switch `\bibliographystyle` to `bmc-mathphys` or `vancouver`.
2. **Structured abstract.** Already in place: Background / Methods / Results / Conclusions / Keywords (per BMC style).
3. **Submission portal.** [Editorial Manager](https://www.editorialmanager.com/popm/).
4. **Open access fee.** Standard BMC APC applies; check institutional waivers.
5. **Collection.** Target the open call **"The health impacts of war and armed conflict"** (active 2025–).
6. **Declarations.** Already in place: Ethics / Consent / Data availability / Competing interests / Funding / Authors' contributions / Acknowledgements.

## Key citations to highlight in cover letter

- Gómez-Ugarte et al. 2025 *Pop Health Metrics* (Gaza Bayesian, complementary).
- Spagat et al. 2026 *Lancet Global Health* (Gaza field survey).
- Khatib et al. 2024, 2025 *Lancet* (Gaza correspondence + capture–recapture).
- Spiegel & Garry 2024 *Lancet* (Counting the dead in Gaza: not impossible, but expensive).
- Burnham et al. 2006 *Lancet* (Iraq cluster survey).
- Checchi & Roberts 2008 *PLOS Medicine* (Documenting mortality in crises).

## Possible reviewers (no conflict)

Diego Alburez-Gutierrez, Ana C. Gómez-Ugarte, Enrique Acosta, Michael Spagat, Patrick Ball, Megan Price, Francesco Checchi.
