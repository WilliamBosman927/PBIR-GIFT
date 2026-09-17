# Citation audit for `references.bib`

## Scope and policy

- The file contains exactly 42 entries with stable keys `ref1`--`ref42`, matching the numbering in `PBIR_GIFT_sections_CLEAN.md`.
- Bibliographic facts were retained from the manuscript only when internally consistent. Metadata was enriched only from an official publisher, official proceedings page, institutional repository, or the cited arXiv record.
- No DOI was guessed. DOI URLs appear only when the DOI was already present in the manuscript or was independently confirmed from an official/primary metadata page.
- `and others` is used where the manuscript supplied only an `et al.` author list and the full list was not independently verified. These entries are valid BibTeX but should be expanded before final submission if the journal requires all authors.
- Springer Nature's `sn-basic` bibliography style can consume these BibTeX entry types and keys. The manuscript must cite them as `\cite{ref1}`, ..., `\cite{ref42}` rather than keeping a hand-written numbered reference list.

## Verified enrichments and corrections

| Ref. | Verification result | Evidence |
|---|---|---|
| 3 | IJCAI title, five authors, pages, and DOI confirmed. | [IJCAI proceedings record](https://www.ijcai.org/proceedings/2020/641) |
| 4 | AAAI volume, issue, pages, and DOI confirmed. | [AAAI article record](https://ojs.aaai.org/index.php/AAAI/article/view/16144) |
| 5 | IJCAI title, five authors, pages, and DOI confirmed. | [IJCAI proceedings record](https://www.ijcai.org/proceedings/2022/557) |
| 6 | Full page/article span and DOI added from an indexed proceedings record. | [DBLP record](https://dblp.org/rec/journals/tist/SunW023.html) |
| 7 | DOI confirmed against the publisher record; volume and pages agree with the manuscript. | [Wiley article record](https://onlinelibrary.wiley.com/doi/10.1111/mafi.12382) |
| 9 | Full author list, NeurIPS volume, and DOI confirmed from the official NeurIPS record/PDF. | [NeurIPS record](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0bf54b80686d2c4dc0808c2e98d430f7-Abstract.html) |
| 10 | Author order corrected to the official record; article span and DOI added. | [Warwick Research Archive record](https://wrap.warwick.ac.uk/id/eprint/190962/) |
| 24 | Full authors confirmed. The arXiv submission is dated 2025, so the manuscript's year 2026 was corrected to 2025. | [arXiv record](https://arxiv.org/abs/2510.02209) |
| 29 | Full author list and 2026 arXiv record confirmed. | [arXiv record](https://arxiv.org/abs/2606.08450) |
| 32 | Authors, venue, volume, issue, and DOI confirmed from the official AAAI page. | [AAAI article record](https://ojs.aaai.org/index.php/AAAI/article/view/11694) |
| 33 | Full author list and NeurIPS volume confirmed from the official proceedings record. | [NeurIPS record](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html) |
| 39 | Authors, volume, pages, year, and DOI confirmed from the publisher page. | [Wiley article record](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1992.tb04681.x) |
| 42 | DOI added after confirmation from the official publisher page. | [Portfolio Management Research record](https://www.pm-research.com/content/iijpormgmt/40/5/94) |

## Incomplete or not independently verified

These are not necessarily wrong; they are fields that should be checked before submission.

| Ref. | Status | Required follow-up |
|---|---|---|
| 1 | Article metadata comes from the manuscript; DOI was not confirmed in this pass. | Verify against IEEE Xplore and add DOI if the record agrees. |
| 2, 8, 11, 18, 20, 25--27, 31 | arXiv identifiers were retained, but every record was not individually re-opened in this pass. | Batch-validate IDs and author lists against arXiv. |
| 12 | Journal, volume, issue, and pages were retained; the journal DOI was not confirmed. | Check the Springer article page and add the publisher DOI. The arXiv version is 2406.08013, but it was not substituted for the journal record. |
| 13--16, 19, 21--23 | Conference titles/pages were retained from the manuscript; DOI and full proceedings metadata were not independently verified. | Verify each item on ACM, IEEE, IJCAI, or ACL Anthology. Ref. 19 especially needs the full expansion of “FLLM”, conference location, and publisher. |
| 16--18, 25, 27--28 | Source bibliography used `et al.`. | Replace `and others` with complete verified author lists before final production if required by the journal. |
| 17, 28 | NeurIPS entries lack volume and page/article identifiers in the manuscript. | Retrieve the official NeurIPS BibTeX record. |
| 21 | IJCAI page range is present, but DOI was not verified. | Retrieve the official IJCAI proceedings record. |
| 24 | The source manuscript stated 2026, but the official arXiv submission is 2025. | Ensure surrounding prose does not rely on a 2026 publication date. |
| 30 | Classic ICML proceedings record has no DOI in the manuscript. | Verify publisher/proceedings metadata; do not add a DOI without a primary record. |
| 33 | Official NeurIPS record confirms the work, but the page range was not copied because it was not present in the manuscript and was not needed for record identity. | Add official pages only from the downloadable NeurIPS BibTeX record. |
| 34--35, 37--38, 40--41 | DOI values were already present in the manuscript and therefore retained. | Run a final DOI-resolution check before submission; some publisher endpoints blocked automated inspection during this pass. |
| 36 | Book chapter metadata has no ISBN/DOI and was retained from the manuscript. | Confirm edition, publisher location, and ISBN using the book's library or publisher record. |

## High-priority checks before submission

1. Expand every `and others` author list.
2. Verify Ref. 19's conference identity and Ref. 12's journal DOI.
3. Confirm whether Applied Intelligence permits citing the 2026 preprints in Refs. 24 and 29 at the date of submission; Ref. 24 is now dated 2025 per arXiv.
4. Run the final `.bib` through the Springer template and inspect the generated reference list for capitalization of `{PPO}`, `{LLM}`, `{GIFT}`, `{FinRL}`, and other protected acronyms.
5. Re-run DOI resolution shortly before submission because the journal's production check may reject malformed or non-resolving identifiers.
