# Computers & Security submission checklist

Requirements as stated in the journal's Guide for Authors (checked via web search on 6 Nov 2026; the
ScienceDirect page itself blocks automated access, so re-read it before submitting):
<https://www.sciencedirect.com/journal/computers-and-security/publish/guide-for-authors>

| Requirement | Status |
|---|---|
| Article length <= 12,000 words incl. abstract, main text, references and appendices | about 11,400 words counting **everything** in `main.pdf` (tables, captions, title page, declarations, references); appendix moved to `supplementary.pdf` |
| Abstract <= 250 words, structured (background/gap, objective, method, results, conclusion) | 248 words (counting every number as a word) |
| Highlights: 3-5 bullets, <= 85 characters each, separate file | `highlights.txt` (5 bullets, 72-82 characters) |
| Keywords | 6 keywords |
| Review model: single anonymized; title page with authors and corresponding author's e-mail | in `main.tex`: Md Mazharul Islam (corresponding; ORCID 0000-0001-6225-1487), Sazzad Hossain (ORCID 0000-0002-9675-0468), Mamunur Rashid Alex -- **TODO: confirm Md Mazharul Islam's department and the corresponding author** |
| Declaration of generative AI use (Elsevier template wording) | in `sections/declarations.tex` -- **TODO: authors confirm after their own review** |
| Declaration of competing interest, funding, CRediT roles | in `sections/declarations.tex` -- **TODO: confirm/complete** |
| Data availability statement | in `sections/declarations.tex` |
| Reference style: any consistent style at submission | elsarticle-harv (author-year) |
| Supplementary material as a separate file | `supplementary.pdf` |
| Line numbers for reviewers | `lineno` |

Before submitting, also:
1. Verify every entry of `refs.bib` with a DOI/Crossref checker (see `docs/citation_provenance.md`), and check
   whether Cho (Mathematics 2026, 10.3390/math14112016) and Ma et al. (Sensors 2026, 10.3390/s26175585) exist as
   described; if so, cite them in Related work.
2. Recompile: `latexmk -pdf main.tex && latexmk -pdf supplementary.tex`.
3. Upload (Elsevier submission system, Computers & Security):
   - **Manuscript:** `main.docx` (Word lets the system extract title, authors, abstract and keywords) or `main.pdf`
     (LaTeX route: PDF only at first submission; LaTeX sources are requested at revision).
     Regenerate the Word file after any change with `latexmk -pdf main.tex && python3 make_docx.py`.
   - **Declaration of competing interests:** tick the no-competing-interests confirmation, or upload the Word file
     generated with Elsevier's declaration tool (https://declarations.elsevier.com/home?journalAcronym=COSE).
   - **Generative AI:** the disclosure statement is the last section before the references (required placement).
   - **Supplementary material:** `supplementary.pdf`; also `highlights.txt` and a cover letter.
