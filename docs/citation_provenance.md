# Citation provenance

Automated verifiers (OpenAlex, Crossref, Semantic Scholar, arXiv API) were **blocked** by the network policy of the environment in which this manuscript was prepared, so entries were verified as follows. **Before submission, run `paper/refs.bib` through a DOI/Crossref checker** (e.g. the journal's reference checker or `verify_bibtex`) — this file records what was and was not machine-checked.

| Group | Entries | Verification |
|---|---|---|
| Corpus papers (30 MAIN + 3 EXTRA) | `deng2021gdn` … `alrawajbeh2025adaptive` | Title/authors/venue/DOI from `XAI_IDS_CPS_Papers_Final.xlsx` (curated on 11–12 Oct 2026 from publisher pages); DOIs cross-checked against the first page of the local PDF where it prints one (#3, #6–11, #13–17, #23, #30, #31); volume/pages for #12, #13, #18, #20–22, #25 read from PDF headers. #24, #28, #29, E2, E3: spreadsheet only (PDF not in repo). |
| HAI | `shin2021hai`, `shin2020hai` | Citation text given in the HAI technical-details PDF (v4, p. 42). DOI of `shin2021hai` from the authors' knowledge — **check**. |
| Web-verified (this session) | `mishra2026conditional`, `venugopalan2025giby`, `puccetti2026latency`, `warnecke2020evaluating`, `kim2022rigorous`, `iec61511` (definition of process safety time) | Web search results (DFKI publication page; arXiv listing; Univ. of Florence repository; KIT library; AAAI OJS). |
| Canonical references | `shapley1953value`, `owen1977unions`, `lundberg2017shap`, `ribeiro2016lime`, `sundararajan2017ig`, `simonyan2014saliency`, `guo2018lemna`, `strumbelj2014explaining`, `castro2009sampling`, `slack2020fooling`, `arp2022dos`, `kaufman2012leakage`, `demsar2006statistical`, `holm1979simple`, `cardenas2011attacks`, `urbina2016limiting`, `downs1993tep`, `kravchik2018cnn` | Standard, widely cited works; bibliographic data from the authors' knowledge — **page numbers and DOIs must be machine-checked before submission**. |

No reference was added that was not either read (PDF), listed in the curated spreadsheet, or confirmed by a web search in this session, except the canonical references above.

**Re-check attempt (30 Oct 2026).** `verify_bibtex` (OpenAlex + arXiv backend) was retried on `fung2024attributions` and `franco2026shats`; both returned `unchecked` because the proxy refused the connection (HTTP 403). The status above is therefore unchanged: no entry has been machine-verified against a bibliographic database.
