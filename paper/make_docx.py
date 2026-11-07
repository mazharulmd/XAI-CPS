"""Convert the LaTeX manuscript to a Word file (main.docx) with pandoc.

The submission system extracts title, authors, abstract and keywords from Word files. This script
builds a flattened, pandoc-friendly copy of main.tex:
  * inlines every \\input (sections, generated tables and macros);
  * resolves \\ref/\\cref and numbers captions and display equations from main.aux, so the Word
    file carries the same figure, table, section and equation numbers as the PDF;
  * replaces the elsarticle front matter with plain paragraphs, and the TikZ figure and PDF
    plots with PNG renderings;
  * rewrites the algorithm as a numbered list.
Run `latexmk -pdf main.tex` first (main.aux must be current). Requires pandoc and pdftoppm.
"""
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build_docx"
BUILD.mkdir(exist_ok=True)


def read(p):
    return (HERE / p).read_text()


def inline(text):
    def rep(m):
        name = m.group(1)
        p = name if name.endswith(".tex") else name + ".tex"
        return inline(read(p))
    return re.sub(r"\\input\{([^}]+)\}", rep, text)


def strip_comments(text):
    return re.sub(r"(?<!\\)%.*", "", text)


def brace_arg(s, i):
    """s[i] == '{'; return (content, index after closing brace)."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
    raise ValueError("unbalanced braces")


def unwrap(s, cmd, nargs_skip):
    """Replace \\cmd{a1}..{an}{body} by body (the first nargs_skip args are dropped)."""
    out, i = [], 0
    pat = "\\" + cmd
    while True:
        k = s.find(pat, i)
        if k < 0:
            out.append(s[i:])
            return "".join(out)
        out.append(s[i:k])
        j = k + len(pat)
        for _ in range(nargs_skip):
            while s[j].isspace():
                j += 1
            _, j = brace_arg(s, j)
        while s[j].isspace():
            j += 1
        body, j = brace_arg(s, j)
        out.append(body)
        i = j


# ----------------------------------------------------------------------------- labels from main.aux
labels = {}
for m in re.finditer(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", read("main.aux")):
    labels[m.group(1)] = m.group(2)
KIND = {"fig": ("Fig.", "Figs."), "tab": ("Table", "Tables"), "sec": ("Section", "Sections"),
        "eq": ("Eq.", "Eqs."), "alg": ("Algorithm", "Algorithms"), "prop": ("Proposition", "Propositions"),
        "app": ("Appendix", "Appendices")}


def ref_text(keys, capital=True):
    keys = [k.strip() for k in keys.split(",")]
    kind = keys[0].split(":")[0]
    one, many = KIND.get(kind, ("", ""))
    nums = [labels.get(k, "??") for k in keys]
    if kind == "eq":
        nums = [f"({n})" for n in nums]
    if len(nums) == 1:
        return f"{one} {nums[0]}".strip()
    joined = ", ".join(nums[:-1]) + " and " + nums[-1]
    return f"{many} {joined}".strip()


def resolve_refs(s):
    # rendered as internal links (blue, clickable) to the first label, like the PDF's hyperref links
    s = re.sub(r"\\[Cc]ref\{([^}]+)\}",
               lambda m: "\\hyperref[" + m.group(1).split(",")[0].strip() + "]{" + ref_text(m.group(1)) + "}", s)
    s = re.sub(r"\\ref\{([^}]+)\}", lambda m: labels.get(m.group(1), "??"), s)
    return s


# ----------------------------------------------------------------------------- figures
def render_figures():
    for name in ("hr1_over_time", "dcl", "cost_accuracy", "temporal_stability"):
        src = HERE / "figures" / f"{name}.pdf"
        if src.exists():
            subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile", str(src), str(BUILD / name)], check=True)
    # Figure 1 (TikZ): crop it from the compiled PDF page
    page = int(re.search(r"\\newlabel\{fig:arch\}\{\{1\}\{(\d+)\}", read("main.aux")).group(1))
    m = re.search(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", read("sections/method.tex"), re.S)
    tikz = m.group(0)
    pre = r"""\documentclass[border=4pt]{standalone}
\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath,amssymb,bm}\usepackage{xcolor}
\usepackage{tikz}\usetikzlibrary{arrows.meta,positioning,fit,backgrounds,calc}
\newcommand{\LX}{\textup{LX}}\newcommand{\TEA}{\textup{TEA}}
\begin{document}"""
    (BUILD / "fig_arch.tex").write_text(pre + tikz + r"\end{document}")
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "fig_arch.tex"], cwd=BUILD, check=True,
                   stdout=subprocess.DEVNULL)
    subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile", "fig_arch.pdf", "fig_arch"], cwd=BUILD, check=True)
    return page


# ----------------------------------------------------------------------------- main transformation
def front_matter(main):
    title = re.search(r"\\title\{(.*?)\}\n", main).group(1)
    abstract = strip_comments(read("sections/abstract.tex")).strip()
    keywords = re.search(r"\\begin\{keyword\}(.*?)\\end\{keyword\}", main, re.S).group(1).strip()
    keywords = re.sub(r"\s*\\sep\s*", "; ", keywords)
    authors = re.findall(r"\\author\[(\w+)\]\{([^\\}]+)", main)
    aff = {}
    for m in re.finditer(r"\\affiliation\[(\w+)\]\{(.*?)\}\}", main, re.S):
        fields = dict(re.findall(r"(\w+)=\{([^}]*)\}", m.group(2) + "}"))
        aff[m.group(1)] = ", ".join(fields[k] for k in ("organization", "addressline", "city", "postcode", "country") if k in fields)
    emails = re.findall(r"\\ead\{([^}]+)\}", main)
    orcid = dict(re.findall(r"\\fntext\[(\w+)\]\{ORCID: \\href\{[^}]+\}\{([^}]+)\}", main))
    order = list(dict.fromkeys(k for k, _ in authors))
    mark = {k: "abc"[i] for i, k in enumerate(order)}
    # map e-mails to authors in order of appearance in the source
    blocks = re.split(r"(?=\\author\[)", main[main.index("\\author["):main.index("\\affiliation[")])
    lines = []
    for b in [b for b in blocks if b.startswith("\\author[")]:
        name = re.search(r"\\author\[\w+\]\{([^\\}]+)", b).group(1).strip()
        key = re.search(r"\\author\[(\w+)\]", b).group(1)
        em = re.findall(r"\\ead\{([^}]+)\}", b)
        oc = re.search(r"\\fnref\{(\w+)\}", b)
        corr = "\\corref" in b
        extra = []
        if em:
            extra.append("e-mail: " + ", ".join(em))
        if oc and oc.group(1) in orcid:
            extra.append("ORCID: " + orcid[oc.group(1)])
        lines.append((name, key, corr, "; ".join(extra)))
    author_line = ", ".join(f"{n}\\textsuperscript{{{mark[k]}{',*' if c else ''}}}" for n, k, c, _ in lines)
    aff_lines = "\n\n".join(f"\\textsuperscript{{{mark[k]}}} {aff[k]}" for k in order)
    contact = "\n\n".join(f"{n}: {x}" for n, _, _, x in lines)
    return (f"\\title{{{title}}}\n\\begin{{document}}\n\\maketitle\n\n{author_line}\n\n{aff_lines}\n\n"
            f"\\textsuperscript{{*}} Corresponding author.\n\n{contact}\n\n"
            f"\\section*{{Abstract}}\n{abstract}\n\n\\textbf{{Keywords:}} {keywords}\n\n")


def transform(body):
    body = strip_comments(body)
    body = re.sub(r"\\begin\{figure\}\[t\]\s*\\centering\s*\\resizebox\{\\linewidth\}\{!\}\{\s*\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}\}",
                  r"\\begin{figure}[t]\n\\centering\n\\includegraphics[width=\\linewidth]{fig_arch.png}", body, flags=re.S)
    body = re.sub(r"\\includegraphics\[[^]]*\]\{figures/([\w_]+)\.pdf\}", r"\\includegraphics[width=\\linewidth]{\1.png}", body)
    body = unwrap(body, "resizebox", 2)
    # captions: prefix with their number from main.aux
    def cap(m):
        env, inner = m.group(1), m.group(2)
        lab = re.search(r"\\label\{([^}]+)\}", inner)
        num = labels.get(lab.group(1), "?") if lab else "?"
        word = "Fig." if env == "figure" else "Table"
        return m.group(0).replace("\\caption{", f"\\caption{{{word} {num}. ", 1)
    body = re.sub(r"\\begin\{(figure|table)\}(.*?)\\end\{\1\}", cap, body, flags=re.S)
    # display equations: tag with their numbers
    def eq(m):
        inner = m.group(1)
        lab = re.search(r"\\label\{([^}]+)\}", inner)
        num = labels.get(lab.group(1)) if lab else None
        inner = re.sub(r"\\label\{[^}]+\}", "", inner)
        inner = inner.replace("\\\\\n", " ")
        return "\\[" + inner + (f"\\qquad ({num})" if num else "") + "\\]"
    eqnum = iter(range(1, 100))
    def eq_unlabelled(m):
        return eq(m)
    body = re.sub(r"\\begin\{(?:equation|multline)\}(.*?)\\end\{(?:equation|multline)\}", eq, body, flags=re.S)
    # algorithm -> numbered list
    def alg(m):
        a = m.group(0)
        cap_ = re.search(r"\\caption\{([^}]*)\}", a).group(1)
        num = labels.get("alg:main", "1")
        steps = re.findall(r"\\(?:Require|State|For|EndFor)\s*(.*)", a)
        items = "\n".join(f"\\item {x.strip()}" for x in steps if x.strip())
        return f"\\textbf{{Algorithm {num}.}} {cap_}\n\\begin{{enumerate}}\n{items}\n\\end{{enumerate}}"
    body = re.sub(r"\\begin\{algorithm\}.*?\\end\{algorithm\}", alg, body, flags=re.S)
    body = resolve_refs(body)
    body = body.replace("\\mathbb 1", "\\mathbb{1}").replace("\\bm\\", "\\boldsymbol\\").replace("\\bm ", "\\boldsymbol ")
    body = re.sub(r"\\[Bb]igg?l", r"\\left", body); body = re.sub(r"\\[Bb]igg?r", r"\\right", body)
    body = re.sub(r"\\paragraph\{([^}]*)\}", lambda m: "\\textbf{" + m.group(1).rstrip(".?") + (".}" if not m.group(1).endswith("?") else "?}"), body)
    body = body.replace("\\cmidrule(lr){2-7}\\cmidrule(lr){8-8}\\cmidrule(lr){9-9}", "")
    body = re.sub(r"\\cmidrule(\([^)]*\))?\{[^}]*\}", "", body)
    body = re.sub(r"\\(tiny|scriptsize|footnotesize)\b", "", body)
    return body


# ----------------------------------------------------------------------------- table column widths
def _cell_text(tc):
    return "".join(re.findall(r"<(?:w|m):t(?: [^>]*)?>([^<]*)</(?:w|m):t>", tc))


def reorder(xml, parent, order):
    """Sort the self-closing children of every <parent>...</parent> into schema order."""
    rank = {n: i for i, n in enumerate(order)}
    def fix(m):
        kids = re.findall(r"<\w+:(\w+)\b[^>]*/>", m.group(2))
        parts = re.findall(r"<\w+:\w+\b[^>]*/>", m.group(2))
        if "".join(parts) != m.group(2).strip():
            return m.group(0)                      # non-self-closing children: leave untouched
        parts = [x for _, x in sorted(zip(kids, parts), key=lambda kv: rank.get(kv[0], 99))]
        return m.group(1) + "".join(parts) + m.group(3)
    return re.sub(rf"(<{parent}>)(.*?)(</{parent}>)", fix, xml, flags=re.S)


SETTINGS_ORDER = """writeProtection view zoom removePersonalInformation removeDateAndTime doNotDisplayPageBoundaries
displayBackgroundShape printPostScriptOverText printFractionalCharacterWidth printFormsData embedTrueTypeFonts
embedSystemFonts saveSubsetFonts saveFormsData mirrorMargins alignBordersAndEdges bordersDoNotSurroundHeader
bordersDoNotSurroundFooter gutterAtTop hideSpellingErrors hideGrammaticalErrors activeWritingStyle proofState
formsDesign attachedTemplate linkStyles stylePaneFormatFilter stylePaneSortMethod documentType mailMerge revisionView
trackRevisions doNotTrackMoves doNotTrackFormatting documentProtection autoFormatOverride styleLockTheme styleLockQFSet
defaultTabStop autoHyphenation consecutiveHyphenLimit hyphenationZone doNotHyphenateCaps showEnvelope summaryLength
clickAndTypeStyle defaultTableStyle evenAndOddHeaders bookFoldRevPrinting bookFoldPrinting bookFoldPrintingSheets
drawingGridHorizontalSpacing drawingGridVerticalSpacing displayHorizontalDrawingGridEvery
displayVerticalDrawingGridEvery doNotUseMarginsForDrawingGridOrigin drawingGridHorizontalOrigin
drawingGridVerticalOrigin doNotShadeFormData noPunctuationKerning characterSpacingControl printTwoOnOne
strictFirstAndLastChars noLineBreaksAfter noLineBreaksBefore savePreviewPicture doNotValidateAgainstSchema
saveInvalidXml ignoreMixedContent alwaysShowPlaceholderText doNotDemarcateInvalidXml saveXmlDataOnly
useXSLTWhenSaving saveThroughXslt showXMLTags alwaysMergeEmptyNamespace updateFields hdrShapeDefaults footnotePr
endnotePr compat docVars rsids mathPr attachedSchema themeFontLang clrSchemeMapping doNotIncludeSubdocsInStats
doNotAutoCompressPictures forceUpgrade captions readModeInkLockDown smartTagType schemaLibrary shapeDefaults
doNotEmbedSmartTags decimalSymbol listSeparator""".split()


def sort_settings(data):
    """Put the children of w:settings into schema order (pandoc's default settings are not)."""
    from lxml import etree
    root = etree.fromstring(data)
    rank = {n: i for i, n in enumerate(SETTINGS_ORDER)}
    kids = list(root)
    for k in kids:
        root.remove(k)
    for k in sorted(kids, key=lambda e: rank.get(etree.QName(e).localname, len(rank))):
        root.append(k)
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def fit_tables(docx, total=9000, per_char=135):
    """Size each table column by its longest content (pandoc gives every column the same width)."""
    import zipfile
    zin = zipfile.ZipFile(docx)
    items = {n: zin.read(n) for n in zin.namelist() if not n.endswith("/")}
    zin.close()
    xml = items["word/document.xml"].decode("utf8")

    def fix(tbl):
        rows = re.findall(r"<w:tr>.*?</w:tr>", tbl, re.S)
        ncol = len(re.findall(r"<w:gridCol ", tbl))
        need = [4.0] * ncol
        for r in rows:
            c = 0
            for tc in re.findall(r"<w:tc>.*?</w:tc>", r, re.S):
                span = int((re.search(r'<w:gridSpan w:val="(\d+)"', tc) or [0, 1])[1])
                if span == 1 and c < ncol:
                    t = _cell_text(tc)
                    longest = max([len(w) for w in t.split()] + [0])
                    need[c] = max(need[c], min(max(len(t), longest), 30) if c == 0 else min(max(len(t) * 0.8, longest), 18))
                c += span
        w = [n * per_char + 240 for n in need]          # + left/right cell margins
        scale = min(1.0, total / sum(w))
        w = [int(x * scale) for x in w]
        tbl = re.sub(r"<w:tblW [^>]*/>", f'<w:tblW w:type="dxa" w:w="{sum(w)}" /><w:tblLayout w:type="fixed" />', tbl, count=1)
        tbl = re.sub(r"<w:tblGrid>.*?</w:tblGrid>", "<w:tblGrid>" + "".join(f'<w:gridCol w:w="{x}" />' for x in w) + "</w:tblGrid>", tbl, count=1, flags=re.S)

        def row(rm):
            r, c, out = rm.group(0), 0, []
            def cell(cm):
                nonlocal c
                tc = cm.group(0)
                span = int((re.search(r'<w:gridSpan w:val="(\d+)"', tc) or [0, 1])[1])
                width = sum(w[c:c + span]); c += span
                tcw = f'<w:tcW w:w="{width}" w:type="dxa" />'
                if "<w:tcPr />" in tc:
                    return tc.replace("<w:tcPr />", f"<w:tcPr>{tcw}</w:tcPr>", 1)
                return tc.replace("<w:tcPr>", "<w:tcPr>" + tcw, 1)
            return re.sub(r"<w:tc>.*?</w:tc>", cell, r, flags=re.S)
        return re.sub(r"<w:tr>.*?</w:tr>", row, tbl, flags=re.S)

    xml = re.sub(r"<w:tbl>.*?</w:tbl>", lambda m: fix(m.group(0)), xml, flags=re.S)
    # schema order fixes: children of w:tblPr and m:rPr in schema order; m:sepChr before m:endChr
    xml = reorder(xml, "w:tblPr", ["tblStyle", "tblpPr", "tblOverlap", "bidiVisual", "tblStyleRowBandSize",
                                   "tblStyleColBandSize", "tblW", "jc", "tblCellSpacing", "tblInd", "tblBorders",
                                   "shd", "tblLayout", "tblCellMar", "tblLook", "tblCaption", "tblDescription"])
    xml = reorder(xml, "m:rPr", ["lit", "nor", "scr", "sty", "brk", "aln"])
    xml = re.sub(r"<m:nor />(<m:sty [^>]*/>)", "<m:nor />", xml)        # nor and sty are mutually exclusive
    xml = reorder(xml, "m:mcPr", ["count", "mcJc"])
    xml = re.sub(r'(<m:endChr m:val="[^"]*" />)(<m:sepChr m:val="[^"]*" />)', r"\2\1", xml)
    items["word/document.xml"] = xml.encode("utf8")
    if "word/numbering.xml" in items:                   # w:nsid must be 8 hex digits
        nb = items["word/numbering.xml"].decode("utf8")
        nb = re.sub(r'(<w:nsid w:val=")([0-9A-Fa-f]{1,7})(")', lambda m: m.group(1) + m.group(2).zfill(8) + m.group(3), nb)
        items["word/numbering.xml"] = nb.encode("utf8")
    items["word/settings.xml"] = sort_settings(items["word/settings.xml"])
    sty = items["word/styles.xml"].decode("utf8")
    sty = sty.replace(" />>", " />").replace('<w:color w:val="345A8A" />', '<w:color w:val="000000" />')
    items["word/styles.xml"] = sty.encode("utf8")
    ct = items["[Content_Types].xml"].decode("utf8")
    if 'Extension="png"' not in ct:
        ct = ct.replace("<Default Extension=\"xml\"", '<Default Extension="png" ContentType="image/png" /><Default Extension="xml"', 1)
        items["[Content_Types].xml"] = ct.encode("utf8")
    st = items["word/settings.xml"].decode("utf8")
    st = re.sub(r"<w:stylePaneFormatFilter [^>]*/>|<w:embedSystemFonts />", "", st)
    z = re.search(r"<w:zoom [^>]*/>", st)
    if z:
        st = st.replace(z.group(0), "", 1)
        st = re.sub(r"(<w:settings[^>]*>)", lambda m: m.group(1) + z.group(0), st, count=1)
        items["word/settings.xml"] = st.encode("utf8")
    zout = zipfile.ZipFile(docx, "w", zipfile.ZIP_DEFLATED)
    for n, b in items.items():
        zout.writestr(n, b)
    zout.close()


def main():
    render_figures()
    main_tex = read("main.tex")
    macros = read("generated/numbers.tex")
    preamble = (r"\documentclass{article}" "\n" + macros + "\n"
                r"\newcommand{\LX}{\textup{LX}}" "\n" r"\newcommand{\TEA}{\textup{TEA}}" "\n")
    body = main_tex[main_tex.index("\\end{frontmatter}") + len("\\end{frontmatter}"):]
    body = body[:body.index("\\bibliographystyle")]
    body = transform(inline(body))
    flat = preamble + front_matter(main_tex) + body + "\n\\section*{References}\n\\end{document}\n"
    (BUILD / "flat.tex").write_text(flat)
    ref = HERE / "word_reference.docx"
    cmd = ["pandoc", "flat.tex", "-o", str(HERE / "main.docx"), "--citeproc",
           f"--bibliography={HERE / 'refs.bib'}", "--resource-path=.", "-M", "link-citations=true", "--number-sections"]
    if ref.exists():
        cmd.append(f"--reference-doc={ref}")
    r = subprocess.run(cmd, cwd=BUILD, capture_output=True, text=True)
    print(r.stderr[-3000:], file=sys.stderr)
    r.check_returncode()
    fit_tables(HERE / "main.docx")
    print("wrote", HERE / "main.docx")


if __name__ == "__main__":
    main()
