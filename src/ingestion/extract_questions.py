"""Extract every question from the GTU question-paper PDFs into one CSV.

The parser is layout-aware: GTU papers put the part labels "(a)/(b)/(c)" in a
fixed left column and the marks ("03", "04", "07") in a fixed right column, so
word coordinates are used to tell labels and marks apart from numbers or
nested "(a)"/"(i)" labels that appear inside question text and tables.
"""
import csv
import re
from collections import Counter
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
OUT_CSV = ROOT / "data" / "processed" / "questions.csv"
INCOMING_DIR = ROOT / "data" / "incoming"

FILE_RE = re.compile(r"^(?P<abbr>\w+) - (?P<season>[SW])(?P<year>\d{4}) \[(?P<code>\d+)\]")
QNO_RE = re.compile(r"^Q\.?\s*(\d+)\s*(.*)$", re.I)
LABEL_RE = re.compile(r"^\(([a-c])\)$", re.I)
MARKS_RE = re.compile(r"^\d{2}$")
PAGE_MARGIN = 60  # points from the top/bottom edge where page numbers sit


def load_rows(pdf_path):
    """Return visual text rows (words grouped by y) and the marks found in the right column."""
    words = []
    with pymupdf.open(pdf_path) as doc:
        width = doc[0].rect.width
        for pno, page in enumerate(doc):
            height = page.rect.height
            for x0, y0, x1, y1, text, *_ in page.get_text("words"):
                if re.fullmatch(r"\d", text) and (y1 < PAGE_MARGIN or y0 > height - PAGE_MARGIN):
                    continue  # page number
                words.append({"page": pno, "x": x0, "y0": y0, "y1": y1, "y": (y0 + y1) / 2, "text": text})

    # Ignore everything above the first "Q.1" (e.g. "Total Marks: 70" in the header).
    first_q = min((w["page"], w["y0"]) for w in words if re.fullmatch(r"Q\.?\d*", w["text"]))
    header = [w for w in words if (w["page"], w["y1"]) <= first_q]

    # Marks column = the dominant x-position of two-digit numbers on the right side.
    candidates = [w for w in words if w not in header and MARKS_RE.match(w["text"]) and w["x"] > 0.7 * width]
    marks_x = Counter(round(w["x"]) for w in candidates).most_common(1)[0][0]
    marks = [w for w in candidates if abs(w["x"] - marks_x) <= 6]
    body = [w for w in words if w not in marks]

    # Part-label column = the dominant x-position of "(a)/(b)/(c)" tokens.
    label_x = Counter(round(w["x"]) for w in body if LABEL_RE.match(w["text"])).most_common(1)[0][0]

    # Group words into visual rows: same page and vertical extents overlapping by >= 50%.
    rows = []
    for w in sorted(body, key=lambda w: (w["page"], w["y"], w["x"])):
        if rows and rows[-1]["page"] == w["page"]:
            r = rows[-1]
            overlap = min(r["y1"], w["y1"]) - max(r["y0"], w["y0"])
            if overlap >= 0.5 * min(r["y1"] - r["y0"], w["y1"] - w["y0"]):
                r["words"].append(w)
                continue
        rows.append({"page": w["page"], "y": w["y"], "y0": w["y0"], "y1": w["y1"], "words": [w]})
    for r in rows:
        r["words"].sort(key=lambda w: w["x"])
    return rows, marks, label_x


def parse(pdf_path):
    rows, marks, label_x = load_rows(pdf_path)
    header, parts, qno = [], [], None
    current = None

    for row in rows:
        words = row["words"]
        if len(words) == 1 and words[0]["text"].upper() == "OR":
            current = None
            continue
        if QNO_RE.match(words[0]["text"]) and words[0]["x"] < label_x:
            m = QNO_RE.match(" ".join(w["text"] for w in words[:2]))
            qno = int(m.group(1))
            words = [w for w in words[1:] if w["text"] != m.group(1)]
            current = None
        if words and LABEL_RE.match(words[0]["text"]) and abs(words[0]["x"] - label_x) <= 8:
            sub = words[0]["text"][1].lower()
            seen = any(p["question_no"] == qno and p["sub_part"] == sub for p in parts)
            current = {"question_no": qno, "sub_part": sub, "is_or_option": seen,
                       "page": row["page"], "y0": row["y0"], "text": [], "marks": None}
            parts.append(current)
            words = words[1:]
        if qno is None:
            header.append(" ".join(w["text"] for w in words))
        elif current and words and not re.fullmatch(r"\*+", words[0]["text"]):
            current["text"].append(" ".join(w["text"] for w in words))

    # A mark belongs to the last part whose label row starts at or above it. Parts split
    # into sub-tasks carry several marks (e.g. 03 + 04), which are summed.
    for mk in marks:
        owner = None
        for p in parts:
            if (p["page"], p["y0"] - 3) <= (mk["page"], mk["y"]):
                owner = p
        if owner is not None:
            owner["marks"] = (owner["marks"] or 0) + int(mk["text"])

    questions = []
    for p in parts:
        questions.append({
            "question_no": p["question_no"],
            "sub_part": p["sub_part"],
            "is_or_option": p["is_or_option"],
            "question": re.sub(r"\s+", " ", " ".join(p["text"])).strip(),
            "marks": p["marks"],
        })
    return " ".join(header), questions


def known_sources():
    """source_file values already present in the original dataset and in incoming CSVs."""
    known = set()
    for f in [OUT_CSV, *INCOMING_DIR.glob("*.csv")]:
        if f.exists():
            with open(f, encoding="utf-8-sig") as fh:
                known |= {row["source_file"] for row in csv.DictReader(fh)}
    return known


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Extract questions from GTU question-paper PDFs.")
    ap.add_argument("--only-new", action="store_true",
                    help="extract only PDFs not yet in the dataset; write them to data/incoming/ (original CSV untouched)")
    ap.add_argument("--output", help="output CSV (default: data/processed/questions.csv, or data/incoming/new_papers_<n>.csv with --only-new)")
    a = ap.parse_args(argv)
    known = known_sources() if a.only_new else set()
    out_csv = Path(a.output) if a.output else (
        INCOMING_DIR / f"new_papers_{len(list(INCOMING_DIR.glob('new_papers_*.csv'))) + 1}.csv" if a.only_new else OUT_CSV)
    if not a.only_new and out_csv.resolve() == OUT_CSV.resolve() and OUT_CSV.exists() and a.output is None:
        print(f"{OUT_CSV} already exists; use --only-new to add new papers, or --output to write elsewhere.")
        return
    out = []
    for pdf in sorted(RAW_DIR.rglob("*.pdf")):
        if pdf.relative_to(ROOT).as_posix() in known:
            continue
        fm = FILE_RE.match(pdf.name)
        if not fm:
            print(f"skipped (unrecognised file name): {pdf.name}")
            continue
        header, questions = parse(pdf)
        name = re.search(r"Subject Name\s*:\s*(.+?)\s+Time", header)
        date = re.search(r"Date\s*:\s*([\d/-]+)", header)
        sem = re.search(r"SEMESTER\W*([IVX]+)", header)
        for q in questions:
            out.append({
                "university": pdf.relative_to(RAW_DIR).parts[0],
                "subject": name.group(1).strip() if name else pdf.parent.name,
                "subject_code": fm.group("code"),
                "semester": sem.group(1) if sem else "",
                "exam_session": "Summer" if fm.group("season") == "S" else "Winter",
                "year": int(fm.group("year")),
                "exam_date": date.group(1) if date else "",
                **q,
                "source_file": pdf.relative_to(ROOT).as_posix(),
            })

    if not out:
        print("No new papers found.")
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    print(f"Wrote {len(out)} questions to {out_csv}")


if __name__ == "__main__":
    main()
