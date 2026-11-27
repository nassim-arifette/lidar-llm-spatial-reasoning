"""Check the built website.

* It can be hosted independently at any URL prefix: every local link resolves
  inside ``site/`` and every anchor exists.
* Every number quoted in the text (an element with ``data-stat``) is the one
  ``site/data/audit.json`` holds for that key, as formatted by
  ``tools/build_web_audit.py``. An element marked ``data-optional`` may wait for
  a run that has not finished; it must then be empty of digits.
* The per-question data carries each archived program's outcome under both
  conventions.
"""
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1] / "site"


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()
        self.duplicates = []
        self.stats = []      # [key, optional, text]
        self.open = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            if attrs["id"] in self.ids:
                self.duplicates.append(attrs["id"])
            self.ids.add(attrs["id"])
        if "data-stat" in attrs:
            self.open = [attrs["data-stat"], "data-optional" in attrs, ""]
            self.stats.append(self.open)
        for key in ("src", "href"):
            if attrs.get(key):
                self.links.append(attrs[key])

    def handle_data(self, data):
        if self.open is not None:
            self.open[2] += data

    def handle_endtag(self, tag):
        self.open = None


def check_numbers(doc, errors):
    """Compare the quoted numbers of the page with the exported results."""
    audit = json.loads((ROOT / "data/audit.json").read_text(encoding="utf-8"))
    stats = audit["stats"]
    for key, optional, text in doc.stats:
        if key in stats:
            if text != stats[key]:
                errors.append(f"index.html: {key} reads {text!r}, the results give {stats[key]!r} (rerun tools/build_web_audit.py)")
        elif not optional:
            errors.append(f"index.html: quoted number {key!r} is not in data/audit.json")
        elif re.search(r"\d", text):
            errors.append(f"index.html: optional number {key!r} has no source yet but reads {text!r}")
    study = json.loads((ROOT / "data/study.json").read_text(encoding="utf-8"))
    for run in study["runs"]:
        entry = next((a for a in audit["archive"] if a["id"] == run["id"]), None)
        if entry is None:
            errors.append(f"study.json: run {run['id']} is missing from audit.json")
            continue
        for field, name in (("executed_correct", "program_published"), ("executed_correct_as_labelled", "program_as_labelled"),
                            ("direct_correct", "direct_1")):
            got = sum(bool(r[field]) for r in run["rows"]) / len(run["rows"])
            if abs(got - entry[name]) > 1e-9:
                errors.append(f"study.json: {run['id']} {field} gives {got:.3f}, audit.json says {entry[name]:.3f}")
    return len(doc.stats)


def main():
    pages = {}
    errors = []
    for file in ROOT.rglob("*.html"):
        doc = Links()
        doc.feed(file.read_text(encoding="utf-8"))
        pages[file.resolve()] = doc
        errors.extend(f"{file.name}: duplicate id {x}" for x in doc.duplicates)
    assert (ROOT / "index.html").resolve() in pages, "site/index.html is missing"
    checked = 0
    for file, doc in pages.items():
        for value in doc.links:
            url = urlsplit(value)
            if url.scheme or url.netloc:
                continue
            checked += 1
            target = (file.parent / unquote(url.path)).resolve() if url.path else file
            if not target.is_relative_to(ROOT.resolve()):
                errors.append(f"{file.name}: dependency outside site: {value}")
            elif not target.exists():
                errors.append(f"{file.name}: missing file: {value}")
            elif url.fragment and target in pages and unquote(url.fragment) not in pages[target].ids:
                errors.append(f"{file.name}: missing anchor: {value}")
    numbers = check_numbers(pages[(ROOT / "index.html").resolve()], errors)
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"OK: {len(pages)} page(s), {checked} local links/assets, independent site root, "
          f"{numbers} quoted numbers match data/audit.json")


if __name__ == "__main__":
    main()
