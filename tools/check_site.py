"""Check that the committed website can be hosted independently at any URL prefix."""
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

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            if attrs["id"] in self.ids:
                self.duplicates.append(attrs["id"])
            self.ids.add(attrs["id"])
        for key in ("src", "href"):
            if attrs.get(key):
                self.links.append(attrs[key])


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
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"OK: {len(pages)} page(s), {checked} local links/assets, independent site root")


if __name__ == "__main__":
    main()
