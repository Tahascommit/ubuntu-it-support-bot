# Scrape the official Ubuntu guides (Server docs, Desktop docs and LTS release notes) into short passages
# for the bot's document library. The page list comes from each site's sitemap, robots.txt allows crawling,
# and we wait 1.2 s between pages (the sites ask for 1 s).
# Run from the project folder:  python scripts/scrape_ubuntu_docs.py
import gzip
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from itbot import html_to_text  # noqa: E402

SITEMAPS = {"server": "https://ubuntu.com/server/docs/doc-sitemap.xml",
            "desktop": "https://documentation.ubuntu.com/desktop/en/latest/sitemap.xml",
            "release-notes": "https://documentation.ubuntu.com/release-notes/sitemap.xml"}
SKIP = ("/contributing/", "/_tags/", "/genindex", "/search", "/py-modindex")
LTS = ("/20.04", "/22.04", "/24.04", "/26.04")          # release notes: only the long-term support versions
DELAY = 1.2
HEADERS = {"User-Agent": "it-support-bot-capstone (student project; one page per second)"}


def get(url):
    body = urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60).read()
    return gzip.decompress(body) if body[:2] == b"\x1f\x8b" else body


def page_urls():
    out = []
    for site, sm in SITEMAPS.items():
        locs = [x.decode() for x in re.findall(rb"<loc>([^<]+)</loc>", get(sm))]
        locs = [u for u in locs if not any(s in u for s in SKIP)]
        if site == "release-notes":
            locs = [u for u in locs if any(v in u for v in LTS)]
        out += [(site, u) for u in dict.fromkeys(locs)]
    return out


def passages(html_bytes, max_words=260):
    """Main article -> list of (title, section heading, text). Long sections are cut at paragraph breaks."""
    soup = BeautifulSoup(html_bytes, "lxml")
    art = soup.find("article") or soup.find(attrs={"role": "main"})
    if art is None:
        return None, []
    for junk in art.select("nav, footer, .headerlink, .feedback, .related-links, script, style, button"):
        junk.decompose()
    h1 = art.find("h1")
    title = h1.get_text(" ", strip=True).replace("¶", "").strip() if h1 else (soup.title.string or "").split(" - ")[0]
    text = html_to_text(str(art)).replace("¶", "")
    out, heading, buf = [], title, []

    def flush():
        words = " ".join(buf).split()
        if len(words) >= 25:
            out.append((title, heading, "\n\n".join(buf).strip()))

    for block in re.split(r"\n{2,}", text):
        m = re.match(r"^#{1,4}\s+(.*)", block)
        if m:                                     # a new heading starts a new passage
            flush()
            heading, buf = m.group(1).strip(), []
            continue
        if sum(len(b.split()) for b in buf) + len(block.split()) > max_words and buf:
            flush()
            buf = []
        buf.append(block)
    flush()
    return title, out


def main():
    urls = page_urls()
    print(f"{len(urls)} pages to fetch", flush=True)
    rows, failed = [], 0
    for i, (site, url) in enumerate(urls):
        try:
            title, parts = passages(get(url))
        except Exception as e:
            failed += 1
            print("  failed", url, type(e).__name__, flush=True)
            time.sleep(DELAY)
            continue
        for k, (t, h, txt) in enumerate(parts):
            rows.append({"doc_id": f"{site}-{i}-{k}", "site": site, "url": url, "title": t, "section": h, "text": txt})
        if i % 25 == 0:
            print(f"{i}/{len(urls)} pages, {len(rows)} passages", flush=True)
        time.sleep(DELAY)
    with gzip.open(ROOT / "data" / "ubuntu_docs.jsonl.gz", "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"saved {len(rows)} passages from {len(urls) - failed} pages ({failed} failed)", flush=True)


if __name__ == "__main__":
    main()
