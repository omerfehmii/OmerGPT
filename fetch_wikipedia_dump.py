#!/usr/bin/env python3
import argparse
import bz2
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET


REFERENCE_SECTION_RE = re.compile(
    r"(?is)\n==+\s*(kaynakça|kaynaklar|dış bağlantılar|dis baglantilar|notlar|"
    r"ayrıca bakınız|ayrica bakiniz|bibliyografya|external links|references|notes|"
    r"see also|further reading)\s*==+.*"
)


def clean_wiki_text(text):
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^>/]*/>", " ", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", " ", text, flags=re.DOTALL)
    text = re.sub(r"\{\|.*?\|\}", " ", text, flags=re.DOTALL)
    text = re.sub(r"\[\[(Category|Kategori):[^\]]+\]\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\[(File|Image|Dosya):[^\]]+\]\]", " ", text, flags=re.IGNORECASE)
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"\{\{[^{}]*\}\}", " ", text)
    match = REFERENCE_SECTION_RE.search(text)
    if match:
        text = text[: match.start()]
    text = re.sub(r"==+[^=]+==+", " ", text)
    text = re.sub(r"\[https?://[^\s\]]+ ([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://[^\s\]]+\]", " ", text)
    text = re.sub(r"\[\[[^\]|]+\|([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[\s*\d+(?:\s*[-–,]\s*\d+)*\s*\]", " ", text)
    text = re.sub(
        r"(?im)^.*\b(ISBN|ISSN|OCLC|doi|PMID|arXiv|Bibcode|CiteSeerX)\b.*$",
        " ",
        text,
    )
    text = text.replace("'''", "").replace("''", "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\r", "")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def iter_pages(stream):
    context = ET.iterparse(stream, events=("end",))
    for _, elem in context:
        if elem.tag.endswith("page"):
            ns = elem.findtext("./{*}ns")
            title = elem.findtext("./{*}title") or ""
            text = elem.findtext(".//{*}text") or ""
            yield ns, title, text
            elem.clear()


def build_dump_url(lang):
    return (
        "https://dumps.wikimedia.org/"
        f"{lang}wiki/latest/{lang}wiki-latest-pages-articles-multistream.xml.bz2"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="tr")
    parser.add_argument("--out", type=str, default="data_wikipedia.txt")
    parser.add_argument("--max_pages", type=int, default=500)
    parser.add_argument("--max_chars", type=int, default=500_000)
    parser.add_argument("--min_chars", type=int, default=200)
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--url", type=str, default="")
    args = parser.parse_args()

    url = args.url or build_dump_url(args.lang)
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "CodexLearning/0.1 (local script)",
        },
    )

    out_chunks = []
    total_chars = 0
    kept_pages = 0
    seen_pages = 0

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            with bz2.BZ2File(resp) as bz:
                for ns, title, text in iter_pages(bz):
                    seen_pages += 1
                    if ns != "0":
                        continue
                    if text.startswith("#REDIRECT") or text.startswith("#YÖNLENDİR"):
                        continue
                    if not args.raw:
                        text = clean_wiki_text(text)
                    if len(text) < args.min_chars:
                        continue
                    out_chunks.append(text)
                    total_chars += len(text) + 2
                    kept_pages += 1
                    if kept_pages % 50 == 0:
                        print(
                            f"kept {kept_pages} pages, {total_chars} chars, "
                            f"scanned {seen_pages}",
                            file=sys.stderr,
                        )
                    if args.max_pages > 0 and kept_pages >= args.max_pages:
                        break
                    if args.max_chars > 0 and total_chars >= args.max_chars:
                        break
    except Exception as exc:
        print(f"download/parse failed: {exc}", file=sys.stderr)
        sys.exit(1)

    if not out_chunks:
        print("no content extracted; try higher --max_pages or --min_chars", file=sys.stderr)
        sys.exit(1)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n\n".join(out_chunks))

    print(
        f"wrote {kept_pages} pages, {total_chars} chars to {args.out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
