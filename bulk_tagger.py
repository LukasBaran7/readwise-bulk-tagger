#!/usr/bin/env python3
"""
Bulk-tag Readwise Reader articles using GPT-5.4-mini.

Usage:
    python bulk_tagger.py                    # Tag 100 untagged articles from 'later'
    python bulk_tagger.py --limit 500        # Tag 500 articles
    python bulk_tagger.py --location archive # Tag archived articles
    python bulk_tagger.py --all              # Tag ALL untagged articles (paginate through everything)
    python bulk_tagger.py --dry-run          # Preview without applying tags
    python bulk_tagger.py --retag            # Re-tag articles that already have tags
"""

import argparse
import functools
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
import urllib.parse
from collections import Counter
from pathlib import Path

# Force all print() calls to flush immediately so output streams in real time
print = functools.partial(print, flush=True)

# --- Config ---
SCRIPT_DIR = Path(__file__).parent
ENV_PATH = SCRIPT_DIR / ".env"

READWISE_BASE = "https://readwise.io/api/v3"

MODEL = "gpt-5.4-mini"

VALID_DOMAINS = {"ai", "tech", "career", "business", "health", "parenting", "politics", "culture"}
VALID_TYPES = {"how-to", "essay", "news", "reference", "case-study"}
VALID_DEPTHS = {"quick-read", "deep-dive"}
ALL_VALID = VALID_DOMAINS | VALID_TYPES | VALID_DEPTHS

TAGGING_PROMPT = """Classify this document by selecting tags from the fixed lists below.

DOMAIN — pick 1 or 2 that best fit (never more than 2):
- ai: artificial intelligence, LLMs, machine learning, AI tools, AI agents, AI coding
- tech: programming, software engineering, databases, infrastructure, developer tools, architecture
- career: leadership, management, hiring, workplace culture, professional growth, productivity, time management
- business: company strategy, startups, industry news, markets, investing, economics, finance
- health: fitness, nutrition, medicine, mental health, biohacking, longevity
- parenting: children, family, relationships, education
- politics: government, policy, international relations, war, society, surveillance
- culture: arts, history, lifestyle, food, travel, entertainment, sports, games

TYPE — pick exactly 1:
- how-to: tutorials, guides, practical instructions, step-by-step
- essay: opinion, analysis, think pieces, arguments, commentary
- news: current events, announcements, reports, breaking stories
- reference: documentation, lists, tools, cheat sheets, comparisons, resources
- case-study: real-world examples, post-mortems, company stories, profiles, interviews

DEPTH — pick exactly 1:
- quick-read: short, surface-level, single idea, easily skimmed
- deep-dive: long-form, dense, nuanced, requires focused attention

Document:
\"\"\"
Title: {title}
Author: {author}
Domain: {domain}
Word count: {word_count}
{content_section}
\"\"\"

Return ONLY a comma-separated list of tags. No labels, no explanation, no extra text.
Always return exactly 3 or 4 tags: 1-2 domain + 1 type + 1 depth.

Examples:
ai, tech, how-to, deep-dive
politics, news, quick-read
health, essay, deep-dive
career, case-study, quick-read
ai, business, news, quick-read"""

# Tags that indicate an article was already processed by this script
OUR_TAGS = ALL_VALID


def load_env():
    if ENV_PATH.exists():
        with open(ENV_PATH) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())


def readwise_get(path, params=None):
    url = f"{READWISE_BASE}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "Authorization": f"Token {os.environ['READWISE_TOKEN']}",
        "Content-Type": "application/json",
    })
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = min(2 ** attempt * 3, 60)
                print(f" [429, wait {wait}s]", end="", flush=True)
                time.sleep(wait)
            else:
                raise
    return None


def add_tags_to_document(doc_id, tag_names):
    """Add tags via the Readwise Reader API."""
    url = f"{READWISE_BASE}/update/{doc_id}/"
    data = json.dumps({"tags": tag_names}).encode()
    req = urllib.request.Request(url, data=data, method="PATCH", headers={
        "Authorization": f"Token {os.environ['READWISE_TOKEN']}",
        "Content-Type": "application/json",
    })
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req) as r:
                return r.status == 200
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = min(2 ** attempt * 3, 30)
                print(f" [tag 429, wait {wait}s]", end="", flush=True)
                time.sleep(wait)
            else:
                print(f" [tag error {e.code}]", end="", flush=True)
                return False
    return False


def fetch_untagged_articles(location, limit, retag=False):
    """Fetch articles, optionally filtering to only untagged ones."""
    articles = []
    cursor = None
    page = 0

    while len(articles) < limit:
        page += 1
        params = {
            "location": location,
            "page_size": min(100, limit - len(articles) + 50),  # fetch extra to account for filtering
        }
        if cursor:
            params["pageCursor"] = cursor

        print(f"  Fetching page {page}...", end="", flush=True)
        data = readwise_get("/list/", params)
        if not data:
            print(" failed")
            break

        batch = data.get("results", [])
        print(f" got {len(batch)} docs", end="", flush=True)

        for doc in batch:
            if len(articles) >= limit:
                break

            # Skip very short or empty docs
            wc = doc.get("word_count") or 0
            if wc < 30:
                continue

            # Skip already-tagged docs unless --retag
            if not retag:
                tags = doc.get("tags") or {}
                existing = set(t.lower() for t in tags.keys())
                if existing & OUR_TAGS:
                    continue

            articles.append(doc)

        filtered_count = len(articles)
        print(f" → {filtered_count} qualifying so far")

        cursor = data.get("nextPageCursor")
        if not cursor:
            print("  No more pages.")
            break

        time.sleep(0.5)

    return articles[:limit]


def get_article_content(article):
    """Extract content for the prompt."""
    html = article.get("html_content") or ""
    if html:
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        words = text.split()
        if len(words) > 2000:
            text = " ".join(words[:2000]) + "..."
        return f"Content: {text}"

    summary = article.get("summary") or ""
    if summary:
        return f"Summary: {summary}"

    return "Content: [not available]"


def classify_article(title, author, domain, word_count, content_section):
    """Call GPT-5.4-mini to classify an article."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in .env")
        sys.exit(1)

    prompt = TAGGING_PROMPT.format(
        title=title,
        author=author or "Unknown",
        domain=domain or "Unknown",
        word_count=word_count or "Unknown",
        content_section=content_section,
    )

    data = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_completion_tokens": 50,
    }).encode()

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req) as r:
                resp = json.loads(r.read())
            raw = resp["choices"][0]["message"]["content"].strip()
            usage = resp.get("usage", {})
            return raw, usage
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = min(2 ** attempt * 5, 30)
                print(f" [LLM 429, wait {wait}s]", end="", flush=True)
                time.sleep(wait)
            else:
                body = e.read().decode()[:200]
                return f"ERROR: {e.code} {body}", {}

    return "ERROR: max retries", {}


def parse_tags(raw_output):
    """Parse and validate tags."""
    tags = [t.strip().lower() for t in raw_output.split(",")]
    tags = [t for t in tags if t]
    valid = [t for t in tags if t in ALL_VALID]

    domains = [t for t in valid if t in VALID_DOMAINS]
    types = [t for t in valid if t in VALID_TYPES]
    depths = [t for t in valid if t in VALID_DEPTHS]

    well_formed = len(domains) >= 1 and len(types) == 1 and len(depths) == 1
    return valid, well_formed


def main():
    parser = argparse.ArgumentParser(description="Bulk-tag Readwise Reader articles")
    parser.add_argument("--limit", "-n", type=int, default=100, help="Number of articles to tag (default: 100)")
    parser.add_argument("--location", "-l", default="later", help="Reader location: later, archive, new, feed (default: later)")
    parser.add_argument("--all", action="store_true", help="Tag all untagged articles (ignore --limit)")
    parser.add_argument("--dry-run", action="store_true", help="Preview tags without applying them")
    parser.add_argument("--retag", action="store_true", help="Re-tag articles that already have taxonomy tags")
    args = parser.parse_args()

    load_env()

    if not os.environ.get("READWISE_TOKEN"):
        print("ERROR: READWISE_TOKEN not set in environment or .env")
        sys.exit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set in environment or .env")
        sys.exit(1)

    limit = 999999 if args.all else args.limit

    print("=" * 60)
    print(f"BULK TAGGER — {MODEL}")
    print(f"Location: {args.location} | Limit: {'ALL' if args.all else limit} | Dry run: {args.dry_run} | Retag: {args.retag}")
    print("=" * 60)

    # Fetch articles
    print(f"\nFetching {'all' if args.all else limit} untagged articles from '{args.location}'...")
    articles = fetch_untagged_articles(args.location, limit, retag=args.retag)
    print(f"\nFound {len(articles)} articles to process.\n")

    if not articles:
        print("Nothing to do!")
        return

    # Show range info
    first = articles[0]
    last = articles[-1]
    first_date = (first.get("saved_at") or first.get("created_at") or "?")[:10]
    last_date = (last.get("saved_at") or last.get("created_at") or "?")[:10]
    first_title = (first.get("title") or "Untitled")[:60]
    last_title = (last.get("title") or "Untitled")[:60]

    print(f"  First: [{first_date}] {first_title}")
    print(f"  Last:  [{last_date}] {last_title}")
    print()

    # Process
    run_start = time.time()
    tagged = 0
    errors = 0
    malformed = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    tag_counts = Counter()

    for i, article in enumerate(articles):
        title = (article.get("title") or "Untitled")[:80]
        author = article.get("author") or "Unknown"
        domain = article.get("site_name") or ""
        word_count = article.get("word_count") or 0
        content_section = get_article_content(article)
        doc_id = article["id"]
        saved_at = (article.get("saved_at") or article.get("created_at") or "")[:10]

        elapsed = time.time() - run_start
        rate = (i / elapsed) if elapsed > 0 and i > 0 else 0
        eta = ((len(articles) - i) / rate) if rate > 0 else 0

        print(f"[{i+1}/{len(articles)}] ({saved_at}) {title}")
        if i > 0 and i % 10 == 0:
            print(f"  ── progress: {tagged} tagged, {errors} err | "
                  f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining | "
                  f"{rate:.1f} art/s")

        # Classify
        raw, usage = classify_article(title, author, domain, word_count, content_section)
        total_prompt_tokens += usage.get("prompt_tokens", 0)
        total_completion_tokens += usage.get("completion_tokens", 0)

        if raw.startswith("ERROR"):
            print(f"  ✗ {raw[:80]}")
            errors += 1
            continue

        tags, well_formed = parse_tags(raw)

        if not well_formed or not tags:
            print(f"  ⚠ Malformed: {raw}")
            malformed += 1
            # Still apply whatever valid tags we got
            if not tags:
                continue

        print(f"  → {', '.join(tags)}")
        tag_counts.update(tags)

        # Apply tags
        if not args.dry_run:
            ok = add_tags_to_document(doc_id, tags)
            if ok:
                tagged += 1
            else:
                print(f"  ✗ Failed to apply tags")
                errors += 1
        else:
            tagged += 1

        # Light delay to avoid rate limits
        time.sleep(0.15)

    # Summary
    total_elapsed = time.time() - run_start
    cost = (total_prompt_tokens * 0.40 + total_completion_tokens * 1.60) / 1_000_000

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Articles processed: {len(articles)}")
    print(f"  Successfully tagged: {tagged}")
    print(f"  Errors: {errors}")
    print(f"  Malformed (partial tags applied): {malformed}")
    print(f"  Tokens: {total_prompt_tokens:,} in + {total_completion_tokens:,} out")
    print(f"  Cost: ${cost:.4f}")
    print(f"  Time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}m)")
    print(f"  Range: [{first_date}] → [{last_date}]")
    if args.dry_run:
        print(f"\n  ** DRY RUN — no tags were actually applied **")

    # Tag distribution
    print(f"\n  Tag distribution:")
    for tag, count in tag_counts.most_common():
        bar = "█" * (count // 2) if count > 1 else "▎"
        print(f"    {tag:<12} {count:>3}  {bar}")



if __name__ == "__main__":
    main()
