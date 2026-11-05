# Collect three extra sets through the Stack Exchange API (no key needed, 300 requests per day):
#   1. Ask Ubuntu questions asked after the April 2024 data dump, with their answers: a test set from
#      the "future" that no model has seen, and fresh knowledge for the bot's library;
#   2. top questions of non-IT Stack Exchange sites (cooking, travel, ...): examples of off-topic questions
#      the bot should politely decline;
#   3. the number of new Ask Ubuntu questions per month after the dump, to extend the activity chart.
# Run from the project folder:  python scripts/fetch_stackexchange_api.py
import gzip
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from itbot import html_to_text  # noqa: E402

API = "https://api.stackexchange.com/2.3"
RAW = ROOT / "raw" / "api"
OUT = ROOT / "data"
SINCE = datetime(2024, 4, 8, tzinfo=timezone.utc)      # the dump ends on 7 April 2024
QUESTION_PAGES = 20                                     # 100 questions per page, sorted by votes
OFF_TOPIC_SITES = ["cooking", "travel", "gardening", "bicycles", "pets", "fitness", "parenting",
                   "music", "money", "law", "history", "outdoors"]
used = 0


def call(path, **params):
    """One API request with retries. Honours the 'backoff' field the API sends when it wants us to slow down."""
    global used
    params.setdefault("site", "askubuntu")
    params = {k: v for k, v in params.items() if v is not None}
    url = f"{API}/{path}?" + urllib.parse.urlencode(params)
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip", "User-Agent": "it-support-bot-capstone"})
            with urllib.request.urlopen(req, timeout=90) as r:
                body = r.read()
            d = json.loads(gzip.decompress(body) if body[:2] == b"\x1f\x8b" else body)
            used += 1
            time.sleep(d.get("backoff", 0) + 0.4)
            return d
        except urllib.error.HTTPError as e:
            msg = e.read()
            msg = gzip.decompress(msg) if msg[:2] == b"\x1f\x8b" else msg
            print("  HTTP", e.code, msg[:200], flush=True)
            if e.code in (400, 403) and b"throttle" in msg:
                sys.exit("API quota used up for today. Run again tomorrow.")
        except Exception as e:
            print("  retry after", type(e).__name__, e, flush=True)
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"giving up on {url}")


def epoch(dt):
    return int(dt.timestamp())


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    # a filter that returns each question with its body and its answers (with bodies) in one request
    flt = call("filters/create", base="default", unsafe="false",
               include="question.body;question.answers;answer.body;question.closed_date;question.closed_reason;"
                       "question.answer_count;question.accepted_answer_id",
               exclude="question.owner;answer.owner", site=None)
    flt = flt["items"][0]["filter"]

    # 1. recent Ask Ubuntu questions and answers
    rows, page = [], 1
    while page <= QUESTION_PAGES:
        d = call("questions", fromdate=epoch(SINCE), sort="votes", order="desc", pagesize=100, page=page, filter=flt)
        (RAW / f"recent_page{page:02d}.json").write_text(json.dumps(d))
        for q in d["items"]:
            answers = [{"answer_id": a["answer_id"], "score": a["score"], "is_accepted": a.get("is_accepted", False),
                        "creation_date": a["creation_date"], "body": html_to_text(a.get("body", ""))}
                       for a in q.get("answers", [])]
            rows.append({"question_id": q["question_id"], "title": html_to_text(q["title"]),
                         "body": html_to_text(q.get("body", "")), "tags": q.get("tags", []), "score": q["score"],
                         "view_count": q.get("view_count"), "answer_count": q.get("answer_count", 0),
                         "accepted_answer_id": q.get("accepted_answer_id"), "creation_date": q["creation_date"],
                         "closed_reason": q.get("closed_reason"), "link": q.get("link"), "answers": answers})
        print(f"recent questions page {page}: {len(d['items'])} questions, quota left {d.get('quota_remaining')}", flush=True)
        if not d.get("has_more"):
            break
        page += 1
    with gzip.open(OUT / "askubuntu_recent_api.jsonl.gz", "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"saved {len(rows)} recent questions", flush=True)

    # 2. off-topic questions from other sites (titles only)
    off = []
    for site in OFF_TOPIC_SITES:
        d = call("questions", site=site, sort="votes", order="desc", pagesize=100, filter="default")
        off += [{"site": site, "question_id": q["question_id"], "title": html_to_text(q["title"]), "tags": q.get("tags", [])}
                for q in d["items"]]
        print(f"off-topic {site}: {len(d['items'])}", flush=True)
    with gzip.open(OUT / "off_topic_questions.jsonl.gz", "wt") as f:
        for r in off:
            f.write(json.dumps(r) + "\n")

    # 3. new questions per month since the dump (filter=total returns only the count)
    months, start = [], datetime(SINCE.year, SINCE.month, 1, tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    while start < now:
        end = datetime(start.year + (start.month == 12), start.month % 12 + 1, 1, tzinfo=timezone.utc)
        d = call("questions", fromdate=epoch(start), todate=epoch(end) - 1, filter="total")
        months.append({"month": start.strftime("%Y-%m"), "questions": d["total"], "complete": end <= now})
        start = end
    (OUT / "askubuntu_monthly_api.json").write_text(json.dumps(months, indent=1))
    print(f"monthly counts for {len(months)} months; {used} requests used", flush=True)


if __name__ == "__main__":
    main()
