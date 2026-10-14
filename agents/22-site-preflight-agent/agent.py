"""Site preflight agent.

A web agent that asks before it acts. Before every read, cite or connect on a
domain it gets a signed decision from CrawlCheck (https://crawlcheck.io), checks
the Ed25519 signature locally, and only then fetches, summarises or cites. Every
decision receipt is kept in receipts/ so the run can be audited afterwards.

No API key is needed for the decisions. OPENAI_API_KEY is optional: with it the
summary is written by gpt-4o-mini; without it the agent prints an extractive one.

    python agent.py --url https://example.com --task "what does this site say it is for"
"""
import argparse
import html
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

from crawlcheck import CrawlCheck

AGENT_NAME = "site-preflight-agent"
RECEIPTS = Path(__file__).parent / "receipts"


def load_env():
    p = Path(__file__).parent / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def domain_of(url):
    return re.sub(r"^www\.", "", urllib.request.urlparse(url).hostname or "")


def preflight(cc, domain, action, confirm):
    """One signed decision, verified locally, written to receipts/."""
    receipt = cc.preflight(domain, action=action, agent=AGENT_NAME)
    check = cc.verify(receipt)
    RECEIPTS.mkdir(exist_ok=True)
    out = RECEIPTS / f"{domain}-{action}-{int(time.time())}.json"
    out.write_text(json.dumps({"receipt": receipt, "verification": check}, indent=2))
    decision = receipt["decision"]
    why = "; ".join(s["why"] for s in receipt.get("steps", [])) or "no reason given"
    print(f"  [{action}] {decision} - {why}")
    print(f"           signature {'verified' if check['accepted'] else 'NOT accepted'} - {out.name}")
    if not check["accepted"]:
        return False
    if decision == "block" or decision == "unsupported":
        return False
    if decision == "require_confirmation":
        return confirm(f"{domain}: {action} needs a human yes ({why}). Continue?")
    return True  # allow or warn


def fetch_text(url):
    req = urllib.request.Request(url, headers={"user-agent": f"{AGENT_NAME}/1.0 (+https://crawlcheck.io/for-agents)"})
    with urllib.request.urlopen(req, timeout=20) as r:
        raw = r.read(400_000).decode("utf-8", "ignore")
    raw = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def summarise(task, text, url):
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=key)
            r = client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "Answer the task from the page text only. Three sentences. Say if the page does not answer it."},
                    {"role": "user", "content": f"Task: {task}\nPage: {url}\n\n{text[:12000]}"},
                ],
                temperature=0,
            )
            return r.choices[0].message.content.strip(), "gpt-4o-mini"
        except Exception as e:  # fall through to the extractive summary
            print(f"  (LLM unavailable: {e.__class__.__name__}; using an extractive summary)")
    words = set(re.findall(r"[a-z]{4,}", task.lower()))
    sentences = re.split(r"(?<=[.!?])\s+", text)
    scored = sorted(sentences, key=lambda s: -len(words & set(re.findall(r"[a-z]{4,}", s.lower()))))
    return " ".join(scored[:3])[:800] or text[:600], "extractive"


def main():
    load_env()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="https://example.com")
    ap.add_argument("--task", default="what does this site say it is for")
    ap.add_argument("--yes", action="store_true", help="answer yes to any confirmation prompt (CI)")
    a = ap.parse_args()
    domain = domain_of(a.url)
    if not domain:
        sys.exit("a full URL is needed, e.g. https://example.com")
    confirm = (lambda q: True) if a.yes else (lambda q: input(q + " [y/N] ").strip().lower() == "y")
    cc = CrawlCheck()

    print(f"\nTask: {a.task}\nSite: {a.url}\n\n1. Resolve - what is known about {domain}")
    answer = cc.resolve(domain)
    acts = answer.get("actions", {})
    print(f"  status {answer.get('status')} - observed {answer.get('freshness', {}).get('observed_at')}")
    print("  " + ", ".join(f"{k}: {v.get('decision')}" for k, v in acts.items() if isinstance(v, dict)))

    print("\n2. Preflight before reading")
    if not preflight(cc, domain, "read", confirm):
        print("\nStopped: reading this site was not allowed. Nothing was fetched.")
        return
    text = fetch_text(a.url)
    print(f"  fetched {len(text):,} characters")

    print("\n3. Preflight before citing")
    may_cite = preflight(cc, domain, "cite", confirm)

    print("\n4. Answer")
    summary, by = summarise(a.task, text, a.url)
    print(f"  ({by})\n  {summary}")
    print(f"\n  Source: {a.url}" if may_cite else "\n  (not cited: the cite decision did not allow it)")
    print(f"\nReceipts: {RECEIPTS}/ - verify any of them later with: python -m crawlcheck verify <file>")


if __name__ == "__main__":
    main()
