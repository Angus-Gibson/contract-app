#!/usr/bin/env python3
"""
APFA Contract Violation Checker — Two-Pass Analysis

Pass 1: Haiku identifies flagged provisions AND selects follow-up questions
        in a single fast structured-output call (no thinking).
Pass 2: User answers the selected questions interactively.
Pass 3: Opus with adaptive thinking generates the final DirectConnect claims.
"""

import json
import sys
import textwrap
from pathlib import Path

import anthropic

PROVISIONS_PATH = Path(__file__).parent / "cba_provisions.json"

# ── Question bank ──────────────────────────────────────────────────────────────
# Haiku selects IDs from this bank; Python renders and collects answers.
# depends_on is enforced in Python (answer-dependent, not provision-dependent).

QUESTIONS = {
    "last_sequence": {
        "text": "Was this your last sequence or series of the bid month?",
        "type": "yn",
    },
    "had_reported": {
        "text": (
            "Had you already reported/signed in at the airport "
            "before being notified of the reschedule?"
        ),
        "type": "yn",
    },
    "split_or_replaced": {
        "text": (
            "Did the Company put you on a modified version of the same sequence, "
            "or were you replaced with entirely different flying?"
        ),
        "type": "choice",
        "choices": [
            "Split onto modified version of same sequence (10.L.1)",
            "Replaced with entirely different flying (10.L.3)",
        ],
        "depends_on": {"last_sequence": "yes"},
    },
    "lll_swap": {
        "text": "Was a Last Live Leg (LLL) Swap in effect on this sequence?",
        "type": "yn",
    },
    "has_premiums": {
        "text": (
            "Did your original sequence carry any premiums — "
            "Lead, Purser, Galley, International, or Speaker?"
        ),
        "type": "yn",
    },
}

# Canonical order for display
QUESTION_ORDER = [
    "last_sequence",
    "had_reported",
    "split_or_replaced",
    "lll_swap",
    "has_premiums",
]

# ── Pass 1: Haiku triage + question selection ──────────────────────────────────

TRIAGE_SYSTEM = """\
You are an expert APFA contract representative. In one step:

1. Analyze the flight attendant's situation against the CBA provisions and identify \
every provision that is POTENTIALLY triggered.

2. Select which follow-up questions are needed to confirm ambiguous provisions. \
Choose from exactly these IDs:

  "last_sequence"     — select if ANY 10.J, 10.K, 10.L, or 10.M provision is flagged
  "had_reported"      — select if any 10.J or 10.T provision is flagged
  "split_or_replaced" — select if any 10.L provision is flagged (shown only when \
last_sequence = yes)
  "lll_swap"          — select if any rescheduling provision (10.J/10.K/10.L/10.M/10.P) \
is flagged
  "has_premiums"      — select if any pay-protection provision (10.J/10.K/10.L/10.V/16) \
is flagged

Return only IDs that are genuinely needed. If all flagged provisions are "definite" \
and no ambiguity exists, return an empty list.\
"""

TRIAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "flagged_provisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "section":       {"type": "string"},
                    "title":         {"type": "string"},
                    "why_triggered": {"type": "string"},
                    "confidence":    {"type": "string", "enum": ["definite", "possible"]},
                },
                "required": ["section", "title", "why_triggered", "confidence"],
                "additionalProperties": False,
            },
        },
        "questions_to_ask": {
            "type": "array",
            "items": {"type": "string"},
        },
        "summary": {"type": "string"},
    },
    "required": ["flagged_provisions", "questions_to_ask", "summary"],
    "additionalProperties": False,
}

# ── Pass 3: Opus final output ──────────────────────────────────────────────────

FINAL_SYSTEM = """\
You are an expert APFA contract representative. You have the FA's situation, a \
preliminary list of potentially triggered provisions, and confirmed answers to \
targeted follow-up questions.

Using the confirmed answers, determine which provisions are DEFINITIVELY triggered \
and produce a ready-to-file DirectConnect claim for each one.

DIRECTCONNECT FORMAT — use exactly this block for each claim:
---
SECTION: [CBA section]
CLAIM: [one-sentence violation description]
REMEDY: [exact pay/credit calculation, or "TBD — provide: [specify what's needed]"]
DEADLINE: [filing deadline]
NOTES: [LLL Swap impact if applicable; Pay No Credit designation if applicable; \
premium carry-forward per 10.V.5 if applicable]
---

RULES:
- Output claims ONLY for definitively confirmed provisions.
- If LLL Swap is confirmed (yes), note its impact on pay protection in every \
  affected claim.
- If premiums are confirmed, list them by name in the 10.V.5 carry-forward claim.
- Keep each claim concise and ready to submit as-is.
- If a provision remains ambiguous after the follow-up answers, include it but note \
  exactly what additional information is still needed.\
"""

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_provisions() -> str:
    if not PROVISIONS_PATH.exists():
        sys.exit(f"ERROR: {PROVISIONS_PATH} not found.")
    with open(PROVISIONS_PATH) as f:
        return json.dumps(json.load(f), indent=2)


def get_situation() -> str:
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])

    print("=" * 60)
    print("APFA Contract Violation Checker")
    print("=" * 60)
    print()
    print("Describe your scheduling situation. Include: sequence")
    print("number, dates, what happened, actual vs. scheduled")
    print("times if known, and Lineholder or Reserve status.")
    print()
    print("Press Enter twice when done:")
    print()

    lines, blank = [], 0
    while blank < 1:
        try:
            line = input()
        except EOFError:
            break
        if line == "":
            blank += 1
        else:
            blank = 0
            lines.append(line)

    situation = "\n".join(lines).strip()
    if not situation:
        sys.exit("No description provided.")
    return situation


def run_triage(
    client: anthropic.Anthropic, situation: str, provisions_json: str
) -> dict:
    """Single Haiku call: returns flagged provisions + question IDs to ask."""
    print("\n[1/3] Scanning provisions (Haiku)...", flush=True)
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=2048,
        system=TRIAGE_SYSTEM,
        messages=[{
            "role": "user",
            "content": f"CBA PROVISIONS:\n{provisions_json}\n\nSITUATION:\n{situation}",
        }],
        output_config={
            "format": {"type": "json_schema", "schema": TRIAGE_SCHEMA}
        },
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def ask_questions(selected_ids: list, answers: dict | None = None) -> dict:
    """Walk QUESTION_ORDER, ask only IDs returned by triage, honour depends_on."""
    if answers is None:
        answers = {}

    print("\n[2/3] Follow-up questions")
    print("-" * 60)

    asked = 0
    for qid in QUESTION_ORDER:
        if qid not in selected_ids:
            continue

        q = QUESTIONS[qid]

        # Enforce depends_on
        skip = False
        for dep_id, required_answer in q.get("depends_on", {}).items():
            if answers.get(dep_id) != required_answer:
                skip = True
                break
        if skip:
            continue

        asked += 1
        print(f"\n[Q{asked}] {q['text']}")

        if q["type"] == "yn":
            while True:
                raw = input("       y/n: ").strip().lower()
                if raw in ("y", "yes"):
                    answers[qid] = "yes"
                    break
                if raw in ("n", "no"):
                    answers[qid] = "no"
                    break
                print("       Please enter y or n.")

        elif q["type"] == "choice":
            for i, choice in enumerate(q["choices"], 1):
                print(f"       {i}. {choice}")
            while True:
                raw = input(f"       Enter 1-{len(q['choices'])}: ").strip()
                if raw.isdigit() and 1 <= int(raw) <= len(q["choices"]):
                    answers[qid] = q["choices"][int(raw) - 1]
                    break
                print(f"       Please enter a number between 1 and {len(q['choices'])}.")

    if asked == 0:
        print("\n  No follow-up questions needed.")

    return answers


def stream_final_output(
    client: anthropic.Anthropic,
    situation: str,
    provisions_json: str,
    triage: dict,
    answers: dict,
) -> None:
    print("\n[3/3] Generating DirectConnect claims (Opus)...")
    print("=" * 60)
    print()

    flagged_text = "\n".join(
        f"  [{p['confidence'].upper()}] {p['section']}: {p['title']}\n"
        f"  {p['why_triggered']}"
        for p in triage["flagged_provisions"]
    )

    answers_text = (
        "\n".join(
            f"  Q: {QUESTIONS[qid]['text']}\n  A: {answers[qid]}"
            for qid in QUESTION_ORDER
            if qid in answers
        )
        or "  (no follow-up questions asked)"
    )

    user_message = (
        f"CBA PROVISIONS:\n{provisions_json}\n\n"
        f"ORIGINAL SITUATION:\n{situation}\n\n"
        f"POTENTIALLY TRIGGERED PROVISIONS:\n{flagged_text}\n\n"
        f"FOLLOW-UP ANSWERS:\n{answers_text}\n\n"
        "Generate the confirmed DirectConnect claim(s)."
    )

    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=4000,
        thinking={"type": "adaptive"},
        system=FINAL_SYSTEM,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        in_thinking = False
        for event in stream:
            if event.type == "content_block_start":
                if event.content_block.type == "thinking":
                    in_thinking = True
                    print("[Confirming triggered provisions...]\n", flush=True)
                elif event.content_block.type == "text" and in_thinking:
                    in_thinking = False
                    print()
            elif event.type == "content_block_delta":
                if event.delta.type == "text_delta":
                    print(event.delta.text, end="", flush=True)

    print("\n")
    print("=" * 60)
    print("File each claim via DirectConnect before the 96-hr deadline.")
    print("=" * 60)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    provisions_json = load_provisions()
    situation = get_situation()

    if len(sys.argv) <= 1:
        print()
        print("Situation entered:")
        print("-" * 40)
        for line in textwrap.wrap(situation, width=60):
            print(line)
        print("-" * 40)

    client = anthropic.Anthropic()

    # Pass 1 — Haiku triage (fast, no thinking)
    triage = run_triage(client, situation, provisions_json)

    if not triage["flagged_provisions"]:
        print(f"\nNo provisions triggered.\n{triage['summary']}")
        return

    print(f"\n  {len(triage['flagged_provisions'])} provision(s) flagged:")
    for p in triage["flagged_provisions"]:
        marker = "✓" if p["confidence"] == "definite" else "?"
        print(f"  [{marker}] {p['section']} — {p['title']}")

    # Pass 2 — follow-up questions (local, no API call)
    answers = ask_questions(triage["questions_to_ask"])

    # Pass 3 — Opus final output (adaptive thinking, streaming)
    stream_final_output(client, situation, provisions_json, triage, answers)


if __name__ == "__main__":
    main()
