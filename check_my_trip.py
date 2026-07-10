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
    "lll_swap_role": {
        "text": "In that LLL Swap, what was your role?",
        "type": "choice",
        "choices": [
            "I swapped ONTO another FA's last live leg (I flew the leg)",
            "I gave AWAY my last live leg (another FA flew it for me)",
        ],
        "depends_on": {"lll_swap": "yes"},
    },
    "is_reserve": {
        "text": "Are you a Reserve (as opposed to a Lineholder)?",
        "type": "yn",
        "depends_on": {"lll_swap": "yes"},
    },
    "lll_highest_value_leg": {
        "text": (
            "Was the leg you gave away the highest-value leg in your sequence "
            "(i.e., would removing it reduce your sequence's total value)?"
        ),
        "type": "yn",
        "depends_on": {
            "lll_swap_role": "I gave AWAY my last live leg (another FA flew it for me)"
        },
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
    "lll_swap_role",
    "is_reserve",
    "lll_highest_value_leg",
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
is flagged, OR if the situation text contains any mention of "last live leg", "LLL", \
or "swap" — select this regardless of which provisions are flagged
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
- If LLL Swap is confirmed (yes), apply 10.P.4 based on the FA's role:
  * SWAPPED ONTO the LLL: this FA receives NO pay for any flights flown under \
the swap AND NO pay protection if their own sequence is disrupted. Explicitly \
suppress any pay or pay-protection claim that would otherwise arise, and note \
"10.P.4 — LLL Swap recipient: no pay, no pay protection" in each affected claim. \
This applies equally if the FA is a Reserve on a day off — Reserve status does \
not create any pay entitlement for LLL Swap flying (10.P.4). Additionally, \
flag the following rig recalculations — these are not suppressed, they still \
apply based on actual times: (a) Sit Rig: if the swap altered segment timing \
within a duty period, recalculate sit rig based on actual sit times after the \
swap; (b) Duty Rig: duty time for this FA ends at the actual release time of \
the LLL leg flown.
  * GAVE AWAY the LLL: this FA retains their original sequence pay protections \
in full. SEQUENCE VALUE ANCHOR: all pay calculations (10.J.10, 10.K, 10.L.1, \
10.L.3, 10.M, 10.T.3 150%) must use the original published sequence value \
BEFORE the swap — never the post-swap composition. If the LLL was the \
highest-value leg (confirmed yes), flag this explicitly: excluding it would \
reduce the sequence value and therefore the 150% base — use pre-swap value. \
If the FA has not provided the original sequence value, include in REMEDY: \
"TBD — provide original published sequence value (pre-swap)." Note "10.P — \
original sequence pay structure retained; sequence value = pre-swap original" \
in each affected claim. Additionally, flag the following rig provisions — \
these require recalculation based on the swap: (a) TAFB Rig: if the swap \
changed sequence composition, recalculate effective TAFB against the original \
sequence; (b) Duty Rig: duty time for the giving-away FA continues per the \
original sequence release time, not the LLL leg's actual release; (c) Sit Rig: \
if the swap altered segment timing within a duty period, recalculate sit rig \
based on actual sit times after the swap.
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

    if not triage["flagged_provisions"] and "lll_swap" not in triage["questions_to_ask"]:
        print(f"\nNo provisions triggered.\n{triage['summary']}")
        return

    print(f"\n  {len(triage['flagged_provisions'])} provision(s) flagged:")
    for p in triage["flagged_provisions"]:
        marker = "✓" if p["confidence"] == "definite" else "?"
        print(f"  [{marker}] {p['section']} — {p['title']}")

    # Pass 2 — follow-up questions (local, no API call)
    # lll_swap_role is always needed when lll_swap is selected; inject it here
    # rather than having Haiku decide — the depends_on gate handles display.
    selected = list(triage["questions_to_ask"])
    if "lll_swap" in selected:
        idx = selected.index("lll_swap") + 1
        for qid in ["lll_swap_role", "is_reserve", "lll_highest_value_leg"]:
            if qid not in selected:
                selected.insert(idx, qid)
                idx += 1
    answers = ask_questions(selected)

    # Pass 3 — Opus final output (adaptive thinking, streaming)
    stream_final_output(client, situation, provisions_json, triage, answers)


if __name__ == "__main__":
    main()
