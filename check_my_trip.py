#!/usr/bin/env python3
"""
APFA Contract Violation Checker
Reads cba_provisions.json, accepts a natural language trip description,
and returns matched provisions + a pre-filled DirectConnect claim message.
"""

import json
import os
import sys
import textwrap
from pathlib import Path

import anthropic

PROVISIONS_PATH = Path(__file__).parent / "cba_provisions.json"

SYSTEM_PROMPT = """\
You are an expert APFA contract representative for American Airlines flight attendants. \
You have deep knowledge of the 2024 APFA-AA Collective Bargaining Agreement.

You will be given:
1. The full structured JSON ruleset of all compensable provisions from the CBA
2. A flight attendant's natural language description of their scheduling situation

Your job is to:
1. Identify ALL provisions whose trigger_condition is met by the described situation
2. For each matched provision, confirm why it applies and calculate the remedy where \
possible (use "TBD — verify actual vs. scheduled times" when times are not provided)
3. Produce a ready-to-submit DirectConnect claim message for each matched provision

DIRECTCONNECT MESSAGE FORMAT:
---
DATE: [use the date from the description, or "DATE TBD"]
FLIGHT/SEQUENCE: [sequence or flight number from description, or "SEQ TBD"]
SECTION: [CBA section number]
CLAIM: [one-sentence description of the violation]
REMEDY: [dollar amount or credit/pay calculation, or "TBD — verify actuals"]
NOTES: [any supporting detail, timing constraints, or filing deadlines]
---

CRITICAL RULES:
- Flag any Last Live Leg Swap involvement — it materially alters pay protection
- Distinguish clearly between Lineholder and Reserve rules
- Note the 96-hour filing deadline for misaward claims (10.T.2)
- Pay No Credit items do NOT count toward monthly max
- If multiple provisions apply, produce a separate DirectConnect message for each
- If a situation is ambiguous (e.g., missing actual times), state what info is needed \
to confirm the violation and calculate the exact remedy

If no provisions are triggered, say so clearly and explain why.\
"""


def load_provisions() -> str:
    if not PROVISIONS_PATH.exists():
        sys.exit(f"ERROR: {PROVISIONS_PATH} not found. Run the extraction step first.")
    with open(PROVISIONS_PATH) as f:
        provisions = json.load(f)
    return json.dumps(provisions, indent=2)


def get_situation() -> str:
    """Get trip description from stdin or command-line argument."""
    if len(sys.argv) > 1:
        # Accept description as command-line argument(s)
        return " ".join(sys.argv[1:])

    # Interactive mode
    print("=" * 60)
    print("APFA Contract Violation Checker")
    print("=" * 60)
    print()
    print("Describe your scheduling situation in plain English.")
    print("Include: sequence number, dates, what happened,")
    print("actual vs. scheduled departure/arrival times if known,")
    print("and whether you are a Lineholder or Reserve FA.")
    print()
    print("Enter description (press Enter twice when done):")
    print()

    lines = []
    blank_count = 0
    while blank_count < 1:
        try:
            line = input()
        except EOFError:
            break
        if line == "":
            blank_count += 1
        else:
            blank_count = 0
            lines.append(line)

    situation = "\n".join(lines).strip()
    if not situation:
        sys.exit("No description provided.")
    return situation


def stream_analysis(situation: str, provisions_json: str) -> None:
    client = anthropic.Anthropic()

    user_message = f"""\
CBA PROVISIONS JSON:
{provisions_json}

FLIGHT ATTENDANT'S SITUATION:
{situation}

Please analyze this situation against the CBA provisions and produce:
1. A list of all matched/triggered provisions with explanations
2. A pre-filled DirectConnect claim message for each matched provision\
"""

    print()
    print("=" * 60)
    print("Analyzing your situation against the CBA...")
    print("=" * 60)
    print()

    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=8000,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        in_thinking = False
        for event in stream:
            if event.type == "content_block_start":
                if event.content_block.type == "thinking":
                    in_thinking = True
                    print("[Analyzing contract provisions...]\n", flush=True)
                elif event.content_block.type == "text":
                    if in_thinking:
                        print()  # blank line after thinking indicator
                    in_thinking = False
            elif event.type == "content_block_delta":
                if event.delta.type == "text_delta":
                    print(event.delta.text, end="", flush=True)

    print()  # final newline
    print()
    print("=" * 60)
    print("Analysis complete.")
    print("File each DirectConnect claim before the 96-hour deadline.")
    print("=" * 60)


def main():
    provisions_json = load_provisions()
    situation = get_situation()

    if len(sys.argv) <= 1:
        # Echo back in interactive mode so the user can confirm
        print()
        print("Situation entered:")
        print("-" * 40)
        for line in textwrap.wrap(situation, width=60):
            print(line)
        print("-" * 40)

    stream_analysis(situation, provisions_json)


if __name__ == "__main__":
    main()
