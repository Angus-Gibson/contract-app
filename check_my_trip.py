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
            "Split onto modified version of same sequence",
            "Replaced with entirely different flying",
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
        "depends_on": {"lll_swap": "yes", "lll_company_approved": "yes"},
    },
    "lll_company_approved": {
        "text": (
            "Was the LLL Swap electronically submitted and approved by the Company "
            "(within the required 15-minute window)?"
        ),
        "type": "yn",
        "depends_on": {"lll_swap": "yes"},
    },
    "is_reserve": {
        "text": "What is your status on this day?",
        "type": "choice",
        "choices": [
            "Lineholder",
            "Reserve — on a day off (RDO)",
            "Reserve — on an active Reserve duty day",
        ],
        "depends_on": {"lll_swap_role": "I swapped ONTO another FA's last live leg (I flew the leg)"},
    },
    # NOTE: lll_was_deadhead applies to both roles. Role-specific FINAL_SYSTEM
    # handling (swapped-onto: legality check; gave-away: TAFB impact) is driven by
    # the lll_swap_role answer already recorded — no role gate needed here.
    "lll_was_deadhead": {
        "text": (
            "Was the Last Live Leg originally scheduled as a deadhead "
            "(repositioning flight) that converted to live flying?"
        ),
        "type": "yn",
        "depends_on": {"lll_swap": "yes", "lll_company_approved": "yes"},
    },
    "lll_highest_value_leg": {
        "text": (
            "Was the leg you gave away the highest-value leg in your sequence "
            "(i.e., would removing it reduce your sequence's total value)? "
            "Note: your answer also affects any 10.T.3 150% error-pay base — "
            "the LLL's value is part of the pre-swap sequence value that anchors that multiplier."
        ),
        "type": "yn",
        "depends_on": {
            "lll_swap_role": "I gave AWAY my last live leg (another FA flew it for me)"
        },
    },
    "gave_away_reserve_status": {
        "text": "Are you a Reserve (as opposed to a Lineholder)?",
        "type": "yn",
        "depends_on": {"lll_swap_role": "I gave AWAY my last live leg (another FA flew it for me)"},
    },
    "gave_away_reserve_day_type": {
        "text": "What is your Reserve status on this day?",
        "type": "choice",
        "choices": [
            "Reserve — on a day off (RDO)",
            "Reserve — on an active Reserve duty day",
        ],
        "depends_on": {"gave_away_reserve_status": "yes"},
    },
    "gave_away_reserve_remaining_days": {
        "text": "At the time of the LLL Swap disruption, did you have any remaining Reserve days in this bid period?",
        "type": "yn",
        "depends_on": {"gave_away_reserve_status": "yes"},
    },
    "is_changeover_sequence": {
        "text": (
            "Was this a changeover sequence "
            "(i.e., did it carry over credit/value from the prior bid month)?"
        ),
        "type": "yn",
        "depends_on": {
            "lll_swap": "yes",
            "lll_company_approved": "yes",
            "lll_swap_role": "I gave AWAY my last live leg (another FA flew it for me)",
        },
    },
    "held_carryover_at_conversion": {
        "text": (
            "Were you holding (awarded) this sequence as a carryover at the time it "
            "converted to a changeover sequence, or did you pick it up after conversion?"
        ),
        "type": "choice",
        "choices": [
            "I was holding it as a carryover at the time of conversion",
            "I picked it up after it was already a changeover sequence",
        ],
        "depends_on": {"is_changeover_sequence": "yes"},
    },
    "post_swap_company_action": {
        "text": (
            "After the LLL Swap was completed (or attempted), did the Company take any "
            "separate scheduling action that disrupted your remaining flying "
            "(e.g., cancel, reroute, or reassign a different sequence)?"
        ),
        "type": "yn",
        "depends_on": {"lll_swap": "yes"},
    },
    "has_premiums": {
        "text": (
            "Did your original sequence carry any premiums — "
            "Lead, Purser, Galley, International, or Speaker?"
        ),
        "type": "yn",
    },
    "premium_types": {
        "text": (
            "Which premiums did your sequence carry? "
            "(Enter all that apply, e.g.: Lead, International)"
        ),
        "type": "freetext",
        "depends_on": {"has_premiums": "yes"},
    },
}

# Canonical order for display
QUESTION_ORDER = [
    "last_sequence",
    "had_reported",
    "split_or_replaced",
    "lll_swap",
    "lll_company_approved",
    "lll_swap_role",
    "is_reserve",
    "lll_was_deadhead",
    "post_swap_company_action",
    "lll_highest_value_leg",
    "gave_away_reserve_status",
    "gave_away_reserve_day_type",
    "gave_away_reserve_remaining_days",
    "is_changeover_sequence",
    "held_carryover_at_conversion",
    "has_premiums",
    "premium_types",
]

# ── Pass 1: Haiku triage + question selection ──────────────────────────────────

TRIAGE_SYSTEM = """\
You are an expert APFA contract representative. In one step:

1. Analyze the flight attendant's situation against the CBA provisions and identify \
every provision that is POTENTIALLY triggered.

2. Select which follow-up questions are needed to confirm ambiguous provisions. \
Choose from exactly these IDs:

  "last_sequence"     — select if ANY 10.J, 10.K, 10.L, or 10.M provision is flagged
  "had_reported"      — select if any 10.J or 10.T provision is flagged, OR if \
lll_swap is selected (because a post-swap Company action may trigger §10.T.3 \
independently, requiring the post-report threshold to be confirmed)
  "split_or_replaced" — select if any 10.L provision is flagged (shown only when \
last_sequence = yes)
  "lll_swap"          — select if any rescheduling provision (10.J/10.K/10.L/10.M/10.P) \
is flagged, OR if the situation text contains any mention of "last live leg", "LLL", \
or "LLL swap" or "last live leg swap" — select this regardless of which provisions are flagged
  "has_premiums"      — select if any pay-protection provision (10.J/10.K/10.L/10.V/16) \
is flagged

Return only IDs that are genuinely needed. If all flagged provisions are "definite" \
and no ambiguity exists, return an empty list.

IMPORTANT — sequence value: if any pay-protection provision (10.J.10, 10.K, 10.L, \
10.M, 10.T.3) is flagged alongside any LLL-related provision or keyword, mark the \
sequence value in why_triggered as PROVISIONAL — do not anchor it to the post-swap \
composition. The correct pre-swap value will be confirmed in Pass 3.

NOTE — do NOT select these IDs; they are injected automatically by the application \
when lll_swap is selected: "lll_company_approved", "lll_swap_role", "is_reserve", \
"lll_was_deadhead", "post_swap_company_action", "lll_highest_value_leg", \
"gave_away_reserve_status", "gave_away_reserve_day_type", \
"gave_away_reserve_remaining_days", "is_changeover_sequence", \
"held_carryover_at_conversion". The only ID you may select that is also \
auto-injected is "has_premiums" — selecting it is harmless as the application \
deduplicates.\
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
            "items": {
                "type": "string",
                "enum": ["last_sequence", "had_reported", "split_or_replaced", "lll_swap", "has_premiums"],
            },
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
- Output claims ONLY for definitively confirmed provisions — confirmed either \
by the triage flagged-provisions list OR by the follow-up answers (e.g., \
LLL Swap confirmed yes in answers triggers 10.P analysis even if no provision \
was formally flagged by triage).
- Reporting threshold (§10.T.3): if had_reported is in the follow-up answers, \
use it to gate all 10.T.3 claims: if had_reported = yes, the FA was post-report \
at the time of the disruption — §10.T.3 eligibility threshold is confirmed; if \
had_reported = no, the disruption was pre-report — suppress §10.T.3 and analyze \
under §10.J.2/§10.J.10 instead. Apply this gate to every 10.T.3 claim regardless \
of LLL Swap status. If had_reported was not asked or not answered, include any \
10.T.3 claim and note "Verify: was the FA post-report (signed in at the airport) \
at the time of the disruption? If pre-report, §10.T.3 does not apply."
- If LLL Swap is confirmed (yes), first check approval: if lll_company_approved = no, \
do NOT apply the 10.P.4 framework — the swap was not Company-authorized. Treat the \
scheduling action as a Company-initiated change and analyze under the applicable \
sections (10.T.3, 10.J, 10.K, 10.L, 10.M) as if the LLL swap never occurred. Note \
in each claim: "LLL Swap approval not confirmed — 10.P.4 framework does not apply; \
analyzed as Company-initiated scheduling change." Disregard any PROVISIONAL sequence value markers in the triage output. For any \
10.T.3 150% calculation, use the original published sequence value BEFORE the \
attempted swap — do not use post-disruption actual flying value, as the \
attempted-but-unapproved swap may itself be the Company error triggering 10.T.3. \
If post_swap_company_action = yes, a SECOND independent Company action also occurred \
after the failed swap attempt — generate standalone claims for each confirmed \
independent violation; both the attempted swap (as a Company-initiated scheduling \
error) and the second action may separately trigger §10.T.3. For the attempted swap \
(event 1), apply the had_reported gate normally. For the second independent action \
(event 2), if had_reported = no was anchored to the first disruption, flag in NOTES: \
"Verify: was the FA post-report at the time of the SECOND Company action? If yes, \
§10.T.3 may apply to that event independently even if had_reported = no for the \
first event." If post_swap_company_action = no, analyze the attempted swap as the \
sole Company-initiated disruption event and do not generate a second independent \
claim. Additionally, if the attempted swap caused actual flying different from the \
original published sequence (e.g., the FA flew or was released from a leg outside \
their original schedule), flag Duty Rig (§11.D.5), Sit Rig (§11.D.6), and TAFB \
Rig (§2.AAA/§11.D.4) for recalculation and generate standalone DirectConnect \
claim blocks per the rig rules at the end of these instructions. \
If lll_company_approved = yes, proceed with the role-based 10.P.4 analysis below. \
If lll_company_approved was not answered, treat as unapproved (Company-initiated \
framework) and flag in NOTES: "LLL Swap approval status not confirmed — applying \
Company-initiated framework; verify approval status before filing." Apply the same \
post_swap_company_action two-event logic as the lll_company_approved=no path: if \
post_swap_company_action = yes, generate standalone claims for each confirmed \
independent violation with the per-event had_reported gate and second-event \
verification note; if post_swap_company_action = no, analyze the attempted swap \
as the sole disruption event and do not generate a second independent claim.
- If LLL Swap is confirmed and approved, apply 10.P.4 based on the FA's role:
  * SWAPPED ONTO the LLL: this FA receives NO pay for any flights flown under \
the swap AND NO pay protection if their own sequence is disrupted. Explicitly \
suppress any pay or pay-protection claim that would otherwise arise, and note \
"10.P.4 — LLL Swap recipient: no pay, no pay protection" in each affected claim. \
Reserve status does not create any pay entitlement for LLL Swap flying (10.P.4). \
Check is_reserve: if "Reserve — on a day off (RDO)", explicitly cite §10.P edge \
case 1 in NOTES: "Reserve on day off confirmed — 10.P.4 applies identically; no \
pay, no pay protection." If "Reserve — on an active Reserve duty day", note §10.P.4 bars \
additional LLL swap pay — but pre-swap Reserve duty day pay owed independently of \
the swap may survive §10.P.4. Flag in NOTES: "Active Reserve duty day confirmed — \
pre-swap Reserve guarantee/duty pay is not extinguished by 10.P.4; verify with \
Local whether accrued duty pay remains owed. LLL swap pay itself: not owed \
(10.P.4 applies)." If "Lineholder", standard 10.P.4 applies. If lll_was_deadhead = yes, flag in NOTES: "Deadhead-to-live conversion \
confirmed — swapping FA was required to work the live segment; verify crew legality \
for the live flight. 10.P.4 still applies; no pay for the converted live leg." \
Additionally, note that §10.M.2 (Company-initiated split pay protection) and \
§10.J.9.a (equipment downgrade pay protection) are barred by 10.P.4 for actions \
caused by the LLL swap itself — do not generate claims for swap-caused disruptions \
under either section. Exception: if post_swap_company_action = yes, an independent \
Company action after swap completion may trigger §10.M.2 or §10.J.9.a independently \
— apply the same causal test used for 10.T.3 and generate those claims if the \
triggering event is the independent action, not the swap. \
Flag the following rig recalculations — these are not suppressed, they still \
apply based on actual times: (a) Sit Rig: if the swap altered segment timing \
within a duty period, recalculate sit rig based on actual sit times after the \
swap; (b) Duty Rig: duty time for this FA ends at the actual release time of \
the LLL leg flown; (c) TAFB Rig: a swapped-onto FA who flew a leg from another \
FA's sequence always has their sequence composition changed — this condition is \
definitively satisfied; generate the standalone TAFB rig claim (§2.AAA/11.D.4) \
and recalculate effective TAFB based on actual times of the LLL leg flown.
  * GAVE AWAY the LLL: this FA retains their original sequence pay protections \
in full. SEQUENCE VALUE ANCHOR: all pay calculations (10.J.10, 10.K, 10.L.1, \
10.L.3, 10.M, 10.T.3 150%) must use the original published sequence value \
BEFORE the swap — never the post-swap composition. If the LLL was the \
highest-value leg (confirmed yes), flag this explicitly: excluding it would \
reduce the sequence value and therefore the 150% base — use pre-swap value. \
If is_changeover_sequence = yes, check held_carryover_at_conversion: if the FA \
was holding the sequence as a carryover at the time of conversion, apply a DOUBLE \
ANCHOR: (1) use the carryover value from the prior bid month as the sequence base \
— not the published changeover-sequence value; (2) then apply the pre-swap anchor \
— never the post-swap composition. Note in REMEDY: "DOUBLE ANCHOR: (1) carryover \
value from prior bid month (§10.J.12 — FA held carryover at conversion); (2) \
pre-swap value — not post-swap composition." If the FA picked it up after \
conversion, use only the single pre-swap anchor — the published changeover-sequence \
value is the base; the prior-bid-month carryover value does not apply. \
If the FA has not provided the original sequence value, include in REMEDY: \
"TBD — provide original published sequence value (pre-swap)." Note "10.P — \
original sequence pay structure retained; sequence value = pre-swap original" \
in each affected claim. If gave_away_reserve_status = yes (FA is a Reserve), check gave_away_reserve_day_type: \
if "Reserve — on an active Reserve duty day", flag in NOTES: "Active Reserve duty day \
confirmed — pre-swap Reserve guarantee/duty pay is not extinguished by any pay \
adjustment; verify with Local whether accrued duty pay remains owed independently." \
If "Reserve — on a day off (RDO)", flag in NOTES: "Giving-away FA was on a Reserve \
day off — verify with Local whether any day-off protections under §12.B.2.d interact \
with the retained pay protections; no active duty obligation exists for this day." \
If gave_away_reserve_remaining_days is not in answers, the FA is a Lineholder — \
skip 10.L.6 entirely; the Reserve eligibility gate does not apply. Otherwise, \
apply 10.L.6 eligibility using gave_away_reserve_remaining_days: if \
gave_away_reserve_remaining_days = no (no remaining Reserve days at time of \
disruption), the Reserve qualifies for 10.L.1 — full pay+credit, no fly obligation; \
if gave_away_reserve_remaining_days = yes (remaining Reserve days existed), 10.L.4 \
applies instead — pay protected for cancelled portions, FA must still fly uncancelled \
portions. Cite the applicable section (10.L.1 or 10.L.4) in SECTION and CLAIM. \
Apply a causal test using post_swap_company_action: (a) if post_swap_company_action \
= yes (a separate Company action disrupted the FA's remaining flying after swap \
completion), apply the had_reported gate before generating the 10.T.3 claim — if \
had_reported = yes, generate a standalone 10.T.3 claim and note "10.T.3 survives — \
caused by independent Company action after swap completion"; if had_reported = no \
(anchored to first disruption), flag in NOTES: "Verify: was the FA post-report at \
the time of this Company action? If yes, §10.T.3 applies independently"; if \
had_reported was not answered, include the 10.T.3 claim with the standard \
"Verify: was the FA post-report?" note; (b) if post_swap_company_action = no, no \
standalone post-swap 10.T.3 claim is warranted under the gave-away framework; \
(c) if post_swap_company_action was not answered, apply the had_reported gate as in \
sub-case (a) and note "Causal determination required — confirm whether a separate \
Company action disrupted this FA's flying after swap completion." \
If lll_was_deadhead = yes, a deadhead-to-live conversion confirms the FA's sequence \
composition changed (the deadhead segment was replaced by live flying) — treat this \
as satisfying the "swap changed sequence composition" condition below and generate \
the standalone TAFB rig claim (§2.AAA/11.D.4). Note in NOTES: "Deadhead-to-live \
conversion confirmed — giving-away FA was released from the live segment; TAFB \
recalculation required against original sequence." \
Additionally, flag the following rig provisions — \
these require recalculation based on the swap: (a) TAFB Rig: if the swap \
changed sequence composition (including when lll_was_deadhead = yes, which \
definitively satisfies this condition), recalculate effective TAFB against the \
original sequence; (b) Duty Rig: duty time for the giving-away FA continues per \
the original sequence release time, not the LLL leg's actual release; (c) Sit Rig: \
if the swap altered segment timing within a duty period, recalculate sit rig \
based on actual sit times after the swap.
- 10.T.3 suppression scope (swapped-onto FA only): §10.P.4 suppresses pay and \
pay-protection claims caused by the LLL swap itself — it does not reach independent \
Company violations that occur after the swap is complete. Before suppressing any \
10.T.3 claim for a swapped-onto FA, apply a causal test using post_swap_company_action: \
(a) if post_swap_company_action = no (no separate Company action after the swap), \
the scheduling error that triggers 10.T.3 was the LLL swap itself — suppress the \
10.T.3 claim under 10.P.4 and note "10.T.3 suppressed — violation caused by LLL \
swap; 10.P.4 bars claim"; (b) if post_swap_company_action = yes (a separate Company \
action occurred after swap completion), that 10.T.3 claim is independent of the \
swap and survives 10.P.4 — generate the 10.T.3 claim and note "10.T.3 survives \
10.P.4 — caused by independent Company action after swap completion"; (c) if \
post_swap_company_action was not answered, include the 10.T.3 claim and note in \
NOTES: "Causal determination required — confirm whether this violation was caused \
by the LLL swap itself (suppressed under 10.P.4) or by an independent Company \
action after swap completion (not suppressed)."
- If premiums are confirmed (has_premiums = yes), list the specific premiums from \
premium_types in the 10.V.5 carry-forward claim. If premium_types was not answered \
or is "(not provided)", include in REMEDY: "TBD — FA to specify which premiums \
(Lead, Purser, Galley, International, Speaker) were carried."
- split_or_replaced / lll_swap interaction: if split_or_replaced is in the answers, \
check for overlap — if the "entirely different flying" IS the LLL swap leg itself, \
do not generate a §10.L.3 claim (§10.P governs); if they are separate events, both \
may generate independent claims; note any interaction in NOTES. If split_or_replaced \
is absent from answers (§10.L was not flagged by triage), examine the situation \
narrative for any sequence modification — if the narrative suggests a sequence change \
occurred, note in NOTES: "Sequence modification may have occurred — §10.L analysis \
not performed; rep should verify whether the LLL swap also constitutes a schedulable \
sequence change."
- Rig recalculations (Sit Rig, Duty Rig, TAFB Rig): whenever a rig recalculation \
is flagged, generate a STANDALONE DirectConnect claim block — do not bury it in \
NOTES of another claim. You will not have actual block/sit/duty times; always use \
REMEDY = "TBD — provide: [specify exact times needed for this rig calculation]". \
SECTION: §11.D.5 (Duty Rig), §11.D.6 (Sit Rig), or §2.AAA/11.D.4 (TAFB Rig). \
DEADLINE: standard 96-hr window. Generate the block regardless of whether you can \
confirm the direction of underpayment — the rep needs the placeholder to collect \
times and calculate.
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
    print("When finished, enter a blank line to submit:")
    print()

    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "":
            if lines:
                break
            print("  (No input yet — please describe your situation first.)")
        else:
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
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
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
    except anthropic.APIError as e:
        sys.exit(f"ERROR: Triage API call failed — {e}")
    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        sys.exit("ERROR: Triage response contained no text block.")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        sys.exit(f"ERROR: Triage returned malformed JSON — {e}\nRaw response:\n{text}")


def ask_questions(selected_ids: list, answers: dict | None = None, detecting: bool = False) -> dict:
    """Walk QUESTION_ORDER, ask only IDs returned by triage, honour depends_on."""
    if answers is None:
        answers = {}

    if detecting:
        print("\n[2/3] Investigating potential claim")
    else:
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

        elif q["type"] == "freetext":
            raw = input("       Answer: ").strip()
            answers[qid] = raw if raw else "(not provided)"

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
    ) or f"  (None formally flagged by triage — LLL Swap confirmed yes via follow-up answers; base all 10.P analysis on answers below)\n  Triage summary: {triage['summary']}"

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
            elif event.type == "content_block_stop" and in_thinking:
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
    required = {"flagged_provisions", "questions_to_ask", "summary"}
    if not required.issubset(triage):
        sys.exit(f"ERROR: Triage response missing required keys. Got: {list(triage.keys())}")

    if not triage["flagged_provisions"] and "lll_swap" not in triage["questions_to_ask"]:
        print(f"\nNo provisions triggered.\n{triage['summary']}")
        return

    if triage["flagged_provisions"]:
        print(f"\n  {len(triage['flagged_provisions'])} provision(s) flagged:")
        for p in triage["flagged_provisions"]:
            marker = "✓" if p["confidence"] == "definite" else "?"
            print(f"  [{marker}] {p['section']} — {p['title']}")
    else:
        print("\n  No provisions formally flagged by keyword scan.")
        print("  LLL swap keyword detected — continuing to follow-up questions.")
    print(f"  {triage['summary']}")

    # Pass 2 — follow-up questions (local, no API call)
    # lll_swap_role is always needed when lll_swap is selected; inject it here
    # rather than having Haiku decide — the depends_on gate handles display.
    selected = list(triage["questions_to_ask"])
    if "split_or_replaced" in selected and "last_sequence" not in selected:
        selected.insert(selected.index("split_or_replaced"), "last_sequence")
    if "lll_swap" in selected and "last_sequence" not in selected:
        selected.insert(0, "last_sequence")
    if "lll_swap" in selected:
        idx = selected.index("lll_swap") + 1
        for qid in [
            "lll_company_approved", "lll_swap_role", "is_reserve",
            "lll_was_deadhead", "post_swap_company_action", "lll_highest_value_leg",
            "gave_away_reserve_status", "gave_away_reserve_day_type",
            "gave_away_reserve_remaining_days",
            "is_changeover_sequence", "held_carryover_at_conversion",
        ]:
            if qid not in selected:
                selected.insert(idx, qid)
                idx += 1
        # had_reported is always needed on LLL paths — 10.T.3 may arise from
        # post_swap_company_action=yes on either role branch regardless of whether
        # Haiku flagged a 10.J/10.T provision.
        if "had_reported" not in selected:
            selected.append("had_reported")
        required_lll = [
            "lll_company_approved", "lll_swap_role", "is_reserve",
            "lll_was_deadhead", "post_swap_company_action", "lll_highest_value_leg",
            "gave_away_reserve_status", "gave_away_reserve_day_type",
            "gave_away_reserve_remaining_days",
            "is_changeover_sequence", "held_carryover_at_conversion",
        ]
        # Dependency chain note: ask_questions() skips questions whose depends_on
        # parent was not met, so presence in selected is sufficient — skipping at
        # display time is correct and intentional. This check guards against the
        # injection loop itself failing to insert a required ID.
        if not all(q in selected for q in required_lll):
            sys.exit("ERROR: Injection block integrity failure — LLL sub-questions missing from selected list.")
        if "has_premiums" not in selected:
            selected.append("has_premiums")
    pay_protection_prefixes = ("10.J", "10.K", "10.L", "10.V", "16")
    if "has_premiums" not in selected and any(
        p["section"].startswith(pay_protection_prefixes)
        for p in triage["flagged_provisions"]
    ):
        selected.append("has_premiums")
    if "has_premiums" in selected and "premium_types" not in selected:
        selected.append("premium_types")
    answers = ask_questions(selected, detecting=not bool(triage["flagged_provisions"]))

    if not triage["flagged_provisions"] and answers.get("lll_swap") != "yes":
        print(f"\nNo LLL Swap confirmed and no provisions formally flagged — no claim to file.\n{triage['summary']}")
        return

    # Pass 3 — Opus final output (adaptive thinking, streaming)
    stream_final_output(client, situation, provisions_json, triage, answers)


if __name__ == "__main__":
    main()
