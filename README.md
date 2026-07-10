    # APFA Contract Violation Checker

    A command-line tool that analyzes American Airlines flight attendant scheduling situations against the APFA CBA
    and generates ready-to-file DirectConnect claims.

    ## How It Works

    The tool runs in three passes:

    **Pass 1 — Triage (Haiku)**
    Scans your situation description against all CBA provisions and identifies every potentially triggered
    provision. Fast, no extended thinking.

    **Pass 2 — Follow-up Questions**
    Asks targeted yes/no and multiple-choice questions to resolve ambiguous provisions. Questions are selected by
    triage — only what's needed is asked.

    **Pass 3 — Claims Generation (Opus)**
    Uses adaptive thinking to produce ready-to-file DirectConnect claim blocks for every definitively triggered
    provision, including exact remedy calculations and filing deadlines.

    ## Usage

    **Interactive mode:**
    ```bash
    python check_my_trip.py
    Describe your situation at the prompt. Include sequence number, dates, what happened, actual vs. scheduled
    times, and whether you are a Lineholder or Reserve.

    Single-line mode:
    python check_my_trip.py "Seq 1234, Oct 3. Cancelled at gate after report. Reassigned to different flying same
    day."

    Output Format

    Each confirmed claim is output as a DirectConnect block:

    ---
    SECTION: [CBA section]
    CLAIM: [one-sentence violation description]
    REMEDY: [pay/credit calculation or TBD with specifics needed]
    DEADLINE: [filing deadline]
    NOTES: [LLL Swap impact, Pay No Credit flags, premium carry-forward]
    ---

    Requirements

    pip install anthropic

    Requires an ANTHROPIC_API_KEY environment variable.

    Files

    ┌─────────────────────┬─────────────────────────────────────────────────────────────────────┐
    │        File         │                               Purpose                               │
    ├─────────────────────┼─────────────────────────────────────────────────────────────────────┤
    │ check_my_trip.py    │ Main CLI — triage, questions, claim generation                      │
    ├─────────────────────┼─────────────────────────────────────────────────────────────────────┤
    │ cba_provisions.json │ CBA provisions reference — sections, triggers, remedies, edge cases │
    └─────────────────────┴─────────────────────────────────────────────────────────────────────┘

    Provisions Coverage

    The tool currently covers:

    - Section 10.J — Rescheduling pay protections (TAFB, Duty, Sit rigs; notification windows; changeovers)
    - Section 10.K — Reserve rescheduling pay protections
    - Section 10.L — Last sequence protections (split and replaced flying)
    - Section 10.M — Changeover sequence pay protections
    - Section 10.P — Last Live Leg Swap (SWAPPED ONTO: no pay, no protection; GAVE AWAY: full sequence pay structure
     retained with pre-swap sequence value anchor)
    - Section 10.T — Double-covered positions, misawards, crew scheduling errors (150% provision)
    - Section 10.V — Premium carry-forward
    - Section 16 — Speaker, International premiums

    Notes

    - File each DirectConnect claim before the 96-hour deadline
    - This tool assists claim identification — always verify against the current APFA CBA
    - LLL Swap cases automatically trigger full 10.P analysis including rig recalculation flags
