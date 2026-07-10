---
description: Audit a specific violation-detection rule against the APFA contract language using the contract-reviewer subagent
argument-hint: [rule-name-or-file]
allowed-tools: Task, Read, Grep, Glob
---

Use the contract-reviewer subagent to audit the following rule from the contract-app CLI: $ARGUMENTS

Steps:
1. Locate the code implementing this rule (search by function/rule name if a name is given, or read the file directly if a path is given).
2. Locate the corresponding contract provision(s) in the repo's contract reference material.
3. Delegate to the contract-reviewer subagent with both pieces of context, and have it produce a findings report in its standard format (Rule / Contract basis / Code location / Status / Notes).
4. Print the subagent's findings report in full — do not summarize or shorten it.

If the rule name doesn't clearly match anything in the codebase or the contract reference material, stop and ask which rule/file is meant rather than guessing.
