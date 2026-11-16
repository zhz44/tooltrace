"""Tool classifier for bash commands issued by agentic LLMs.

Standalone: no dependencies beyond stdlib. Importable from both the
sitecustomize hook (live tracing) and offline analysis scripts
(re-classifying preserved command_raw fields).

Design: each bash command is classified into a single tool category by
identifying the *dominant-cost* command in the pipeline. We handle:

  1. Heredoc writes: `cat <<EOF > file ... EOF` → "heredoc_write"
     (semantically a write, not a cat)
  2. Leading env-var assignments: `PATH=/foo FOO=bar grep ...`
     (skip until we find a non-assignment token)
  3. `cd X && actual_command`: skip the cd, classify the rest
  4. Compound commands (`&&`, `||`, `;`, `|`): pick the highest-rank
     tool across all parts. Rank reflects expected wall-clock cost.

The classifier is intentionally lossy — `command_raw` is preserved in
the JSONL so we can always re-classify offline if categories change.
"""

from __future__ import annotations

import re

# Tool rank table. Higher rank = more dominant in a compound command.
# Rough buckets: 100s = slow (test/build), 50s = moderate (search/git),
# 20s = fast (text), 10s = trivial (echo/cd). Calibrate from real data
# once we have it; for now these are educated guesses.
TOOL_RANK: dict[str, int] = {
    # Slow / variable cost
    "pytest": 100, "tox": 100, "unittest": 95,
    "make": 90, "cmake": 90, "ninja": 90,
    "pip": 85, "conda": 85,
    "python": 60, "python3": 60,  # script execution; can be fast or slow
    "node": 60, "npm": 70, "yarn": 70,
    "docker": 80,

    # Moderate cost
    "git": 50,
    "find": 45, "fd": 45, "locate": 40,
    "grep": 35, "rg": 35, "ag": 35, "ack": 35,
    "diff": 35, "patch": 35,
    "tar": 50, "zip": 50, "unzip": 50, "gzip": 40,

    # Fast text manipulation
    "sed": 25, "awk": 25, "tr": 20, "cut": 20, "sort": 25, "uniq": 20,
    "head": 15, "tail": 15, "wc": 15, "nl": 15,
    "cat": 15, "less": 15, "ls": 15, "tree": 20,

    # Trivial
    "echo": 5, "printf": 5,
    "cd": 1, "pwd": 1, "true": 1, "false": 1, "test": 5,
    "mkdir": 10, "rm": 10, "mv": 10, "cp": 10, "ln": 10, "touch": 10,
    "env": 5, "export": 1,

    # Special — set by detection logic, not by token match
    "heredoc_write": 30,
}

# Compound separators we recognize. We split on these (outside quotes,
# approximately) and classify each part.
_COMPOUND_RE = re.compile(r"(?:&&|\|\||;|(?<!\|)\|(?!\|))")


def _strip_quoted(s: str) -> str:
    """Crude: blank out quoted regions so they don't trip the splitter.

    Replaces content inside matching single or double quotes with spaces,
    preserving length so positions stay valid. Doesn't handle nested
    quoting or escapes perfectly — good enough for bash from an LLM.
    """
    out = []
    in_q = None  # None, "'", or '"'
    for ch in s:
        if in_q:
            if ch == in_q:
                in_q = None
                out.append(ch)
            else:
                out.append(" ")
        else:
            if ch in ('"', "'"):
                in_q = ch
                out.append(ch)
            else:
                out.append(ch)
    return "".join(out)


def _is_env_assignment(token: str) -> bool:
    """`FOO=bar` style env-var prefix.

    Heuristic: contains '=' before any '/', '-', or quote. Excludes
    `--key=val` flags and path-y things.
    """
    if "=" not in token:
        return False
    if token.startswith("-"):
        return False
    key = token.split("=", 1)[0]
    return key.isidentifier() or all(c.isalnum() or c == "_" for c in key)


def _first_verb(part: str) -> str | None:
    """Find the first non-prefix token in a single command segment.

    Skips leading env-var assignments. Returns the binary name (path-stripped,
    quote-stripped), or None for empty segments.
    """
    tokens = part.strip().split()
    if not tokens:
        return None
    i = 0
    while i < len(tokens) and _is_env_assignment(tokens[i]):
        i += 1
    if i >= len(tokens):
        return None
    first = tokens[i]
    # Strip path prefix
    first = first.rsplit("/", 1)[-1]
    # Strip quotes
    first = first.strip("'\"")
    # Strip parentheses from subshell starts
    first = first.lstrip("(")
    return first or None


def _detect_heredoc(command: str) -> bool:
    """Heredoc write: `cat <<EOF > file ... EOF` or similar.

    Strong signal: `<<` followed by a token and (later) the same token
    on its own line, with redirection somewhere. We use a lax check:
    presence of `<<` and either `>` or `>>` in the same command.
    """
    if "<<" not in command:
        return False
    # Redirect somewhere (writing to file) OR cat'ed into a file
    return (">" in command) or (">>" in command)


def classify_tool(command: str) -> str:
    """Return the dominant-cost tool category for a bash command.

    Always returns a non-empty string; "other" for unrecognized.
    """
    cmd = command.strip()
    if not cmd:
        return "empty"

    # Special case: heredoc-style file write
    if _detect_heredoc(cmd):
        return "heredoc_write"

    # Strip bash line continuations (backslash followed by newline).
    # Otherwise compound parts after `\<NL>` get a stray `\` as the first
    # token, which breaks _first_verb. Real-world commands from LLM agents
    # often use line continuations for readability.
    cmd_unfolded = cmd.replace("\\\n", " ")

    # Mask quoted regions so we don't split on && that's inside a string
    masked = _strip_quoted(cmd_unfolded)

    # Split into compound parts; classify each; take the highest-rank one
    parts = _COMPOUND_RE.split(masked)
    candidates: list[str] = []
    for part in parts:
        verb = _first_verb(part)
        if verb:
            # Normalize: strip trailing punctuation
            verb = verb.rstrip(":,;")
            candidates.append(verb)

    if not candidates:
        return "empty"



    # Score each candidate AND record the normalized base name we matched.
    # Returns (score, normalized_name).
    def score_and_normalize(v: str) -> tuple[int, str]:
        if v in TOOL_RANK:
            return TOOL_RANK[v], v
        # Fuzzy: variants like python3.11, pytest-xdist, grep-3 — collapse to base
        for k in TOOL_RANK:
            if v.startswith(k):
                return TOOL_RANK[k], k
        return 0, v

    # Pick the verb with the highest score, ties broken by first occurrence
    best_score, best = score_and_normalize(candidates[0])
    for v in candidates[1:]:
        s, normalized = score_and_normalize(v)
        if s > best_score:
            best = normalized
            best_score = s

    # Aliases — collapse semantically-equivalent tools
    aliases = {"python3": "python", "rg": "grep", "ag": "grep", "ack": "grep"}
    best = aliases.get(best, best)

    # Unknown verbs (rank 0) collapse to "other" so we don't pollute
    # the category space with one-off binary names. command_raw is
    # preserved in the JSONL for offline re-classification.
    return best if best_score > 0 else "other"
