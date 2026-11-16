"""The classifier against example commands.

Add a case here whenever a command is found to be classified wrongly.
"""

from __future__ import annotations

import pytest

from tooltrace.classifier import classify_tool

# (command, expected tool type)
CASES: list[tuple[str, str]] = [
    # --- Simple cases ---
    ("grep -r 'pattern' .", "grep"),
    ("sed -i 's/foo/bar/' file.py", "sed"),
    ("cat file.py", "cat"),
    ("ls /testbed", "ls"),
    ("pytest tests/test_foo.py", "pytest"),
    ("python script.py", "python"),
    ("python3 -c 'print(1)'", "python"),  # aliased to python
    ("git diff", "git"),

    # --- Compound: dominant-cost wins ---
    ("cd /testbed && pytest -x", "pytest"),
    ("cd /testbed && pip install -e .", "pip"),
    ("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached", "git"),
    ("grep -l foo . | xargs sed -i 's/foo/bar/'", "grep"),    # grep+sed equal-ish, grep wins by ordering
    ("find . -name '*.py' | head -20", "find"),
    ("ls -la && pwd", "ls"),

    # --- Env-var prefixes ---
    ("PYTHONPATH=/foo grep pattern file", "grep"),
    ("CUDA_VISIBLE_DEVICES=0,1 python train.py", "python"),
    ("FOO=bar BAZ=qux pytest", "pytest"),

    # --- Path-prefixed binaries ---
    ("/usr/bin/grep pattern file", "grep"),
    ("./node_modules/.bin/jest", "other"),    # jest not in our table -> other

    # --- Heredoc writes ---
    ("cat <<EOF > new_file.py\nimport foo\nEOF", "heredoc_write"),
    ("cat << 'EOF' >> append_to.txt\nstuff\nEOF", "heredoc_write"),

    # --- Edge cases ---
    ("", "empty"),
    ("   ", "empty"),
    ("# this is a comment", "other"),         # '#' isn't in our table; "other" is fine
    ("true", "true"),
    ("echo hi", "echo"),

    # --- Tricky: && inside quoted string shouldn't split ---
    ("echo 'foo && bar' && pytest", "pytest"),

    # --- Versioned binaries ---
    ("python3.11 -m pip install foo", "python"),
    ("pytest-xdist tests/", "pytest"),

    # --- Line continuations (real-world LLM-generated commands) ---
    ("cd docs && \\\nsed -i 's/foo/bar/' file && \\\nmake html", "make"),
    ("cd src && \\\npython setup.py build", "python"),
    ("cd /testbed && \\\npytest tests/test_foo.py -v", "pytest"),
]


@pytest.mark.parametrize("command, expected", CASES)
def test_classify_tool(command, expected):
    assert classify_tool(command) == expected
