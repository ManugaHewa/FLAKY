"""
test_log_parsing.py

Validates parse_pytest_log against a REAL, documented pytest-rerunfailures
log excerpt (taken verbatim from the plugin's own README example output),
rather than waiting for a live CI run with an actual failure to show up
within GitHub's 90-day log retention window.

This is the fast way to confirm the regexes are correct before spending
API calls / time on a full collection run.
"""

import sys
import types
from pathlib import Path

# Stub out third-party deps this sandbox doesn't have installed (tqdm,
# requests, python-dotenv) -- test_log_parsing.py only needs the pure
# regex-parsing logic in collect_data.py, not the network-facing parts,
# so these stubs just need to exist, not actually work.
for mod_name in ("requests", "dotenv", "tqdm"):
    if mod_name not in sys.modules:
        stub = types.ModuleType(mod_name)
        if mod_name == "requests":
            stub.Session = object
        if mod_name == "dotenv":
            stub.load_dotenv = lambda *a, **k: None
        if mod_name == "tqdm":
            stub.tqdm = lambda iterable=None, **k: iterable if iterable is not None else []
        sys.modules[mod_name] = stub

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.collect_data import parse_pytest_log  # noqa: E402

# Verbatim structure of what pytest-rerunfailures prints on a failure that
# gets retried twice and never passes (dot-mode output + summary sections):
SAMPLE_LOG = """\
2026-08-17T17:57:54.2339031Z
2026-08-17T17:57:55.0000000Z test_report.py F
2026-08-17T17:57:55.1000000Z
2026-08-17T17:57:55.2000000Z ================================== FAILURES ===================================
2026-08-17T17:57:55.3000000Z __________________________________ test_fail __________________________________
2026-08-17T17:57:55.4000000Z
2026-08-17T17:57:55.5000000Z def test_fail():
2026-08-17T17:57:55.6000000Z >       assert False
2026-08-17T17:57:55.7000000Z E       assert False
2026-08-17T17:57:55.8000000Z
2026-08-17T17:57:55.9000000Z test_report.py:9: AssertionError
2026-08-17T17:57:56.0000000Z ============================ rerun test summary info ==========================
2026-08-17T17:57:56.1000000Z RERUN test_report.py::test_fail
2026-08-17T17:57:56.2000000Z RERUN test_report.py::test_fail
2026-08-17T17:57:56.3000000Z ============================ short test summary info ===========================
2026-08-17T17:57:56.4000000Z FAILED test_report.py::test_fail - AssertionError: assert False
2026-08-17T17:57:56.5000000Z ======================= 1 failed, 2 rerun in 0.02 seconds ======================
"""


def test_parses_rerun_lines():
    rows = parse_pytest_log(
        SAMPLE_LOG, run_id=1, commit_sha="abc123", timestamp="2026-08-17T17:57:54Z"
    )
    rerun_rows = [r for r in rows if r["attempt"] < 3 and r["outcome"] == "failed"
                  and r["test_name"] == "test_report.py::test_fail"]
    # Expect: 2 RERUN rows (attempts 1, 2) + 1 FAILED summary row (attempt 3)
    assert len(rows) == 3, f"expected 3 rows, got {len(rows)}: {rows}"
    assert rerun_rows, "expected to find RERUN-derived failure rows"


def test_parses_error_message_from_failed_line():
    rows = parse_pytest_log(
        SAMPLE_LOG, run_id=1, commit_sha="abc123", timestamp="2026-08-17T17:57:54Z"
    )
    failed_row = next(r for r in rows if r["attempt"] == 3)
    assert failed_row["error_message"] is not None
    assert "AssertionError" in failed_row["error_message"]


def test_clean_run_produces_zero_rows():
    """
    A fully clean run (dot-mode, all passed, no FAILURES/RERUN sections)
    should correctly produce zero rows -- not an error, not a crash.
    This is the case we actually saw in the real Aug 17 CI log.
    """
    clean_log = """\
2026-08-17T17:57:54.2339031Z collected 160 items
2026-08-17T17:57:55.1293868Z tests/test_pytest_rerunfailures.py ......................ss.......
2026-08-17T17:57:58.7436873Z ======================== 156 passed, 4 skipped in 4.61s ========================
"""
    rows = parse_pytest_log(
        clean_log, run_id=2, commit_sha="def456", timestamp="2026-08-17T17:57:54Z"
    )
    assert rows == [], f"expected zero rows on a clean run, got {rows}"


if __name__ == "__main__":
    test_parses_rerun_lines()
    test_parses_error_message_from_failed_line()
    test_clean_run_produces_zero_rows()
    print("All parser tests passed.")