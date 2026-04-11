"""
Task 2 Grader — PII Privacy Shielding
Scores how well the agent masked PII in a VIEW named 'masked_customers'.
All checks are pure SQL → 100% deterministic.

Score breakdown:
  - View exists                   : gate  (0.0 if missing)
  - Email masking correct         : 0.25  (username hidden, domain kept)
  - Credit card masking correct   : 0.25  (last 4 visible, rest ****)
  - SSN masking correct           : 0.25  (last 2 digits visible)
  - Safe columns intact           : 0.15  (id, first_name, last_name, country, signup_year)
  - Row count matches source      : 0.10
"""

import re
import sqlite3
from typing import Tuple


EMAIL_MASKED_RE     = re.compile(r"^\*{3}@.+\..+$")          # ***@domain.com
CC_MASKED_RE        = re.compile(r"^\*{4}-\*{4}-\*{4}-\d{4}$") # ****-****-****-1234
SSN_MASKED_RE       = re.compile(r"^\*{5}-\*{2}-\d{2}$")     # *****-**-12

REQUIRED_VIEW       = "masked_customers"
SOURCE_TABLE        = "customers_pii"
EXPECTED_ROWS       = 500
SAFE_COLUMNS        = ["id", "first_name", "last_name", "country", "signup_year", "risk_tier"]


def grade(conn: sqlite3.Connection) -> Tuple[float, dict]:
    cursor = conn.cursor()
    breakdown: dict = {}

    # ── Gate: Does the view exist? ───────────────────────────────────
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='view' AND name=?;",
        (REQUIRED_VIEW,)
    )
    view_exists = cursor.fetchone() is not None
    breakdown["view_exists"] = view_exists

    if not view_exists:
        return 0.01, {"error": f"View '{REQUIRED_VIEW}' does not exist.", **breakdown}

    # Try to query the view (count first, then sample)
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {REQUIRED_VIEW};")
        total_count = cursor.fetchone()[0]
        cursor.execute(f"SELECT * FROM {REQUIRED_VIEW} LIMIT 100;")
        rows = cursor.fetchall()
        col_names = [desc[0].lower() for desc in cursor.description]
    except Exception as e:
        return 0.01, {"error": f"View query failed: {e}", **breakdown}

    breakdown["row_count"] = total_count
    breakdown["columns"]   = col_names

    if total_count == 0:
        return 0.01, {"error": "View exists but returned 0 rows.", **breakdown}

    row_dicts = [dict(zip(col_names, r)) for r in rows]

    # ── 1. Email masking (0.25) ──────────────────────────────────────
    email_col = next((c for c in col_names if "email" in c), None)
    if email_col:
        email_scores = [
            1.0 if (v and EMAIL_MASKED_RE.match(str(v))) else 0.0
            for r in row_dicts for k, v in r.items() if k == email_col
        ]
        email_score = sum(email_scores) / max(len(email_scores), 1)
    else:
        email_score = 0.0
    breakdown["email_score"] = round(email_score, 4)

    # ── 2. Credit card masking (0.25) ────────────────────────────────
    cc_col = next((c for c in col_names if "credit" in c or "card" in c), None)
    if cc_col:
        cc_scores = [
            1.0 if (v and CC_MASKED_RE.match(str(v))) else 0.0
            for r in row_dicts for k, v in r.items() if k == cc_col
        ]
        cc_score = sum(cc_scores) / max(len(cc_scores), 1)
    else:
        cc_score = 0.0
    breakdown["cc_score"] = round(cc_score, 4)

    # ── 3. SSN masking (0.25) ────────────────────────────────────────
    ssn_col = next((c for c in col_names if "ssn" in c), None)
    if ssn_col:
        ssn_scores = [
            1.0 if (v and SSN_MASKED_RE.match(str(v))) else 0.0
            for r in row_dicts for k, v in r.items() if k == ssn_col
        ]
        ssn_score = sum(ssn_scores) / max(len(ssn_scores), 1)
    else:
        ssn_score = 0.0
    breakdown["ssn_score"] = round(ssn_score, 4)

    # ── 4. Safe columns intact (0.15) ────────────────────────────────
    present_safe = [c for c in SAFE_COLUMNS if c in col_names]
    safe_score = len(present_safe) / len(SAFE_COLUMNS)
    breakdown["safe_columns_present"] = present_safe
    breakdown["safe_score"] = round(safe_score, 4)

    # ── 5. Row count matches source (0.10) ───────────────────────────
    row_count_score = (
        1.0 if total_count == EXPECTED_ROWS
        else max(0.0, 1.0 - abs(total_count - EXPECTED_ROWS) / EXPECTED_ROWS)
    )
    breakdown["row_count_score"] = round(row_count_score, 4)

    # ── Final weighted score ─────────────────────────────────────────
    final = (
        0.25 * email_score
      + 0.25 * cc_score
      + 0.25 * ssn_score
      + 0.15 * safe_score
      + 0.10 * row_count_score
    )
    final = round(min(0.99, max(0.01, final)), 4)
    breakdown["final_score"] = final
    return final, breakdown
