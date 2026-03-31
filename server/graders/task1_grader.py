"""
Task 1 Grader: Data Cleaning
Scores how well the agent cleaned the `employee_records` table.
All checks are pure SQL.

Score breakdown:
  - Gender normalization   : 0.30 (all must be 'Male', 'Female', or 'Unknown')
  - Birth date ISO-8601    : 0.25 (all birth_dates must match YYYY-MM-DD)
  - Hire date ISO-8601     : 0.25 (all hire_dates must match YYYY-MM-DD)
  - Null country fix       : 0.10 (NULL country → 'Unknown')
  - No data loss           : 0.10 (2,000 rows must still exist)
"""

import re
import sqlite3
from typing import Tuple

ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def grade(conn: sqlite3.Connection) -> Tuple[float, dict]:
    cursor = conn.cursor()
    breakdown = {}

    # ── Total row count ──────────────────────────────────────────────
    cursor.execute("SELECT COUNT(*) FROM employee_records;")
    total_rows = cursor.fetchone()[0]
    breakdown["total_rows"] = total_rows

    if total_rows == 0:
        return 0.0, {"error": "Table is empty — all data was deleted!", **breakdown}

    # ── 1. Gender normalization (0.30) ───────────────────────────────
    cursor.execute(
        "SELECT COUNT(*) FROM employee_records "
        "WHERE gender NOT IN ('Male','Female','Unknown') OR gender IS NULL;"
    )
    bad_gender = cursor.fetchone()[0]
    gender_score = max(0.0, 1.0 - bad_gender / total_rows)
    breakdown["bad_gender_rows"] = bad_gender
    breakdown["gender_score"]    = round(gender_score, 4)

    # ── 2. Birth date normalization (0.25) ───────────────────────────
    cursor.execute("SELECT birth_date FROM employee_records;")
    bdates = [row[0] for row in cursor.fetchall()]
    bad_bdates = [d for d in bdates if d is None or not ISO_DATE_RE.match(str(d))]
    bdate_score = max(0.0, 1.0 - len(bad_bdates) / total_rows)
    breakdown["bad_birth_date_rows"] = len(bad_bdates)
    breakdown["birth_date_score"]    = round(bdate_score, 4)

    # ── 3. Hire date normalization (0.25) ────────────────────────────
    cursor.execute("SELECT hire_date FROM employee_records;")
    hdates = [row[0] for row in cursor.fetchall()]
    bad_hdates = [d for d in hdates if d is None or not ISO_DATE_RE.match(str(d))]
    hdate_score = max(0.0, 1.0 - len(bad_hdates) / total_rows)
    breakdown["bad_hire_date_rows"] = len(bad_hdates)
    breakdown["hire_date_score"]    = round(hdate_score, 4)

    # ── 4. Null country fix (0.10) ───────────────────────────────────
    cursor.execute("SELECT COUNT(*) FROM employee_records WHERE country IS NULL;")
    null_country = cursor.fetchone()[0]
    country_score = 1.0 if null_country == 0 else max(0.0, 1.0 - null_country / total_rows)
    breakdown["null_country_rows"] = null_country
    breakdown["country_score"]     = round(country_score, 4)

    # ── 5. No data loss (0.10) ───────────────────────────────────────
    MIN_ROWS = 2000
    no_loss_score = 1.0 if total_rows >= MIN_ROWS else total_rows / MIN_ROWS
    breakdown["expected_rows"] = MIN_ROWS
    breakdown["no_loss_score"] = round(no_loss_score, 4)

    # ── Final weighted score ─────────────────────────────────────────
    final = (
        0.30 * gender_score
      + 0.25 * bdate_score
      + 0.25 * hdate_score
      + 0.10 * country_score
      + 0.10 * no_loss_score
    )
    final = round(min(1.0, max(0.0, final)), 4)
    breakdown["final_score"] = final
    return final, breakdown
