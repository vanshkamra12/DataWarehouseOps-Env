"""
Task 3 Grader — Query Optimization & Indexing
Scores how well the agent optimized the slow financial reporting query.
Uses SQLite's EXPLAIN QUERY PLAN output — 100% deterministic.

The target slow query does a FULL TABLE SCAN on sales_transactions.
The agent must CREATE an index so that EXPLAIN shows SEARCH (index scan)
instead of SCAN (full table scan).

Score breakdown:
  - Index created on sale_date         : 0.30  (correct column)
  - Index created on status            : 0.20  (correct second column)
  - EXPLAIN shows SEARCH not SCAN      : 0.35  (proven optimization)
  - Query still returns correct data   : 0.15  (correctness check)
"""

import sqlite3
from typing import Tuple

SLOW_QUERY = """
SELECT
    p.category,
    st.region_name,
    SUM(st.total_amount)  AS total_revenue,
    COUNT(st.id)          AS order_count,
    AVG(st.discount_pct)  AS avg_discount_pct
FROM sales_transactions st
JOIN products p ON st.product_id = p.id
WHERE st.sale_date >= '2023-01-01'
  AND st.sale_date <  '2024-01-01'
  AND st.status = 'completed'
GROUP BY p.category, st.region_name
ORDER BY total_revenue DESC;
""".strip()

EXPECTED_MIN_ROWS = 3  # Real Northwind data across real categories/countries


def grade(conn: sqlite3.Connection) -> Tuple[float, dict]:
    cursor = conn.cursor()
    breakdown: dict = {}

    # ── 1. Check indexes that exist on sales_transactions ────────────
    cursor.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name='sales_transactions';"
    )
    indexes = cursor.fetchall()
    index_sqls = [row[1].lower() if row[1] else "" for row in indexes]
    index_names = [row[0] for row in indexes]
    breakdown["indexes_found"] = index_names

    has_sale_date_idx = any("sale_date" in sql for sql in index_sqls)
    has_status_idx    = any("status" in sql for sql in index_sqls)

    date_score   = 1.0 if has_sale_date_idx else 0.0
    status_score = 1.0 if has_status_idx    else 0.0

    breakdown["has_sale_date_index"] = has_sale_date_idx
    breakdown["has_status_index"]    = has_status_idx
    breakdown["date_score"]          = date_score
    breakdown["status_score"]        = status_score

    # ── 2. EXPLAIN QUERY PLAN — does it use a SEARCH? ────────────────
    try:
        cursor.execute(f"EXPLAIN QUERY PLAN {SLOW_QUERY}")
        plan_rows = cursor.fetchall()
        # Convert each row to a plain string (works with both sqlite3.Row and tuples)
        plan_text = " ".join(
            " ".join(str(cell) for cell in row) for row in plan_rows
        ).upper()
        uses_index = "SEARCH" in plan_text and "SCAN SALES_TRANSACTIONS" not in plan_text
    except Exception as e:
        plan_text  = str(e)
        uses_index = False

    explain_score = 1.0 if uses_index else 0.0
    breakdown["explain_plan"]   = plan_text[:500]  # Truncate for safety
    breakdown["uses_index"]     = uses_index
    breakdown["explain_score"]  = explain_score

    # ── 3. Query correctness — does it return plausible results? ─────
    try:
        cursor.execute(SLOW_QUERY)
        result_rows = cursor.fetchall()
        correct_score = 1.0 if len(result_rows) >= EXPECTED_MIN_ROWS else 0.0
        breakdown["result_row_count"] = len(result_rows)
    except Exception as e:
        correct_score = 0.0
        breakdown["query_error"] = str(e)

    breakdown["correct_score"] = correct_score

    # ── Final weighted score ─────────────────────────────────────────
    final = (
        0.30 * date_score
      + 0.20 * status_score
      + 0.35 * explain_score
      + 0.15 * correct_score
    )
    final = round(min(1.0, max(0.0, final)), 4)
    breakdown["final_score"] = final
    return final, breakdown
