-- Task 3: Query Optimization & Indexing
-- A financial reporting system runs a critical aggregation query across
-- a large sales_transactions table. It is doing a FULL TABLE SCAN,
-- making it extremely slow at scale.
--
-- The agent is given the slow query and must:
--   1) Run EXPLAIN QUERY PLAN to diagnose why it's slow
--   2) CREATE the correct index(es) on the right columns
--   3) Optionally: rewrite the query to be more efficient
--   4) Re-run EXPLAIN to verify a SCAN has become a SEARCH (Index Scan)
--
-- The grader checks: does EXPLAIN on the final query show "SEARCH" not "SCAN"?

CREATE TABLE IF NOT EXISTS sales_transactions (
    id              INTEGER PRIMARY KEY,
    order_id        TEXT NOT NULL,
    customer_id     INTEGER NOT NULL,
    product_id      INTEGER NOT NULL,
    region          TEXT NOT NULL,
    sale_date       TEXT NOT NULL,
    quantity        INTEGER NOT NULL,
    unit_price      REAL NOT NULL,
    discount        REAL NOT NULL DEFAULT 0.0,
    total_amount    REAL NOT NULL,
    status          TEXT NOT NULL  -- 'completed','pending','refunded'
);

CREATE TABLE IF NOT EXISTS products (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    category    TEXT NOT NULL,
    supplier_id INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS regions (
    id   INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    zone TEXT NOT NULL
);

INSERT INTO products VALUES
(1, 'Laptop Pro',    'Electronics',  101),
(2, 'USB Hub',       'Accessories',  102),
(3, 'Desk Chair',    'Furniture',    103),
(4, 'Monitor 27"',   'Electronics',  101),
(5, 'Keyboard MX',   'Accessories',  102),
(6, 'Standing Desk', 'Furniture',    103),
(7, 'Webcam HD',     'Electronics',  104),
(8, 'Headphones',    'Electronics',  104),
(9, 'Mouse Pad',     'Accessories',  102),
(10,'Filing Cabinet','Furniture',    103);

INSERT INTO regions VALUES
(1, 'North America', 'West'),
(2, 'Europe',        'East'),
(3, 'Asia Pacific',  'East'),
(4, 'Latin America', 'West');

-- Generate 2000 synthetic transactions for a realistic slow scan scenario
WITH RECURSIVE cnt(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM cnt WHERE x<2000)
INSERT INTO sales_transactions
SELECT
    x,
    'ORD-' || printf('%06d', x),
    (x % 500) + 1,
    (x % 10) + 1,
    CASE (x % 4) WHEN 0 THEN 'North America' WHEN 1 THEN 'Europe' WHEN 2 THEN 'Asia Pacific' ELSE 'Latin America' END,
    date('2023-01-01', '+' || (x % 365) || ' days'),
    (x % 10) + 1,
    round(50.0 + (x % 200), 2),
    round((x % 20) * 0.01, 2),
    round((50.0 + (x % 200)) * ((1+(x%10)) - (x%20)*0.01), 2),
    CASE (x % 3) WHEN 0 THEN 'completed' WHEN 1 THEN 'pending' ELSE 'refunded' END
FROM cnt;

-- THE SLOW QUERY (given to the agent):
-- SELECT
--     p.category,
--     st.region,
--     SUM(st.total_amount) AS total_revenue,
--     COUNT(st.id)         AS order_count
-- FROM sales_transactions st
-- JOIN products p ON st.product_id = p.id
-- WHERE st.sale_date BETWEEN '2023-06-01' AND '2023-09-30'
--   AND st.status = 'completed'
-- GROUP BY p.category, st.region
-- ORDER BY total_revenue DESC;
--
-- SOLUTION: CREATE INDEX idx_sales_date_status ON sales_transactions(sale_date, status);
-- After indexing, EXPLAIN QUERY PLAN shows "SEARCH" instead of "SCAN"
