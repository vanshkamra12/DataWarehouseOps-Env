"""
Real-World Data Generator — DataWarehouseOps-Env
=================================================
Uses REAL public databases (not synthetic) as the foundation:

  Task 1 — Based on the Northwind database (real Microsoft trading company data)
            Employees table extended to 2,000 rows via Faker, with real dirty-data
            patterns introduced programmatically.

  Task 2 — Based on the Chinook database (real digital music store — used by
            Apple, Amazon, etc. for SQL training). Real customer emails, phones,
            addresses from 20+ countries. Synthetic SSN + credit card added.

  Task 3 — The Northwind 'Order Details' table has 609,283 REAL transaction rows.
            We load a 100k-row subset with Products and Categories for the
            optimization task. This is the same data used in SQL Server benchmarks.

Both databases are recognized industry standards:
  - Northwind: https://github.com/jpwhite3/northwind-SQLite3
  - Chinook:   https://github.com/lerocha/chinook-database
"""

from __future__ import annotations

import random
import re
import sqlite3
import string
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent

try:
    from faker import Faker
    _fake = Faker()
    _FAKER_AVAILABLE = True
except ImportError:
    _FAKER_AVAILABLE = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rnd(seed: int) -> random.Random:
    r = random.Random(seed)
    if _FAKER_AVAILABLE:
        Faker.seed(seed)
    return r


def _dirty_gender(r: random.Random, correct: str) -> str:
    variants = {
        "Male":   ["m", "male", "MALE", "M", "Man", "man", "MAN", "Male"],
        "Female": ["f", "female", "FEMALE", "F", "Woman", "woman", "Female"],
    }
    if r.random() < 0.48 and correct in variants:
        return r.choice(variants[correct])
    if r.random() < 0.07:
        return r.choice(["N/A", "unknown", "not specified", "Unknown", "Other"])
    return correct


def _dirty_date(r: random.Random, iso_str: str) -> str:
    """Corrupt an ISO date into various messy formats."""
    try:
        dt = datetime.strptime(iso_str[:10], "%Y-%m-%d")
    except Exception:
        return iso_str

    # On macOS strftime doesn't support %-d; use a workaround
    day_no_pad = str(dt.day)
    month_name = dt.strftime("%B")
    year = dt.strftime("%Y")

    fmt_pool = [
        dt.strftime("%Y-%m-%d"),              # Correct ISO
        dt.strftime("%d-%m-%Y"),               # EU style 15-04-1984
        dt.strftime("%Y/%m/%d"),               # Slash 1990/07/04
        f"{month_name} {day_no_pad} {year}",   # "March 5 1985"
        f"{month_name} {day_no_pad}, {year}",  # "March 5, 1985"
        f"{dt.month}/{dt.day}/{year}",          # US style 3/5/1985
        f"{day_no_pad} {month_name} {year}",   # "5 March 1985"
    ]
    weights = [0.48, 0.12, 0.10, 0.10, 0.08, 0.07, 0.05]
    return r.choices(fmt_pool, weights=weights, k=1)[0]


def _fake_cc(r: random.Random) -> str:
    prefix = r.choice(["4532", "4916", "5425", "3714", "6011"])
    groups = [prefix] + ["".join([str(r.randint(0, 9)) for _ in range(4)]) for _ in range(3)]
    return "-".join(groups)


def _fake_ssn(r: random.Random) -> str:
    return f"{r.randint(100,999):03d}-{r.randint(10,99):02d}-{r.randint(1000,9999):04d}"


def _fake_email(r: random.Random, first: str, last: str) -> str:
    domains = ["gmail.com", "yahoo.com", "outlook.com", "company.io",
               "enterprise.co", "work.org", "personal.net", "fastmail.com",
               "proton.me", "icloud.com", "hotmail.com"]
    sep = r.choice([".", "_", ""])
    num = str(r.randint(1, 99)) if r.random() < 0.3 else ""
    # Sanitize names
    fn = re.sub(r"[^a-zA-Z]", "", first).lower() or "user"
    ln = re.sub(r"[^a-zA-Z]", "", last).lower()  or "x"
    return f"{fn}{sep}{ln}{num}@{r.choice(domains)}"


# ---------------------------------------------------------------------------
# Task 1 — Based on Northwind Employees + Faker expansion
# ---------------------------------------------------------------------------

def _load_northwind(target_conn: sqlite3.Connection) -> sqlite3.Connection:
    """Copy the Northwind DB into an in-memory connection."""
    nw_path = DATA_DIR / "northwind.db"
    src = sqlite3.connect(str(nw_path))
    src.backup(target_conn)
    src.close()
    return target_conn


def generate_task1(conn: sqlite3.Connection, seed: int = 42, n_rows: int = 2000) -> None:
    """
    Load real Northwind Employees as a base (9 real rows with real BirthDates,
    HireDates, names, cities, countries), then expand to n_rows using Faker.
    Introduce dirty-data patterns: broken dates, inconsistent genders, NULL countries.

    The resulting `employee_records` table mirrors what you'd find in a real
    HR data warehouse import from a legacy HRIS system.
    """
    r = _rnd(seed)

    # ── 1. Load real Northwind into a temp DB ───────────────────────────
    nw_path = DATA_DIR / "northwind.db"
    nw_conn = sqlite3.connect(str(nw_path))
    nw_cur  = nw_conn.cursor()
    nw_cur.execute("""
        SELECT EmployeeID, LastName, FirstName, Title, TitleOfCourtesy,
               BirthDate, HireDate, City, Region, Country
        FROM Employees;
    """)
    real_employees = nw_cur.fetchall()
    nw_conn.close()

    # ── 2. Create target table ──────────────────────────────────────────
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS employee_records (
            id             INTEGER PRIMARY KEY,
            full_name      TEXT NOT NULL,
            email          TEXT,
            title          TEXT,
            gender         TEXT,
            birth_date     TEXT,
            hire_date      TEXT,
            department     TEXT,
            city           TEXT,
            country        TEXT,
            salary         REAL,
            years_exp      INTEGER
        );
    """)

    departments = ["Engineering", "Sales", "Marketing", "Finance", "HR",
                   "Operations", "Product", "Legal", "Support", "Data"]
    titles = ["Ms.", "Mr.", "Dr.", "Mrs.", "Prof."]

    rows = []
    row_id = 1

    # ── 3. Insert real Northwind employees first (with dirty dates) ─────
    for emp in real_employees:
        eid, last, first, title, cort, bdstr, hdstr, city, region, country = emp
        gender_true = "Female" if cort in ("Ms.", "Mrs.", "Dr.") else "Male"
        dirty_gender = _dirty_gender(r, gender_true)
        dirty_bd = _dirty_date(r, str(bdstr)) if bdstr else None
        dirty_hd = _dirty_date(r, str(hdstr)) if hdstr else None
        rows.append((
            row_id, f"{first} {last}",
            _fake_email(r, first, last),
            title,
            dirty_gender,
            dirty_bd, dirty_hd,
            r.choice(departments),
            city,
            country,
            round(r.uniform(45000, 180000), 2),
            r.randint(1, 25),
        ))
        row_id += 1

    # ── 4. Expand to n_rows with Faker ──────────────────────────────────
    countries_pool = [
        "USA", "UK", "Canada", "Australia", "Germany", "France",
        "India", "Brazil", "Japan", "Mexico", "Spain", "Italy",
        "Netherlands", "Sweden", "South Korea", "Singapore", "UAE",
    ]

    while row_id <= n_rows:
        birth_year = r.randint(1960, 2000)
        birth_dt   = datetime(birth_year, r.randint(1, 12), r.randint(1, 28))
        hire_dt    = datetime(r.randint(2010, 2024), r.randint(1, 12), r.randint(1, 28))
        gender_t   = r.choice(["Male"] * 52 + ["Female"] * 48)
        country    = r.choice(countries_pool) if r.random() > 0.07 else None  # 7% NULL

        if _FAKER_AVAILABLE:
            fn, ln = _fake.first_name(), _fake.last_name()
            city = _fake.city()
        else:
            fn = r.choice(["Alex", "Blake", "Casey", "Dana", "Evan"])
            ln = r.choice(["Smith", "Johnson", "Williams", "Brown", "Jones"])
            city = r.choice(["London", "New York", "Berlin", "Tokyo", "Paris"])

        rows.append((
            row_id, f"{fn} {ln}",
            _fake_email(r, fn, ln),
            r.choice(titles),
            _dirty_gender(r, gender_t),
            _dirty_date(r, birth_dt.strftime("%Y-%m-%d")),
            _dirty_date(r, hire_dt.strftime("%Y-%m-%d")),
            r.choice(departments),
            city,
            country,
            round(r.uniform(30000, 220000), 2),
            r.randint(0, 30),
        ))
        row_id += 1

    conn.executemany(
        "INSERT INTO employee_records VALUES (?,?,?,?,?,?,?,?,?,?,?,?);",
        rows
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Task 2 — Based on Chinook Customer table (real digital music store PII)
# ---------------------------------------------------------------------------

def generate_task2(conn: sqlite3.Connection, seed: int = 42, n_rows: int = 500) -> None:
    """
    Load all 59 real Chinook Customer records (real names, emails, phones,
    addresses from 20+ countries), then expand to n_rows using Faker.
    Add synthetic SSN and credit_card columns so agents can practice PII masking.

    The Chinook database is used by Apple, Amazon, and dozens of universities
    worldwide as a standard SQL training dataset.
    """
    r = _rnd(seed)

    # ── 1. Load real Chinook customers ──────────────────────────────────
    ck_path = DATA_DIR / "chinook.db"
    ck_conn = sqlite3.connect(str(ck_path))
    ck_cur  = ck_conn.cursor()
    ck_cur.execute("""
        SELECT CustomerId, FirstName, LastName, Company, Address,
               City, State, Country, PostalCode, Phone, Fax, Email
        FROM Customer;
    """)
    real_customers = ck_cur.fetchall()
    ck_conn.close()

    # ── 2. Create target table ──────────────────────────────────────────
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS customers_pii (
            id              INTEGER PRIMARY KEY,
            first_name      TEXT NOT NULL,
            last_name       TEXT NOT NULL,
            company         TEXT,
            email           TEXT NOT NULL,
            phone           TEXT,
            address         TEXT,
            city            TEXT,
            country         TEXT NOT NULL,
            postal_code     TEXT,
            credit_card     TEXT NOT NULL,
            ssn             TEXT NOT NULL,
            signup_year     INTEGER NOT NULL,
            account_balance REAL NOT NULL,
            risk_tier       TEXT NOT NULL
        );
    """)

    tiers = ["Low", "Low", "Low", "Medium", "Medium", "High", "Critical"]

    rows = []
    row_id = 1

    # ── 3. Insert real Chinook customers (with synthetic PII added) ─────
    for cust in real_customers:
        cid, fn, ln, company, addr, city, state, country, postal, phone, fax, email = cust
        rows.append((
            row_id,
            fn, ln,
            company,
            email,                       # REAL email from Chinook
            phone or fax or "",          # REAL phone
            addr,
            city,
            country or "Unknown",
            postal or "",
            _fake_cc(r),                 # Synthetic — cannot use real card numbers
            _fake_ssn(r),                # Synthetic — GDPR
            r.randint(2015, 2024),
            round(r.uniform(-500, 200000), 2),
            r.choice(tiers),
        ))
        row_id += 1

    # ── 4. Expand to n_rows with Faker ──────────────────────────────────
    countries_pool = [
        "USA", "UK", "Canada", "Australia", "Germany", "France",
        "India", "Brazil", "Japan", "Mexico", "Spain", "Italy",
        "Netherlands", "Sweden", "Denmark", "Norway", "Portugal",
        "Argentina", "Chile", "Czech Republic", "Hungary", "Poland",
    ]

    while row_id <= n_rows:
        if _FAKER_AVAILABLE:
            fn = _fake.first_name()
            ln = _fake.last_name()
            city = _fake.city()
            phone = _fake.phone_number()[:20]
            addr = _fake.street_address()
            postal = _fake.postcode()
        else:
            fn = r.choice(["Anna","Ben","Clara","David","Emma"])
            ln = r.choice(["Smith","Brown","Jones","Davis","Wilson"])
            city, phone, addr, postal = "New York", "+1-555-0100", "123 Main St", "10001"

        country = r.choice(countries_pool)
        rows.append((
            row_id,
            fn, ln,
            None,
            _fake_email(r, fn, ln),
            phone,
            addr,
            city,
            country,
            postal,
            _fake_cc(r),
            _fake_ssn(r),
            r.randint(2015, 2024),
            round(r.uniform(-500, 200000), 2),
            r.choice(tiers),
        ))
        row_id += 1

    conn.executemany(
        "INSERT INTO customers_pii VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);",
        rows
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Task 3 — Based on Northwind Order Details (609,283 REAL rows)
# ---------------------------------------------------------------------------

def generate_task3(conn: sqlite3.Connection, seed: int = 42, n_rows: int = 100_000) -> None:
    """
    Load a subset of the real Northwind 'Order Details' + Products + Categories
    tables. The Northwind dataset was created by Microsoft and is used in SQL
    Server, Oracle, and PostgreSQL training worldwide.

    We sample n_rows from the 609,283 real transaction lines and join with
    real Products and Categories — real prices, real product names, real order IDs.

    The slow query the agent must optimize runs a multi-table join with date
    filtering — the exact pattern used in production BI/reporting systems.
    """
    r = _rnd(seed)

    # ── 1. Load Northwind into the target connection ─────────────────────
    nw_path = DATA_DIR / "northwind.db"
    src = sqlite3.connect(str(nw_path))
    src.backup(conn)    # Copy entire DB into target in-memory DB
    src.close()

    # ── 2. Remove giant tables we don't need (keep DB lean) ─────────────
    conn.executescript("""
        DROP TABLE IF EXISTS CustomerCustomerDemo;
        DROP TABLE IF EXISTS CustomerDemographics;
        DROP TABLE IF EXISTS EmployeeTerritories;
        DROP TABLE IF EXISTS Territories;
        DROP TABLE IF EXISTS Regions;
        DROP TABLE IF EXISTS Shippers;
        DROP TABLE IF EXISTS Suppliers;
        DROP TABLE IF EXISTS Employees;
        DROP TABLE IF EXISTS Customers;
    """)

    # ── 3. Sample n_rows from the real Order Details ─────────────────────
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM \"Order Details\";")
    total = cursor.fetchone()[0]

    # Create a clean transactions table from the real Northwind data
    conn.executescript(f"""
        CREATE TABLE IF NOT EXISTS sales_transactions AS
        SELECT
            od.rowid                               AS id,
            od.OrderID                             AS order_id,
            o.CustomerID                           AS customer_id,
            od.ProductID                           AS product_id,
            p.ProductName                          AS product_name,
            c.CategoryName                         AS category,
            o.ShipCountry                          AS region,
            o.OrderDate                            AS sale_date,
            od.Quantity                            AS quantity,
            od.UnitPrice                           AS unit_price,
            CAST(od.Discount * 100 AS INTEGER)     AS discount_pct,
            ROUND(od.UnitPrice * od.Quantity * (1 - od.Discount), 2) AS total_amount,
            CASE
                WHEN od.Discount = 0.0 THEN 'completed'
                WHEN od.Discount > 0.15 THEN 'promotional'
                ELSE 'completed'
            END                                    AS status,
            o.EmployeeID                           AS sales_rep_id
        FROM "Order Details" od
        JOIN Orders o   ON od.OrderID   = o.OrderID
        JOIN Products p ON od.ProductID = p.ProductID
        JOIN Categories c ON p.CategoryID = c.CategoryID
        ORDER BY RANDOM()
        LIMIT {min(n_rows, total)};
    """)

    # ── 4. Drop the now-redundant raw tables ─────────────────────────────
    conn.executescript("""
        DROP TABLE IF EXISTS "Order Details";
        DROP TABLE IF EXISTS Orders;
    """)

    conn.commit()


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def seed_database(conn: sqlite3.Connection, task_id: str, scenario_seed: int = 42) -> None:
    """
    Seed the given connection with data for the specified task.
    Uses REAL public databases (Northwind, Chinook) as the foundation.
    scenario_seed controls which Faker variant is used for expanded rows.
    """
    generators = {
        "task1_data_cleaning":      generate_task1,
        "task2_pii_masking":        generate_task2,
        "task3_query_optimization": generate_task3,
    }
    fn = generators.get(task_id)
    if fn:
        fn(conn, seed=scenario_seed)
    else:
        raise ValueError(f"Unknown task_id: {task_id}")
