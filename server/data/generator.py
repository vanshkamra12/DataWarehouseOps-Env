"""
Data seed generator for DataWarehouseOps-Env.
Generates ALL data synthetically using Faker — no external .db files needed.
This ensures the evaluator can run without Git LFS or external database files.
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
        dt = datetime.strptime(iso_str, "%Y-%m-%d")
    except Exception:
        return iso_str
    fmts = [
        "%B %d %Y",      # March 05 1985
        "%d-%m-%Y",      # 05-03-1985
        "%Y/%m/%d",      # 1985/03/05
        "%-m/%-d/%Y",    # 3/5/1985
        "%d %b %Y",      # 05 Mar 1985
        "%m.%d.%Y",      # 03.05.1985
        "%b %-d, %Y",    # Mar 5, 1985
    ]
    try:
        return dt.strftime(r.choice(fmts))
    except ValueError:
        # %-m / %-d may fail on Windows; fallback
        return dt.strftime("%m/%d/%Y")


def _fake_cc(r: random.Random) -> str:
    prefix = r.choice(["4", "5", "37", "6011"])
    total  = 16 if prefix != "37" else 15
    rest   = total - len(prefix)
    return prefix + "".join(str(r.randint(0, 9)) for _ in range(rest))


def _fake_ssn(r: random.Random) -> str:
    return f"{r.randint(100,999)}-{r.randint(10,99)}-{r.randint(1000,9999)}"


DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com",
           "company.co", "example.org", "work.io", "mail.net"]


def _fake_email(r: random.Random, first: str, last: str) -> str:
    sep = r.choice([".", "_", ""])
    num = str(r.randint(1, 999)) if r.random() < 0.4 else ""
    fn = re.sub(r"[^a-zA-Z]", "", first).lower() or "user"
    ln = re.sub(r"[^a-zA-Z]", "", last).lower()  or "x"
    return f"{fn}{sep}{ln}{num}@{r.choice(DOMAINS)}"


# ---------------------------------------------------------------------------
# Task 1 — Fully synthetic dirty employee records
# ---------------------------------------------------------------------------

def generate_task1(conn: sqlite3.Connection, seed: int = 42, n_rows: int = 2000) -> None:
    """
    Generate synthetic employee records with targeted data quality issues.
    """
    r = _rnd(seed)

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
    countries_pool = [
        "USA", "UK", "Canada", "Australia", "Germany", "France",
        "India", "Brazil", "Japan", "Mexico", "Spain", "Italy",
        "Netherlands", "Sweden", "South Korea", "Singapore", "UAE",
    ]

    rows = []
    for row_id in range(1, n_rows + 1):
        birth_year = r.randint(1960, 2000)
        birth_dt   = datetime(birth_year, r.randint(1, 12), r.randint(1, 28))
        hire_dt    = datetime(r.randint(2010, 2024), r.randint(1, 12), r.randint(1, 28))
        gender_t   = r.choice(["Male"] * 52 + ["Female"] * 48)
        country    = r.choice(countries_pool) if r.random() > 0.07 else None  # 7% NULL

        if _FAKER_AVAILABLE:
            fn, ln = _fake.first_name(), _fake.last_name()
            city = _fake.city()
        else:
            fn = r.choice(["Alex", "Blake", "Casey", "Dana", "Evan",
                           "Fiona", "Grace", "Henry", "Iris", "Jack"])
            ln = r.choice(["Smith", "Johnson", "Williams", "Brown", "Jones",
                           "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"])
            city = r.choice(["London", "New York", "Berlin", "Tokyo", "Paris",
                             "Mumbai", "Sydney", "Toronto", "Dubai", "Seoul"])

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

    conn.executemany(
        "INSERT INTO employee_records VALUES (?,?,?,?,?,?,?,?,?,?,?,?);",
        rows
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Task 2 — Fully synthetic PII customer records
# ---------------------------------------------------------------------------

def generate_task2(conn: sqlite3.Connection, seed: int = 42, n_rows: int = 500) -> None:
    """
    Generate synthetic customer records with PII columns for masking task.
    """
    r = _rnd(seed)

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
    countries_pool = [
        "USA", "UK", "Canada", "Australia", "Germany", "France",
        "India", "Brazil", "Japan", "Mexico", "Spain", "Italy",
        "Netherlands", "Sweden", "Denmark", "Norway", "Portugal",
        "Argentina", "Chile", "Czech Republic", "Hungary", "Poland",
    ]

    rows = []
    for row_id in range(1, n_rows + 1):
        if _FAKER_AVAILABLE:
            fn = _fake.first_name()
            ln = _fake.last_name()
            city = _fake.city()
            phone = _fake.phone_number()[:20]
            addr = _fake.street_address()
            postal = _fake.postcode()
            company = _fake.company() if r.random() > 0.3 else None
        else:
            fn = r.choice(["Anna","Ben","Clara","David","Emma",
                           "Frank","Gina","Hugo","Ivy","Jake"])
            ln = r.choice(["Smith","Brown","Jones","Davis","Wilson",
                           "Taylor","Anderson","Thomas","Jackson","White"])
            city, phone, addr, postal = "New York", "+1-555-0100", "123 Main St", "10001"
            company = None

        country = r.choice(countries_pool)
        rows.append((
            row_id,
            fn, ln,
            company,
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

    conn.executemany(
        "INSERT INTO customers_pii VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);",
        rows
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Task 3 — Fully synthetic sales_transactions for index optimization
# ---------------------------------------------------------------------------

def generate_task3(conn: sqlite3.Connection, seed: int = 42, n_rows: int = 100_000) -> None:
    """
    Generate synthetic sales transactions and products tables for query optimization.
    """
    r = _rnd(seed)

    # Create products table
    categories = ["Electronics", "Clothing", "Food & Beverage", "Home & Garden",
                  "Sports", "Books", "Automotive", "Health & Beauty"]
    products = []
    for pid in range(1, 201):  # 200 products
        cat = r.choice(categories)
        if _FAKER_AVAILABLE:
            name = f"{_fake.word().capitalize()} {_fake.word().capitalize()}"
        else:
            name = f"Product-{pid}"
        price = round(r.uniform(5, 500), 2)
        products.append((pid, name, cat, price))

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS products (
            id         INTEGER PRIMARY KEY,
            name       TEXT NOT NULL,
            category   TEXT NOT NULL,
            base_price REAL NOT NULL
        );
    """)
    conn.executemany("INSERT INTO products VALUES (?,?,?,?);", products)

    # Create sales_transactions table
    regions = ["North America", "Europe", "Asia Pacific", "Latin America",
               "Middle East", "Africa", "Oceania"]
    statuses = ["completed", "completed", "completed", "completed",
                "pending", "cancelled", "refunded", "promotional"]

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sales_transactions (
            id            INTEGER PRIMARY KEY,
            order_id      INTEGER NOT NULL,
            customer_id   INTEGER NOT NULL,
            product_id    INTEGER NOT NULL,
            region_name   TEXT NOT NULL,
            sale_date     TEXT NOT NULL,
            quantity      INTEGER NOT NULL,
            unit_price    REAL NOT NULL,
            discount_pct  INTEGER NOT NULL,
            total_amount  REAL NOT NULL,
            status        TEXT NOT NULL,
            sales_rep_id  INTEGER NOT NULL
        );
    """)

    rows = []
    for tid in range(1, n_rows + 1):
        pid = r.randint(1, 200)
        base_price = products[pid - 1][3]
        qty = r.randint(1, 50)
        disc = r.choice([0, 0, 0, 5, 10, 15, 20, 25])
        total = round(base_price * qty * (1 - disc / 100), 2)
        # Dates spread across 2020-2024
        year = r.randint(2020, 2024)
        month = r.randint(1, 12)
        day = r.randint(1, 28)
        sale_date = f"{year:04d}-{month:02d}-{day:02d}"

        rows.append((
            tid,
            r.randint(10000, 99999),  # order_id
            r.randint(1, 5000),       # customer_id
            pid,
            r.choice(regions),
            sale_date,
            qty,
            base_price,
            disc,
            total,
            r.choice(statuses),
            r.randint(1, 50),  # sales_rep_id
        ))

    conn.executemany(
        "INSERT INTO sales_transactions VALUES (?,?,?,?,?,?,?,?,?,?,?,?);",
        rows
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def seed_database(conn: sqlite3.Connection, task_id: str, scenario_seed: int = 42) -> None:
    """Seed the given connection with data for the specified task."""
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
