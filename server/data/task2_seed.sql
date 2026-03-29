-- Task 2: PII Privacy Shielding
-- A financial platform stored raw sensitive PII. Under the new GDPR policy,
-- data scientists must only access masked data.
-- The agent must create a VIEW called "masked_customers" that:
--   1) Hides full email -> shows only domain part: e.g. "***@example.com"
--   2) Censors credit card numbers -> shows only last 4 digits: "****-****-****-1234"
--   3) Masks SSN -> shows only last 2 digits: "*****-**-XX"
--   4) Keeps: id, first_name, last_name, country, signup_year intact for analytics
--   5) Row count of the view must EXACTLY match the source table

CREATE TABLE IF NOT EXISTS customers_pii (
    id              INTEGER PRIMARY KEY,
    first_name      TEXT NOT NULL,
    last_name       TEXT NOT NULL,
    email           TEXT NOT NULL,
    credit_card     TEXT NOT NULL,
    ssn             TEXT NOT NULL,
    country         TEXT NOT NULL,
    signup_year     INTEGER NOT NULL,
    account_balance REAL NOT NULL
);

INSERT INTO customers_pii VALUES
(1,  'Alice',   'Johnson',  'alice.johnson@gmail.com',    '4532-1234-5678-9012', '123-45-6789', 'USA',       2020, 15230.50),
(2,  'Bob',     'Smith',    'bob.smith@yahoo.com',        '5425-9876-5432-1011', '987-65-4321', 'UK',        2019, 8750.00),
(3,  'Carol',   'White',    'carol.white@outlook.com',    '4916-1111-2222-3333', '111-22-3333', 'Canada',    2021, 42100.75),
(4,  'Dan',     'Brown',    'dan.brown@company.org',      '3714-4965-9837-1012', '444-55-6666', 'Australia', 2022, 3200.00),
(5,  'Eve',     'Davis',    'eve.davis@hotmail.com',      '4532-9999-8888-7777', '777-88-9999', 'USA',       2020, 91000.00),
(6,  'Frank',   'Miller',   'frank.miller@work.io',       '5425-3333-4444-5555', '000-11-2222', 'Germany',   2018, 5500.25),
(7,  'Grace',   'Lee',      'grace.lee@personal.net',     '4916-6666-7777-8888', '333-44-5555', 'France',    2023, 22000.00),
(8,  'Henry',   'Wilson',   'henry.wilson@enterprise.co', '3714-1234-5678-9012', '666-77-8888', 'India',     2021, 780.50),
(9,  'Iris',    'Clark',    'iris.clark@mail.com',        '4532-2222-3333-4444', '999-00-1111', 'Brazil',    2022, 11500.00),
(10, 'Jack',    'Taylor',   'jack.taylor@domain.edu',     '5425-7777-8888-9999', '222-33-4444', 'Japan',     2019, 67000.00);

-- IMPORTANT: The agent must create a VIEW named exactly "masked_customers"
-- Grader checks:
--   1. VIEW exists
--   2. email column contains '***@' prefix (no real usernames)
--   3. credit_card column matches pattern '****-****-****-XXXX'
--   4. ssn column matches pattern '*****-**-XX' (only 2 digits visible)
--   5. id, first_name, last_name, country, signup_year are UNMASKED
--   6. Row count == 10
