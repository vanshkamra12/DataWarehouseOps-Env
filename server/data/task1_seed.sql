-- Task 1: Enterprise Data Cleaning
-- A customer_records table imported from a legacy CRM with classic dirty-data issues:
--   1) Inconsistent gender strings: 'm','Male','MALE','female','F','f','unknown','N/A'
--   2) Broken date formats: 'March 5 1990', '05-03-1990', '1990/03/05', proper ISO 8601
--   3) Missing/null values in required columns (country)
--   4) Duplicate email addresses

CREATE TABLE IF NOT EXISTS customer_records (
    id          INTEGER PRIMARY KEY,
    full_name   TEXT NOT NULL,
    email       TEXT,
    gender      TEXT,
    birth_date  TEXT,
    country     TEXT,
    signup_date TEXT
);

INSERT INTO customer_records VALUES
(1,  'Alice Johnson',   'alice@example.com',   'Female',   '1990-03-05',       'USA',       '2022-01-10'),
(2,  'Bob Smith',       'bob@example.com',     'm',        'March 5 1985',     'UK',        '2022-01-15'),
(3,  'Carol White',     'carol@example.com',   'FEMALE',   '05-07-1992',       'Canada',    '2022-02-01'),
(4,  'Dan Brown',       'dan@example.com',     'male',     '1988/11/23',       'Australia', '2022-02-14'),
(5,  'Eve Davis',       'eve@example.com',     'F',        'January 1, 1995',  'USA',       '2022-03-01'),
(6,  'Frank Miller',    'frank@example.com',   'M',        '1993-06-15',       NULL,        '2022-03-10'),
(7,  'Grace Lee',       'grace@example.com',   'female',   '1991-09-30',       'Germany',   '2022-04-05'),
(8,  'Henry Wilson',    'henry@example.com',   'unknown',  '1987-12-12',       'France',    '2022-04-20'),
(9,  'Iris Clark',      'iris@example.com',    'N/A',      '1996-07-08',       'India',     '2022-05-01'),
(10, 'Jack Taylor',     'jack@example.com',    'MALE',     '15-04-1984',       NULL,        '2022-05-15'),
(11, 'Karen Martinez',  'karen@example.com',   'f',        '1999-02-28',       'Brazil',    '2022-06-01'),
(12, 'Liam Anderson',   'alice@example.com',   'male',     '1997-10-10',       'Mexico',    '2022-06-20'),
(13, 'Mia Thomas',      'mia@example.com',     'Female',   'September 9 2000', 'Japan',     '2022-07-01'),
(14, 'Noah Jackson',    'noah@example.com',    'male',     '1994-03-21',       'USA',       '2022-07-15'),
(15, 'Olivia Harris',   'olivia@example.com',  'FEMALE',   '2001/01/01',       'UK',        '2022-08-01'),
(16, 'Paul Martin',     'paul@example.com',    'M',        '12-12-1989',       NULL,        '2022-08-10'),
(17, 'Quinn Garcia',    'quinn@example.com',   'unknown',  '1992-11-11',       'Spain',     '2022-09-01'),
(18, 'Rose Rodriguez',  'rose@example.com',    'f',        '1990-05-25',       'Italy',     '2022-09-15'),
(19, 'Sam Lewis',       'sam@example.com',     'MALE',     'July 4, 1986',     'Canada',    '2022-10-01'),
(20, 'Tina Walker',     null,                  'Female',   '1998-08-18',       'USA',       '2022-10-20');

-- Ground truth for Task 1 grader (not exposed to agent)
-- DIRTY rows: gender not in ('Male','Female'), date not ISO-8601, country IS NULL, duplicate email
-- Expected transformations:
--   gender -> normalize to 'Male' or 'Female' or 'Unknown'
--   birth_date -> ISO-8601 YYYY-MM-DD
--   country NULL -> 'Unknown'
