"""
Bootstrap a realistic SQLite database for the SQL Debugger environment.
Schema: e-commerce orders domain (customers, orders, products, order_items, reviews).
"""
import sqlite3
from pathlib import Path


DDL = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id   INTEGER PRIMARY KEY,
    name          TEXT    NOT NULL,
    email         TEXT    UNIQUE NOT NULL,
    country       TEXT    NOT NULL,
    joined_date   TEXT    NOT NULL   -- ISO-8601
);

CREATE TABLE IF NOT EXISTS products (
    product_id    INTEGER PRIMARY KEY,
    name          TEXT    NOT NULL,
    category      TEXT    NOT NULL,
    price         REAL    NOT NULL,
    stock         INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS orders (
    order_id      INTEGER PRIMARY KEY,
    customer_id   INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date    TEXT    NOT NULL,
    status        TEXT    NOT NULL  -- 'pending','shipped','delivered','cancelled'
);

CREATE TABLE IF NOT EXISTS order_items (
    item_id       INTEGER PRIMARY KEY,
    order_id      INTEGER NOT NULL REFERENCES orders(order_id),
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    quantity      INTEGER NOT NULL,
    unit_price    REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS reviews (
    review_id     INTEGER PRIMARY KEY,
    customer_id   INTEGER NOT NULL REFERENCES customers(customer_id),
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    rating        INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
    review_date   TEXT    NOT NULL
);

-- Indexes (for efficiency task grading)
CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_orders_date     ON orders(order_date);
CREATE INDEX IF NOT EXISTS idx_items_order     ON order_items(order_id);
CREATE INDEX IF NOT EXISTS idx_items_product   ON order_items(product_id);
CREATE INDEX IF NOT EXISTS idx_reviews_product ON reviews(product_id);
"""

SEED_DATA = """
INSERT OR IGNORE INTO customers VALUES
  (1,'Alice Smith','alice@example.com','US','2022-01-15'),
  (2,'Bob Jones','bob@example.com','UK','2022-03-20'),
  (3,'Carol White','carol@example.com','US','2022-05-10'),
  (4,'Dave Brown','dave@example.com','CA','2021-11-01'),
  (5,'Eve Davis','eve@example.com','AU','2023-02-28'),
  (6,'Frank Lee','frank@example.com','US','2023-06-01'),
  (7,'Grace Kim','grace@example.com','KR','2021-07-14');

INSERT OR IGNORE INTO products VALUES
  (1,'Laptop Pro 15','Electronics',1299.99,50),
  (2,'Wireless Mouse','Electronics',29.99,200),
  (3,'USB-C Hub','Electronics',49.99,150),
  (4,'Python Cookbook','Books',39.99,80),
  (5,'Standing Desk','Furniture',599.99,20),
  (6,'Ergonomic Chair','Furniture',449.99,15),
  (7,'Noise Cancelling Headphones','Electronics',299.99,60),
  (8,'Mechanical Keyboard','Electronics',149.99,100),
  (9,'SQL Mastery Book','Books',34.99,90),
  (10,'Desk Lamp','Furniture',49.99,120);

INSERT OR IGNORE INTO orders VALUES
  (1,1,'2024-01-10','delivered'),
  (2,1,'2024-03-05','delivered'),
  (3,2,'2024-02-14','shipped'),
  (4,3,'2024-01-20','delivered'),
  (5,3,'2024-04-01','cancelled'),
  (6,4,'2024-03-18','delivered'),
  (7,5,'2024-04-22','pending'),
  (8,6,'2024-05-01','delivered'),
  (9,7,'2024-01-05','delivered'),
  (10,1,'2024-05-15','pending'),
  (11,2,'2024-05-20','shipped'),
  (12,4,'2024-06-01','delivered');

INSERT OR IGNORE INTO order_items VALUES
  (1,1,1,1,1299.99),(2,1,2,2,29.99),
  (3,2,7,1,299.99),(4,2,8,1,149.99),
  (5,3,3,3,49.99),(6,3,4,1,39.99),
  (7,4,5,1,599.99),(8,4,6,1,449.99),
  (9,5,2,1,29.99),
  (10,6,1,1,1299.99),(11,6,3,2,49.99),
  (12,7,9,2,34.99),(13,7,10,1,49.99),
  (14,8,2,3,29.99),(15,8,8,1,149.99),
  (16,9,4,1,39.99),(17,9,9,1,34.99),
  (18,10,7,1,299.99),
  (19,11,1,1,1299.99),(20,11,5,1,599.99),
  (21,12,6,1,449.99),(22,12,10,2,49.99);

INSERT OR IGNORE INTO reviews VALUES
  (1,1,1,5,'2024-01-20'),
  (2,1,2,4,'2024-01-20'),
  (3,2,3,3,'2024-02-20'),
  (4,3,5,5,'2024-02-01'),
  (5,3,6,4,'2024-02-01'),
  (6,4,1,5,'2024-03-25'),
  (7,5,9,4,'2024-04-30'),
  (8,6,2,2,'2024-05-10'),
  (9,7,4,5,'2024-01-12'),
  (10,1,7,4,'2024-03-12');
"""

SCHEMA_TEXT = """
-- customers(customer_id PK, name, email, country, joined_date)
-- products(product_id PK, name, category, price, stock)
-- orders(order_id PK, customer_id FK, order_date, status)
--   status ∈ {'pending','shipped','delivered','cancelled'}
-- order_items(item_id PK, order_id FK, product_id FK, quantity, unit_price)
-- reviews(review_id PK, customer_id FK, product_id FK, rating 1-5, review_date)
"""


def get_db(path: str = ":memory:") -> sqlite3.Connection:
    """Return a seeded SQLite connection."""
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(DDL)
    conn.executescript(SEED_DATA)
    conn.commit()
    return conn
