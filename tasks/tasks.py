"""
Task definitions for the SQL Debugger environment.

Each task has:
  - id, name, difficulty
  - business_requirement  (natural language)
  - buggy_query           (what the agent starts with)
  - max_steps
  - grade(rows, error, query) → (score: float, feedback: str)
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_rows(rows: List[Dict]) -> List[tuple]:
    """Sort and flatten rows for comparison."""
    return sorted(tuple(sorted(r.items())) for r in rows)


def _col(rows: List[Dict], col: str) -> List[Any]:
    return [r.get(col) for r in rows]


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------

@dataclass
class Task:
    id: str
    name: str
    difficulty: str           # easy | medium | hard
    business_requirement: str
    buggy_query: str
    max_steps: int
    grader: Callable[[List[Dict], Optional[str], str], Tuple[float, str]]
    expected_hint: str = ""   # shown in README, not to agent


# ---------------------------------------------------------------------------
# TASK 1 – Easy: Syntax & Basic Semantic Fix
# ---------------------------------------------------------------------------
# Requirement: List total revenue per product category for DELIVERED orders,
#              ordered by revenue descending.
# Bugs planted:
#   1. Uses `SUM(quantity)` instead of `SUM(quantity * unit_price)`
#   2. Missing WHERE clause for status filter
#   3. ORDER BY direction missing (defaults asc, should be desc)

TASK_EASY_BUGGY = """
SELECT p.category,
       SUM(oi.quantity) AS total_revenue
FROM   orders o
JOIN   order_items oi ON oi.order_id  = o.order_id
JOIN   products    p  ON p.product_id = oi.product_id
GROUP  BY p.category
ORDER  BY total_revenue;
"""

EXPECTED_EASY = [
    {"category": "Electronics", "total_revenue": round(1299.99+29.99+299.99+149.99+49.99*3+29.99*3+149.99+299.99+1299.99+49.99*2+29.99*3+149.99, 2)},
    {"category": "Books",       "total_revenue": round(39.99+34.99+39.99+34.99, 2)},
    {"category": "Furniture",   "total_revenue": round(599.99+449.99+1299.99+49.99, 2)},
]

# Precomputed correct answer (delivered orders only):
# We'll verify dynamically in the grader against the live DB.
_EASY_CORRECT_CATEGORIES = {"Electronics", "Books", "Furniture"}

def _grade_easy(rows: List[Dict], error: Optional[str], query: str) -> Tuple[float, str]:
    score = 0.0
    parts = []

    if error:
        parts.append(f"Query failed: {error}")
        # partial: syntax nearly right?
        if "SyntaxError" not in error and "no such column" not in error:
            score += 0.05
        return round(max(0.001, score), 3), " | ".join(parts)

    # Check: has rows
    if not rows:
        return 0.05, "Query ran but returned no rows."

    # Check: correct columns present
    cols = set(rows[0].keys())
    if "category" not in cols:
        parts.append("Missing 'category' column")
    else:
        score += 0.1

    if "total_revenue" not in cols and not any(
        c.lower() in ("revenue","total","sum") for c in cols
    ):
        parts.append("Missing revenue column")
    else:
        score += 0.1

    # Check: correct number of categories (3)
    categories = {r.get("category") for r in rows}
    if categories == _EASY_CORRECT_CATEGORIES:
        score += 0.2
        parts.append("✓ All 3 categories present")
    else:
        parts.append(f"Expected categories {_EASY_CORRECT_CATEGORIES}, got {categories}")

    # Check: revenue uses price (not just quantity)
    rev_col = next((c for c in cols if "rev" in c.lower() or "total" in c.lower() or "sum" in c.lower()), None)
    if rev_col:
        rev_vals = [r[rev_col] for r in rows if r.get("category") == "Electronics"]
        if rev_vals and max(rev_vals) > 100:
            score += 0.2
            parts.append("✓ Revenue values look like prices (not just counts)")
        else:
            parts.append("Revenue values seem too small — are you summing unit_price×quantity?")

    # Check: filters to delivered only (no cancelled orders)
    # Delivered-only means order 5 (cancelled) items excluded
    # Cancelled order 5 had 1 item: product_id=2, qty=1 → shouldn't appear in totals
    query_lower = query.lower()
    if "delivered" in query_lower or ("status" in query_lower and "cancel" not in query_lower):
        score += 0.2
        parts.append("✓ Filters by delivered status")
    else:
        parts.append("Missing WHERE status='delivered' filter")

    # Check: ordered descending
    if len(rows) >= 2:
        revs = [r.get(rev_col, 0) for r in rows if rev_col]
        if revs == sorted(revs, reverse=True):
            score += 0.2
            parts.append("✓ Results ordered descending by revenue")
        else:
            parts.append("Results should be ordered by revenue DESC")

    # Clamp score strictly between 0 and 1 as per hackathon requirement
    final_score = max(0.001, min(score, 0.999))
    return round(final_score, 3), " | ".join(parts) if parts else "Keep going!"


TASK_EASY = Task(
    id="task_easy",
    name="SQL Syntax Fix",
    difficulty="easy",
    business_requirement=(
        "Show total revenue (sum of quantity × unit_price) per product CATEGORY "
        "for DELIVERED orders only, ordered from highest revenue to lowest. "
        "Columns: category, total_revenue."
    ),
    buggy_query=TASK_EASY_BUGGY.strip(),
    max_steps=10,
    grader=_grade_easy,
    expected_hint="Fix: SUM(oi.quantity*oi.unit_price), add WHERE o.status='delivered', add DESC",
)


# ---------------------------------------------------------------------------
# TASK 2 – Medium: Logic Repair (wrong JOIN + aggregation bug)
# ---------------------------------------------------------------------------
# Requirement: For each customer who has placed at least 2 DELIVERED orders,
#              show their name, number of delivered orders, and total amount spent.
#              Order by total_spent DESC.
# Bugs planted:
#   1. LEFT JOIN reviews instead of LEFT JOIN orders (brings in wrong table)
#   2. COUNT(*) counts all joined rows not distinct orders
#   3. HAVING threshold wrong (>= 1 instead of >= 2)

TASK_MEDIUM_BUGGY = """
SELECT c.name,
       COUNT(*) AS delivered_orders,
       SUM(oi.quantity * oi.unit_price) AS total_spent
FROM   customers c
JOIN   orders o   ON o.customer_id = c.customer_id
JOIN   order_items oi ON oi.order_id = o.order_id
LEFT JOIN reviews r ON r.customer_id = c.customer_id
WHERE  o.status = 'delivered'
GROUP  BY c.customer_id
HAVING COUNT(*) >= 1
ORDER  BY total_spent DESC;
"""

def _grade_medium(rows: List[Dict], error: Optional[str], query: str) -> Tuple[float, str]:
    score = 0.0
    parts = []

    if error:
        parts.append(f"Query error: {error}")
        score += 0.02
        return round(max(0.001, min(score, 0.999)), 3), " | ".join(parts)

    if not rows:
        return 0.05, "No rows returned."

    cols = set(rows[0].keys())

    # Check: name column
    if any("name" in c.lower() for c in cols):
        score += 0.1
        parts.append("✓ Has name column")
    else:
        parts.append("Missing customer name column")

    # Correct answer: customers with ≥2 delivered orders
    # Orders: 1→delivered, 2→delivered, 4→delivered (Alice=2), 6→delivered (Dave=1 — excluded)
    # Alice: orders 1,2 = delivered → 2 orders ✓
    # Dave:  orders 6,12 = delivered → 2 orders ✓
    # Carol: order 4 = delivered (order 5 cancelled) → 1 order ✗
    expected_names = {"Alice Smith", "Dave Brown"}

    returned_names = set()
    name_col = next((c for c in cols if "name" in c.lower()), None)
    if name_col:
        returned_names = {r[name_col] for r in rows}

    if returned_names == expected_names:
        score += 0.35
        parts.append(f"✓ Correct customers: {expected_names}")
    elif expected_names.issubset(returned_names):
        score += 0.15
        parts.append(f"Includes correct customers but has extras: {returned_names - expected_names}")
    elif returned_names & expected_names:
        score += 0.08
        parts.append(f"Partially correct customers: {returned_names & expected_names}")
    else:
        parts.append(f"Wrong customers returned. Expected {expected_names}")

    # Check: HAVING >= 2
    query_lower = query.lower()
    if re.search(r"having\s+count.*>=\s*2", query_lower):
        score += 0.2
        parts.append("✓ HAVING COUNT >= 2")
    elif "having" in query_lower:
        parts.append("HAVING clause present but threshold may be wrong (need >= 2)")
        score += 0.05

    # Check: no spurious LEFT JOIN reviews (which inflates counts)
    if "reviews" not in query_lower:
        score += 0.15
        parts.append("✓ No spurious JOIN to reviews table")
    else:
        parts.append("Joining reviews table inflates row counts — remove it")

    # Check: count distinct orders not rows
    if "count(distinct" in query_lower and "order_id" in query_lower:
        score += 0.1
        parts.append("✓ Uses COUNT(DISTINCT order_id)")
    elif returned_names == expected_names:
        score += 0.05  # correct result even with COUNT(*) if reviews join removed
        parts.append("Result correct — COUNT(*) works if reviews join removed")

    # Check: ordered desc
    spent_col = next((c for c in cols if "spent" in c.lower() or "total" in c.lower()), None)
    if spent_col and len(rows) >= 2:
        vals = [r[spent_col] for r in rows]
        if vals == sorted(vals, reverse=True):
            score += 0.1
            parts.append("✓ Ordered by total_spent DESC")

    # Clamp score strictly between 0 and 1 as per hackathon requirement
    final_score = max(0.001, min(score, 0.999))
    return round(final_score, 3), " | ".join(parts)


TASK_MEDIUM = Task(
    id="task_medium",
    name="SQL Logic Repair",
    difficulty="medium",
    business_requirement=(
        "Find customers who have placed at least 2 delivered orders. "
        "Show: name, delivered_orders (count of delivered orders), total_spent (sum of quantity×unit_price). "
        "Order by total_spent descending. "
        "Note: only count delivered orders; ignore cancelled/pending/shipped."
    ),
    buggy_query=TASK_MEDIUM_BUGGY.strip(),
    max_steps=15,
    grader=_grade_medium,
    expected_hint="Fix: remove LEFT JOIN reviews, use COUNT(DISTINCT o.order_id) >= 2",
)


# ---------------------------------------------------------------------------
# TASK 3 – Hard: Optimization + Correctness (window functions + efficiency)
# ---------------------------------------------------------------------------
# Requirement: For each product, compute:
#   - product name
#   - category
#   - total units sold (across ALL delivered order_items)
#   - avg_rating (from reviews, NULL if no reviews)
#   - revenue_rank: rank within category by revenue (1=highest), using RANK()
#   - category_revenue_pct: this product's revenue as % of its category total
# Only include products that have been sold at least once.
# Bugs planted:
#   1. Uses ROW_NUMBER() instead of RANK() (wrong for ties)
#   2. revenue_rank is global not per-category (missing PARTITION BY)
#   3. category_revenue_pct divides by wrong denominator (product price not category revenue)
#   4. Doesn't filter to delivered orders (includes cancelled)
#   5. Doesn't handle NULL avg_rating (INNER JOIN reviews drops unsold-reviewed products)

TASK_HARD_BUGGY = """
SELECT p.name,
       p.category,
       SUM(oi.quantity)                                    AS total_units_sold,
       AVG(r.rating)                                       AS avg_rating,
       ROW_NUMBER() OVER (ORDER BY SUM(oi.quantity * oi.unit_price) DESC)
                                                           AS revenue_rank,
       ROUND(
         SUM(oi.quantity * oi.unit_price) / p.price * 100, 2
       )                                                   AS category_revenue_pct
FROM   products p
JOIN   order_items oi ON oi.product_id = p.product_id
JOIN   orders o       ON o.order_id    = oi.order_id
JOIN   reviews r      ON r.product_id  = p.product_id
GROUP  BY p.product_id
ORDER  BY revenue_rank;
"""

_HARD_REQUIRED_COLS = {"name", "category", "total_units_sold", "avg_rating",
                        "revenue_rank", "category_revenue_pct"}

def _grade_hard(rows: List[Dict], error: Optional[str], query: str) -> Tuple[float, str]:
    score = 0.0
    parts = []
    query_lower = query.lower()

    if error:
        parts.append(f"Query error: {error}")
        if "window" in error.lower() or "syntax" in error.lower():
            score += 0.02
        return round(max(0.001, score), 3), " | ".join(parts)

    if not rows:
        return 0.051, "No rows returned — check JOINs and WHERE clause."

    cols = {c.lower() for c in rows[0].keys()}

    # --- Column presence (0.15) ---
    required = {"name", "category", "total_units_sold", "avg_rating",
                 "revenue_rank", "category_revenue_pct"}
    present = required & cols
    col_score = len(present) / len(required) * 0.15
    score += col_score
    missing = required - cols
    if missing:
        parts.append(f"Missing columns: {missing}")
    else:
        parts.append("✓ All required columns present")

    # --- Delivered-only filter (0.15) ---
    if "delivered" in query_lower:
        score += 0.15
        parts.append("✓ Filters to delivered orders")
    else:
        parts.append("Missing WHERE o.status='delivered' — includes cancelled orders")

    # --- LEFT JOIN reviews (0.1) ---
    if re.search(r"left\s+join\s+reviews", query_lower):
        score += 0.1
        parts.append("✓ Uses LEFT JOIN for reviews (handles products with no reviews)")
    else:
        parts.append("Should LEFT JOIN reviews so products without reviews aren't excluded")

    # --- RANK() with PARTITION BY category (0.2) ---
    has_rank = "rank()" in query_lower
    has_partition = "partition by" in query_lower and "category" in query_lower
    if has_rank and has_partition:
        score += 0.2
        parts.append("✓ RANK() OVER (PARTITION BY category ...) — correct window function")
    elif has_rank:
        score += 0.07
        parts.append("Has RANK() but missing PARTITION BY category")
    elif "row_number" in query_lower:
        parts.append("ROW_NUMBER() doesn't handle ties — use RANK()")
    elif has_partition:
        score += 0.05
        parts.append("Has PARTITION BY but check window function type")

    # --- category_revenue_pct correctness (0.2) ---
    # Should divide by SUM(all products in same category), not p.price
    if "sum(" in query_lower and "over" in query_lower and "partition" in query_lower:
        score += 0.2
        parts.append("✓ category_revenue_pct uses window SUM for denominator")
    elif "p.price" in query_lower and "category_revenue_pct" in query_lower:
        parts.append("category_revenue_pct denominator wrong — should be category total revenue not p.price")
    else:
        parts.append("category_revenue_pct: use SUM(revenue) OVER (PARTITION BY category) as denominator")
        score += 0.05  # partial for attempting it

    # --- Result sanity: products with sales only (0.1) ---
    name_col = next((c for c in rows[0].keys() if "name" in c.lower()), None)
    if name_col:
        sold_products = {r[name_col] for r in rows}
        # Products NOT sold in delivered orders should be excluded
        if "Desk Lamp" not in sold_products:
            score += 0.05
            parts.append("✓ Products with no delivered sales excluded")
        if len(sold_products) <= 8:
            score += 0.05
            parts.append(f"✓ Reasonable number of products returned ({len(sold_products)})")

    # Bonus: if all checks pass, give full marks
    if score >= 0.95:
        score = 0.999

    # Clamp score strictly between 0 and 1 as per hackathon requirement
    final_score = max(0.001, min(score, 0.999))
    return round(final_score, 3), " | ".join(parts)


TASK_HARD = Task(
    id="task_hard",
    name="SQL Optimization & Correctness",
    difficulty="hard",
    business_requirement=(
        "For each product that was sold in at least one DELIVERED order, produce a report with:\n"
        "  • name — product name\n"
        "  • category — product category\n"
        "  • total_units_sold — total quantity sold across all delivered orders\n"
        "  • avg_rating — average review rating (NULL if no reviews exist)\n"
        "  • revenue_rank — rank of this product by revenue WITHIN its category "
        "(1 = highest revenue in that category); use RANK() to handle ties correctly\n"
        "  • category_revenue_pct — this product's revenue as a percentage of "
        "its category's total revenue (round to 2 decimal places)\n"
        "Order results by category, then revenue_rank."
    ),
    buggy_query=TASK_HARD_BUGGY.strip(),
    max_steps=20,
    grader=_grade_hard,
    expected_hint=(
        "Fix: LEFT JOIN reviews, WHERE o.status='delivered', "
        "RANK() OVER (PARTITION BY p.category ORDER BY revenue DESC), "
        "category_revenue_pct = revenue / SUM(revenue) OVER (PARTITION BY p.category)"
    ),
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASKS: Dict[str, Task] = {
    "task_easy":   TASK_EASY,
    "task_medium": TASK_MEDIUM,
    "task_hard":   TASK_HARD,
}
