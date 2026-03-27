"""
Tests for the SQL Debugger OpenEnv environment.

Run with: python -m pytest tests/ -v
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env.environment import SQLDebuggerEnv
from env.models import Action, Observation, Reward
from tasks.tasks import TASKS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def easy_env():
    return SQLDebuggerEnv("task_easy")


@pytest.fixture
def medium_env():
    return SQLDebuggerEnv("task_medium")


@pytest.fixture
def hard_env():
    return SQLDebuggerEnv("task_hard")


# ---------------------------------------------------------------------------
# Spec compliance
# ---------------------------------------------------------------------------

class TestOpenEnvSpec:
    def test_reset_returns_observation(self, easy_env):
        obs = easy_env.reset()
        assert isinstance(obs, Observation)
        assert obs.step == 0
        assert obs.task_id == "task_easy"
        assert obs.score == 0.0

    def test_step_returns_tuple(self, easy_env):
        easy_env.reset()
        action = Action(sql="SELECT 1")
        result = easy_env.step(action)
        assert len(result) == 4
        obs, reward, done, info = result
        assert isinstance(obs, Observation)
        assert isinstance(reward, Reward)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_reward_range(self, easy_env):
        easy_env.reset()
        _, reward, _, _ = easy_env.step(Action(sql="SELECT 1"))
        assert 0.0 <= reward.value <= 1.0
        assert 0.0 <= reward.correctness <= 1.0
        assert 0.0 <= reward.efficiency <= 1.0
        assert reward.step_penalty <= 0.0

    def test_state_returns_dict(self, easy_env):
        easy_env.reset()
        state = easy_env.state()
        assert isinstance(state, dict)
        assert "task_id" in state
        assert "step" in state
        assert "done" in state

    def test_done_after_max_steps(self, easy_env):
        obs = easy_env.reset()
        max_steps = obs.max_steps
        for _ in range(max_steps):
            obs, _, done, _ = easy_env.step(Action(sql="SELECT 1"))
        assert done is True

    def test_error_after_done(self, easy_env):
        easy_env.reset()
        for _ in range(easy_env._task.max_steps):
            easy_env.step(Action(sql="SELECT 1"))
        with pytest.raises(RuntimeError):
            easy_env.step(Action(sql="SELECT 1"))

    def test_reset_clears_state(self, easy_env):
        easy_env.reset()
        easy_env.step(Action(sql="SELECT 1"))
        easy_env.step(Action(sql="SELECT 2"))
        obs = easy_env.reset()
        assert obs.step == 0
        assert obs.score == 0.0

    def test_all_tasks_exist(self):
        for task_id in ["task_easy", "task_medium", "task_hard"]:
            env = SQLDebuggerEnv(task_id)
            obs = env.reset()
            assert obs.task_id == task_id

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError):
            SQLDebuggerEnv("task_nonexistent")


# ---------------------------------------------------------------------------
# Grader determinism
# ---------------------------------------------------------------------------

class TestGraderDeterminism:
    """Graders must return same score for same inputs."""

    def _run_query(self, env, sql):
        env.reset()
        _, reward, _, info = env.step(Action(sql=sql))
        return info["score"]

    def test_easy_grader_deterministic(self, easy_env):
        sql = "SELECT p.category, SUM(oi.quantity * oi.unit_price) AS total_revenue FROM orders o JOIN order_items oi ON oi.order_id = o.order_id JOIN products p ON p.product_id = oi.product_id WHERE o.status = 'delivered' GROUP BY p.category ORDER BY total_revenue DESC"
        s1 = self._run_query(easy_env, sql)
        s2 = self._run_query(easy_env, sql)
        assert s1 == s2

    def test_medium_grader_deterministic(self, medium_env):
        sql = "SELECT c.name, COUNT(DISTINCT o.order_id) AS delivered_orders, SUM(oi.quantity * oi.unit_price) AS total_spent FROM customers c JOIN orders o ON o.customer_id = c.customer_id JOIN order_items oi ON oi.order_id = o.order_id WHERE o.status = 'delivered' GROUP BY c.customer_id HAVING COUNT(DISTINCT o.order_id) >= 2 ORDER BY total_spent DESC"
        s1 = self._run_query(medium_env, sql)
        s2 = self._run_query(medium_env, sql)
        assert s1 == s2

    def test_hard_grader_deterministic(self, hard_env):
        sql = "SELECT 1 AS name, 'x' AS category, 1 AS total_units_sold, NULL AS avg_rating, 1 AS revenue_rank, 100.0 AS category_revenue_pct"
        s1 = self._run_query(hard_env, sql)
        s2 = self._run_query(hard_env, sql)
        assert s1 == s2


# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------

class TestRewardShaping:
    def test_correct_easy_query_scores_high(self, easy_env):
        correct_sql = """
        SELECT p.category,
               SUM(oi.quantity * oi.unit_price) AS total_revenue
        FROM   orders o
        JOIN   order_items oi ON oi.order_id = o.order_id
        JOIN   products p ON p.product_id = oi.product_id
        WHERE  o.status = 'delivered'
        GROUP  BY p.category
        ORDER  BY total_revenue DESC
        """
        easy_env.reset()
        _, reward, _, info = easy_env.step(Action(sql=correct_sql))
        assert info["score"] >= 0.8, f"Expected >= 0.8, got {info['score']}"

    def test_buggy_query_scores_low(self, easy_env):
        easy_env.reset()
        _, _, _, info = easy_env.step(Action(sql="SELECT 1"))
        assert info["score"] < 0.5

    def test_partial_fix_scores_intermediate(self, easy_env):
        """Fixing only the revenue formula (not the filter) gives intermediate score."""
        partial_fix = """
        SELECT p.category,
               SUM(oi.quantity * oi.unit_price) AS total_revenue
        FROM   orders o
        JOIN   order_items oi ON oi.order_id = o.order_id
        JOIN   products p ON p.product_id = oi.product_id
        GROUP  BY p.category
        ORDER  BY total_revenue DESC
        """
        easy_env.reset()
        _, _, _, info = easy_env.step(Action(sql=partial_fix))
        assert 0.1 < info["score"] < 0.9, f"Expected intermediate score, got {info['score']}"

    def test_score_monotonically_accessible(self, easy_env):
        """best_score in info should never decrease."""
        easy_env.reset()
        best = 0.0
        correct_sql = """
        SELECT p.category, SUM(oi.quantity * oi.unit_price) AS total_revenue
        FROM orders o JOIN order_items oi ON oi.order_id = o.order_id
        JOIN products p ON p.product_id = oi.product_id
        WHERE o.status = 'delivered'
        GROUP BY p.category ORDER BY total_revenue DESC
        """
        for sql in [correct_sql, "SELECT 1", correct_sql]:
            _, _, done, info = easy_env.step(Action(sql=sql))
            assert info["best_score"] >= best
            best = info["best_score"]
            if done:
                break


# ---------------------------------------------------------------------------
# Database integrity
# ---------------------------------------------------------------------------

class TestDatabase:
    def test_db_has_expected_tables(self, easy_env):
        tables = easy_env._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {r[0] for r in tables}
        expected = {"customers", "products", "orders", "order_items", "reviews"}
        assert expected.issubset(table_names)

    def test_db_seeded_with_data(self, easy_env):
        count = easy_env._db.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        assert count >= 5
        count = easy_env._db.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        assert count >= 10

    def test_reset_recreates_db(self, easy_env):
        # Tamper with DB
        easy_env._db.execute("DELETE FROM customers")
        easy_env._db.commit()
        count_before = easy_env._db.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        assert count_before == 0
        # Reset
        easy_env.reset()
        count_after = easy_env._db.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        assert count_after >= 5
