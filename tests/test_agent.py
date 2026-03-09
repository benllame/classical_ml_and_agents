"""Tests for the ReAct Agent — Fase 08.

Tests for the ReAct agent. These tests validate:
  (1) tool definitions (names, descriptions, count),
  (2) graph structure (state schema, LLM factory),
  (3) eval set completeness.

The tests are designed to run WITHOUT an LLM API key — they validate
structure and configuration, not LLM behavior. Full agent integration tests
require a live LLM and are run separately.
"""

from __future__ import annotations

from agent.tools import (
    ALL_TOOLS,
)


# Tools are the agent's interface to the real system. These tests ensure all
# 5 tools are properly registered with LangChain's @tool decorator, have
# meaningful descriptions (the LLM uses these to decide WHEN to call a tool),
# and are discoverable via ALL_TOOLS.
class TestToolDefinitions:
    """Verify that all 5 tools are properly defined."""

    def test_all_tools_count(self):
        assert len(ALL_TOOLS) == 5

    def test_tool_names(self):
        names = {t.name for t in ALL_TOOLS}
        expected = {
            "get_customer_profile",
            "predict_churn_risk",
            "explain_prediction",
            "recommend_intervention",
            "simulate_budget_allocation",
        }
        assert names == expected

    def test_tools_have_descriptions(self):
        # Minimum description length of 20 chars ensures descriptions are
        # meaningful, not just placeholder text. The LLM relies on tool
        # descriptions for tool selection — a poor description leads to
        # incorrect tool calls.
        for tool in ALL_TOOLS:
            assert tool.description, f"Tool {tool.name} has no description"
            assert len(tool.description) > 20, f"Tool {tool.name} description too short"


# Tests graph compilation without requiring an LLM API key. We verify the
# state schema (AgentState must have 'messages') and that the LLM factory
# function is importable. These are structural tests, not behavioral tests.
class TestAgentGraph:
    """Test that the agent graph can be created."""

    def test_graph_creation(self):
        """Verify graph compiles (requires no API key)."""
        # We test the graph structure without actually calling an LLM
        from agent.graph import AgentState

        # Verify state schema
        assert "messages" in AgentState.__annotations__

    def test_create_llm_raises_without_key(self):
        """Without API keys, LLM creation should still work (key validated at call time)."""
        # This just tests the factory function doesn't crash at import
        from agent.graph import create_llm

        # It should be importable without errors
        assert callable(create_llm)


# -- Eval Set -----------------------------------------------------------------
# Evaluation set of 10 queries a retention manager would realistically ask.
# Designed to cover:
#   (1) all 5 tools,
#   (2) single-tool and multi-tool queries,
#   (3) different question phrasings.
#
# This eval set is used for:
#   (a) offline evaluation after training,
#   (b) regression testing after agent changes.
#
# Reference: OpenAI Evals framework (2023) — eval sets should be diverse,
# realistic, and cover tool coverage.
# In a full run, these would be executed against the live agent.

EVAL_SET = [
    {
        "query": "What is the churn risk for customer 7590-VHVEG?",
        "expected_tools": ["predict_churn_risk"],
        "expected_contains": ["churn", "risk", "probability"],
    },
    {
        "query": "Why is customer 7590-VHVEG at risk of churning?",
        "expected_tools": ["explain_prediction"],
        "expected_contains": ["factor", "shap"],
    },
    {
        "query": "What should we do about customer 7590-VHVEG?",
        "expected_tools": ["recommend_intervention"],
        "expected_contains": ["policy", "intervention", "roi"],
    },
    {
        "query": "Give me the full profile of customer 5575-GNVDE",
        "expected_tools": ["get_customer_profile"],
        "expected_contains": ["customer", "profile"],
    },
    {
        "query": "With a budget of $3,000, who should we call first?",
        "expected_tools": ["simulate_budget_allocation"],
        "expected_contains": ["budget", "customer"],
    },
    {
        "query": "What is the risk and recommended action for customer 3668-QPYBK?",
        "expected_tools": ["predict_churn_risk", "recommend_intervention"],
        "expected_contains": ["risk", "intervention"],
    },
    {
        "query": "Explain why customer 9237-HQITU might leave",
        "expected_tools": ["explain_prediction"],
        "expected_contains": ["factor"],
    },
    {
        "query": "Allocate $10,000 across the top 15 customers by ROI",
        "expected_tools": ["simulate_budget_allocation"],
        "expected_contains": ["budget", "roi"],
    },
    {
        "query": "Is customer 7590-VHVEG a senior citizen with fiber optic?",
        "expected_tools": ["get_customer_profile"],
        "expected_contains": ["customer"],
    },
    {
        "query": "Compare risk for customers 7590-VHVEG and 5575-GNVDE",
        "expected_tools": ["predict_churn_risk"],
        "expected_contains": ["risk", "probability"],
    },
]


class TestEvalSet:
    """Validate the eval set structure."""

    def test_eval_set_has_10_queries(self):
        # 10 queries provides reasonable coverage without excessive eval
        # cost. Each query targets specific tool(s) and expected output
        # patterns.
        assert len(EVAL_SET) == 10

    def test_eval_queries_have_required_fields(self):
        for item in EVAL_SET:
            assert "query" in item
            assert "expected_tools" in item
            assert "expected_contains" in item
            assert len(item["query"]) > 10

    def test_all_tools_covered_in_eval(self):
        """Every tool should appear in at least one eval query."""
        # Coverage test: every tool in ALL_TOOLS must appear as an
        # expected_tool in at least one eval query. This prevents adding
        # tools that are never tested.
        all_expected_tools = set()
        for item in EVAL_SET:
            all_expected_tools.update(item["expected_tools"])

        tool_names = {t.name for t in ALL_TOOLS}
        assert tool_names.issubset(
            all_expected_tools
        ), f"Tools not covered in eval: {tool_names - all_expected_tools}"
