"""
Responsibilities:
  1. Load config/agents.yaml and config/tasks.yaml (no config.yaml — removed)
  2. Read knowledge/user_preferences.txt and inject into every agent prompt
  3. Resolve tool names from agents.yaml → actual tool instances from tools/
  4. Build each agent: system prompt = preferences + role + goal + backstory + constraints
  5. Build each task: user message = task description + current state context
  6. Wire everything into a LangGraph graph with conditional routing
  7. Expose QuantCrew.run() as the single public interface
"""

import json
import yaml
import os
from pathlib import Path
from typing import Optional
from datetime import datetime
from functools import lru_cache

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from langgraph.graph import StateGraph, END

from tools import TOOL_REGISTRY


LLM_MODEL       = "gpt-4o-mini"
MAX_RETRIES     = "3"
SEARCH_RESULTS  = "5"


# ── Paths ──────────────────────────────────────────────────────────────────────

BASE          = Path(__file__).parent
AGENTS_PATH   = BASE / "config" / "agents.yaml"
TASKS_PATH    = BASE / "config" / "tasks.yaml"
KNOWLEDGE_DIR = BASE / "knowledge"
OUTPUT_DIR    = BASE / "output"


# ── YAML + knowledge loaders (cached) ─────────────────────────────────────────

@lru_cache(maxsize=1)
def _agents() -> dict:
    return yaml.safe_load(AGENTS_PATH.read_text())

@lru_cache(maxsize=1)
def _tasks() -> dict:
    return yaml.safe_load(TASKS_PATH.read_text())

@lru_cache(maxsize=1)
def _knowledge() -> str:
    """Concatenate all .txt files in knowledge/ into one injected block."""
    texts = []
    for f in sorted(KNOWLEDGE_DIR.glob("*.txt")):
        texts.append(f"--- {f.name} ---\n{f.read_text().strip()}")
    return "\n\n".join(texts)


# ── Shared PipelineState ───────────────────────────────────────────────────────

class PipelineState(BaseModel):
    # ── User input ─────────────────────────────────────────────────────────────
    asset_class:         Optional[str]   = None   # equities | crypto | macro | mixed
    user_tickers:        Optional[list]  = None
    user_theme:          Optional[str]   = None

    # ── market_agent ───────────────────────────────────────────────────────────
    theme:               Optional[str]   = None
    tickers:             Optional[list]  = None
    hypothesis:          Optional[str]   = None

    # ── fundamental_agent ──────────────────────────────────────────────────────
    fundamental_pass:    Optional[bool]  = None
    fundamental_reason:  Optional[str]   = None

    # ── regime_agent ───────────────────────────────────────────────────────────
    regime:              Optional[str]   = None   # TRENDING | MEAN_REVERTING | MIXED
    volatility:          Optional[str]   = None   # HIGH | LOW | EXPANDING | CONTRACTING

    # ── sentiment_agent ────────────────────────────────────────────────────────
    sentiment:           Optional[str]   = None   # UNDERCROWDED | NEUTRAL | CROWDED | HYPE
    sentiment_risk:      Optional[str]   = None   # LOW | MED | HIGH

    # ── risk_agent ─────────────────────────────────────────────────────────────
    risk_score:          Optional[str]   = None   # LOW | MED | HIGH
    risks:               Optional[list]  = None

    # ── alpha_scorer_agent ─────────────────────────────────────────────────────
    alpha_score:         Optional[float] = None
    alpha_reasoning:     Optional[str]   = None

    # ── manager_agent ──────────────────────────────────────────────────────────
    decision:            Optional[str]   = None   # PROCEED | FIXABLE_REJECT | HARD_VETO
    priority:            Optional[str]   = None   # HIGH | MED | LOW
    manager_reasoning:   Optional[str]   = None

    # ── quant_agent ────────────────────────────────────────────────────────────
    strategy_type:       Optional[str]   = None
    signals:             Optional[list]  = None
    universe:            Optional[list]  = None   # List[str] — explicit instruments
    horizon:             Optional[str]   = None

    # ── Pipeline control ───────────────────────────────────────────────────────
    retry_count:         int             = 0
    rejection_feedback:  Optional[str]   = None
    hard_veto_message:   Optional[str]   = None
    run_complete:        bool            = False


# ── Agent / prompt builder ────────────────────────────────────────────────────

def _build_system_prompt(agent_key: str) -> str:
    """
    System prompt = knowledge block + role + goal + backstory + constraints.
    Knowledge is always injected first so global directives take precedence.
    """
    agent       = _agents()[agent_key]
    constraints = "\n".join(
        f"  - {c}" for c in agent.get("behaviour", {}).get("constraints", [])
    )

    return f"""
{_knowledge()}

{'━'*44}
YOUR ROLE
{'━'*44}
Role:      {agent['role']}
Goal:      {agent['goal'].strip()}
Backstory: {agent['backstory'].strip()}

{'━'*44}
YOUR CONSTRAINTS
{'━'*44}
{constraints if constraints else "No additional constraints."}

{'━'*44}
OUTPUT FORMAT
{'━'*44}
Always respond with a single valid JSON object matching the expected output schema.
No preamble, no markdown fences, no commentary outside the JSON.
""".strip()


def _get_tools(agent_key: str) -> list:
    """Resolve tool names from agents.yaml → real instances via TOOL_REGISTRY."""
    names = _agents()[agent_key].get("tools", [])
    tools = []
    for name in names:
        if name not in TOOL_REGISTRY:
            raise ValueError(
                f"Tool '{name}' listed for agent '{agent_key}' "
                f"has no entry in tools/TOOL_REGISTRY"
            )
        tools.append(TOOL_REGISTRY[name]())
    return tools


def _llm(structured_output_schema=None, with_tools: list | None = None) -> ChatOpenAI:
    base = ChatOpenAI(
        model=LLM_MODEL
    )
    if with_tools:
        return base.bind_tools(with_tools)
    if structured_output_schema:
        return base.with_structured_output(structured_output_schema)
    return base


# ── Task runner ────────────────────────────────────────────────────────────────

def _run_task(task_key: str, state: PipelineState, output_schema: type):
    """
    Execute one task end-to-end:
      1. Load task + agent definition from YAML
      2. Build system prompt (knowledge + role + constraints)
      3. Build user message (task description + declared state inputs)
      4. ReAct loop if agent has search tools, else single structured call
      5. Return parsed output object
    """
    task        = _tasks()[task_key]
    agent_key   = task["agent"]
    agent_cfg   = _agents()[agent_key]
    uses_search = bool(agent_cfg.get("tools"))

    system_prompt = _build_system_prompt(agent_key)
    tools         = _get_tools(agent_key) if uses_search else []

    # Only surface the fields this task declared as inputs
    state_dict    = state.model_dump()
    context_lines = [
        f"  {field}: {state_dict[field]}"
        for field in task.get("inputs", [])
        if field in state_dict and state_dict[field] is not None
    ]
    context_block = "\n".join(context_lines) if context_lines else "  (no prior context)"

    user_message = (
        f"TASK: {task_key.replace('_', ' ').upper()}\n\n"
        f"{task['description'].strip()}\n\n"
        f"{'━'*44}\n"
        f"CURRENT PIPELINE STATE (your declared inputs)\n"
        f"{'━'*44}\n"
        f"{context_block}\n\n"
        f"Respond with a JSON object matching the expected output schema exactly."
    )

    if uses_search:
        return _react_loop(system_prompt, user_message, tools, output_schema)
    else:
        return _structured_call(system_prompt, user_message, output_schema)


def _log(msg: str, indent: int = 0):
    """Timestamped logger used across all nodes."""
    prefix = "  " * indent
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  [{ts}] {prefix}{msg}")


def _react_loop(system_prompt, user_message, tools, output_schema):
    """ReAct loop: LLM searches until satisfied, then emits structured output."""
    tool_map     = {t.name: t for t in tools}
    research_llm = _llm(with_tools=tools)
    output_llm   = _llm(structured_output_schema=output_schema)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    search_count = 0
    for round_num in range(8):
        response = research_llm.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            if search_count == 0:
                _log("No searches performed — reasoning from existing knowledge only", indent=1)
            else:
                _log(f"Search phase complete after {search_count} query/queries", indent=1)
            break
        for call in response.tool_calls:
            search_count += 1
            query = call["args"].get("query", str(call["args"]))
            _log(f"Search {search_count}: [{query}]", indent=1)
            result = tool_map[call["name"]].invoke(call["args"])
            if isinstance(result, list):
                for item in result:
                    title   = item.get("title",   "no title")
                    url     = item.get("url",     "")
                    snippet = item.get("content", "")[:120].replace("\n", " ")
                    _log(f"  result: {title[:70]}", indent=2)
                    _log(f"  url:    {url[:70]}",   indent=2)
                    if snippet:
                        _log(f"  snip:   {snippet}...", indent=2)
            messages.append(ToolMessage(
                content=json.dumps(result),
                tool_call_id=call["id"],
            ))

    research_summary = "\n\n".join(
        m.content for m in messages
        if hasattr(m, "content") and isinstance(m.content, str) and m.content
    )

    return output_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            f"Based on your research:\n\n{research_summary}\n\n"
            "Now produce the final structured JSON output."
        )),
    ])


def _structured_call(system_prompt, user_message, output_schema):
    """Single structured LLM call — no search tools."""
    return _llm(structured_output_schema=output_schema).invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ])


# ── Output schemas (mirror tasks.yaml expected_output) ────────────────────────

class MarketOut(BaseModel):
    theme:      str
    tickers:    list[str]
    hypothesis: str

class FundamentalOut(BaseModel):
    fundamental_pass:   bool
    fundamental_reason: str

class RegimeOut(BaseModel):
    regime:     str   # TRENDING | MEAN_REVERTING | MIXED
    volatility: str   # HIGH | LOW | EXPANDING | CONTRACTING

class SentimentOut(BaseModel):
    sentiment:      str   # UNDERCROWDED | NEUTRAL | CROWDED | HYPE
    sentiment_risk: str   # LOW | MED | HIGH

class RiskOut(BaseModel):
    risk_score: str        # LOW | MED | HIGH
    risks:      list[str]

class AlphaOut(BaseModel):
    alpha_score:     float = Field(ge=0.0, le=10.0)
    alpha_reasoning: str

class ManagerOut(BaseModel):
    decision:           str   # PROCEED | FIXABLE_REJECT | HARD_VETO
    priority:           str   # HIGH | MED | LOW
    reasoning:          str
    rejection_feedback: str = ""
    hard_veto_message:  str = ""

class QuantOut(BaseModel):
    strategy_type: str
    signals:       list[str]
    universe:      list[str]  # explicit list of instruments
    horizon:       str


# ── Node functions ─────────────────────────────────────────────────────────────

def _node_header(name: str, state: PipelineState):
    """Print a clear section header before each node runs."""
    label = {
        "market_node":       "MARKET AGENT — News & Catalyst Scanner",
        "fundamental_node":  "FUNDAMENTAL AGENT — Sanity Checker",
        "regime_node":       "REGIME AGENT — Market Classifier",
        "sentiment_node":    "SENTIMENT AGENT — Crowding Detector",
        "risk_node":         "RISK AGENT — Stress Tester",
        "alpha_scorer_node": "ALPHA SCORER — Potential Ranker",
        "manager_node":      "MANAGER AGENT — Decision Maker",
        "quant_node":        "QUANT AGENT — Strategy Designer",
    }.get(name, name.upper())
    print(f"\n{'═'*60}")
    print(f"  {label}")
    if state.retry_count > 0:
        print(f"  (retry #{state.retry_count})")
    print(f"{'═'*60}")


def market_node(state: PipelineState) -> dict:
    _node_header("market_node", state)

    if state.rejection_feedback:
        _log("Previous attempt was rejected. Feedback received:")
        _log(f"[{state.rejection_feedback}]", indent=1)
        _log("Will search for a different catalyst and tickers this time.")

    mode = (
        f"ticker-first ({state.user_tickers})" if state.user_tickers
        else f"theme-guided ({state.user_theme})" if state.user_theme
        else "autonomous — scanning live news for catalyst"
    )
    _log(f"Entry mode: {mode}")
    _log("Starting web search for recent market-moving news...")

    out: MarketOut = _run_task("identify_alpha_theme", state, MarketOut)

    _log("Catalyst identified. Output:")
    _log(f"Theme    : {out.theme}", indent=1)
    _log(f"Tickers  : {out.tickers}", indent=1)
    _log(f"Hypothesis: {out.hypothesis}", indent=1)

    return {
        "theme":              out.theme,
        "tickers":            out.tickers,
        "hypothesis":         out.hypothesis,
        "rejection_feedback": None,
    }


def fundamental_node(state: PipelineState) -> dict:
    _node_header("fundamental_node", state)
    _log(f"Checking {len(state.tickers)} tickers: {state.tickers}")
    _log("Will verify: liquidity, market cap, earnings health, distress flags.")
    _log("Searching for fundamental data...")

    out: FundamentalOut = _run_task("check_fundamentals", state, FundamentalOut)

    status = "PASS" if out.fundamental_pass else "FAIL"
    _log(f"Verdict: {status}")
    _log("Reasoning:")
    for line in out.fundamental_reason.split(". "):
        line = line.strip()
        if line:
            _log(f"  {line}.", indent=1)

    if not out.fundamental_pass:
        _log("NOTE: Failed tickers will be dropped. Pipeline continues with clean assets.")

    return {
        "fundamental_pass":   out.fundamental_pass,
        "fundamental_reason": out.fundamental_reason,
    }


def regime_node(state: PipelineState) -> dict:
    _node_header("regime_node", state)
    _log(f"Classifying market regime for: {state.tickers}")
    _log(f"Theme context: {state.theme}")
    _log("Searching for recent price action, volatility data, and benchmark behavior...")

    out: RegimeOut = _run_task("classify_regime", state, RegimeOut)

    _log("Regime classification complete:")
    _log(f"Trend      : {out.regime}", indent=1)
    _log(f"Volatility : {out.volatility}", indent=1)

    regime_implications = {
        "TRENDING":      "momentum strategies favored, trend-following signals applicable",
        "MEAN_REVERTING":"fade-the-extreme strategies favored, stat-arb applicable",
        "MIXED":         "elevated noise — strategy selection requires caution",
    }
    vol_implications = {
        "HIGH":        "wide stops needed, position sizing down",
        "LOW":         "tighter spreads, higher conviction entries viable",
        "EXPANDING":   "volatility increasing — reduce size, watch for regime shift",
        "CONTRACTING": "volatility compressing — breakout setup may be forming",
    }
    _log(f"Implication: {regime_implications.get(out.regime, '')}", indent=1)
    _log(f"Vol note   : {vol_implications.get(out.volatility, '')}", indent=1)

    return {"regime": out.regime, "volatility": out.volatility}


def sentiment_node(state: PipelineState) -> dict:
    _node_header("sentiment_node", state)
    _log(f"Assessing sentiment and crowding for: {state.tickers}")
    _log(f"Theme: {state.theme}")
    _log("Searching for news flow intensity, analyst consensus, social signals...")

    out: SentimentOut = _run_task("assess_sentiment", state, SentimentOut)

    _log("Sentiment assessment complete:")
    _log(f"Sentiment      : {out.sentiment}", indent=1)
    _log(f"Crowding risk  : {out.sentiment_risk}", indent=1)

    sentiment_notes = {
        "UNDERCROWDED": "potential contrarian opportunity — low positioning means room to run",
        "NEUTRAL":      "balanced positioning — no significant crowding concern",
        "CROWDED":      "elevated reversal risk — many already positioned, late entry risk",
        "HYPE":         "extreme crowding — high probability of sharp reversal, proceed with caution",
    }
    _log(f"Note: {sentiment_notes.get(out.sentiment, '')}", indent=1)

    if out.sentiment_risk == "HIGH":
        _log("WARNING: HIGH sentiment risk detected. Risk agent will stress test this heavily.", indent=1)

    return {"sentiment": out.sentiment, "sentiment_risk": out.sentiment_risk}


def risk_node(state: PipelineState) -> dict:
    _node_header("risk_node", state)
    _log(f"Stress testing idea: {state.theme}")
    _log(f"Tickers in scope: {state.tickers}")
    _log(f"Context going in — Regime: {state.regime} | Vol: {state.volatility} | Sentiment: {state.sentiment}")
    _log("Searching for macro risks, regulatory threats, and tail events...")
    _log("Posture: ADVERSARIAL — attempting to break the thesis.")

    out: RiskOut = _run_task("stress_test_risks", state, RiskOut)

    _log("Risk assessment complete:")
    _log(f"Risk score : {out.risk_score}", indent=1)
    _log(f"Risks identified ({len(out.risks)}):", indent=1)
    for i, risk in enumerate(out.risks, 1):
        _log(f"{i}. {risk}", indent=2)

    if out.risk_score == "HIGH":
        _log("CAUTION: HIGH risk score. Alpha scorer will apply a -2.0 penalty.", indent=1)
    elif out.risk_score == "MED":
        _log("Manageable risk level. Idea can proceed with appropriate sizing.", indent=1)
    else:
        _log("Low risk environment. Favorable conditions for the thesis.", indent=1)

    return {"risk_score": out.risk_score, "risks": out.risks}


def alpha_scorer_node(state: PipelineState) -> dict:
    _node_header("alpha_scorer_node", state)
    _log("Scoring alpha potential across 4 dimensions (0-2.5 each):")
    _log("1. Regime alignment  — does regime support the implied strategy?", indent=1)
    _log("2. Sentiment edge    — is sentiment mispriced vs fundamentals?", indent=1)
    _log("3. Fundamental quality — how clean are the underlying assets?", indent=1)
    _log("4. Expected persistence — is this a durable move or a one-day catalyst?", indent=1)
    _log("No search — scoring purely from pipeline outputs collected so far.")

    out: AlphaOut = _run_task("score_alpha_potential", state, AlphaOut)

    _log(f"Alpha score: {out.alpha_score:.1f} / 10.0")
    _log("Scoring breakdown:")
    for line in out.alpha_reasoning.split(";"):
        line = line.strip()
        if line:
            _log(f"  {line}", indent=1)

    threshold = 6.0
    if out.alpha_score >= threshold:
        _log(f"Score {out.alpha_score:.1f} >= threshold {threshold} — eligible to PROCEED")
    else:
        _log(f"Score {out.alpha_score:.1f} < threshold {threshold} — likely REJECT or VETO ahead")

    return {"alpha_score": out.alpha_score, "alpha_reasoning": out.alpha_reasoning}


def manager_node(state: PipelineState) -> dict:
    _node_header("manager_node", state)
    _log("Aggregating full pipeline output for final decision:")
    _log(f"  Theme      : {state.theme}", indent=1)
    _log(f"  Tickers    : {state.tickers}", indent=1)
    _log(f"  Fund pass  : {state.fundamental_pass}", indent=1)
    _log(f"  Regime     : {state.regime} | Vol: {state.volatility}", indent=1)
    _log(f"  Sentiment  : {state.sentiment} | Risk: {state.sentiment_risk}", indent=1)
    _log(f"  Risk score : {state.risk_score}", indent=1)
    _log(f"  Alpha score: {state.alpha_score}", indent=1)
    _log(f"  Retry count: {state.retry_count}", indent=1)
    _log("Decision criteria:")
    _log("  PROCEED       → alpha >= 6.0, fund pass, risk LOW/MED", indent=1)
    _log("  FIXABLE_REJECT → correctable issue (bad tickers, weak catalyst, mismatch)", indent=1)
    _log("  HARD_VETO     → structural failure (risk HIGH, alpha < 3.0, fund fail + unfixable)", indent=1)
    _log("No search — deciding on evidence in state only.")

    out: ManagerOut = _run_task("make_research_decision", state, ManagerOut)

    _log(f"Decision: {out.decision}")

    if out.decision == "PROCEED":
        _log(f"Priority  : {out.priority}", indent=1)
        _log("Reasoning :")
        for line in out.reasoning.split(". "):
            line = line.strip()
            if line:
                _log(f"  {line}.", indent=2)

    elif out.decision == "FIXABLE_REJECT":
        _log("Reason for rejection:")
        _log(out.reasoning, indent=2)
        _log("Feedback to market_agent for next attempt:")
        _log(f"[{out.rejection_feedback}]", indent=2)
        _log(f"Retry count will become: {state.retry_count + 1}")
        if state.retry_count + 1 >= 2:
            _log("WARNING: Next retry will be the last allowed before forced escalation.")

    elif out.decision == "HARD_VETO":
        _log("Structural failure — this idea cannot be fixed by retrying:")
        _log(out.reasoning, indent=2)
        _log("User-facing message:")
        _log(f"[{out.hard_veto_message}]", indent=2)

    updates = {
        "decision":          out.decision,
        "priority":          out.priority,
        "manager_reasoning": out.reasoning,
    }
    if out.decision == "FIXABLE_REJECT":
        updates["rejection_feedback"] = out.rejection_feedback
        updates["retry_count"]        = state.retry_count + 1
    elif out.decision == "HARD_VETO":
        updates["hard_veto_message"] = out.hard_veto_message
        updates["run_complete"]      = True
    return updates


def quant_node(state: PipelineState) -> dict:
    _node_header("quant_node", state)
    _log("Translating approved idea into a quantitative research brief.")
    _log(f"Working with:")
    _log(f"  Theme    : {state.theme}", indent=1)
    _log(f"  Tickers  : {state.tickers}", indent=1)
    _log(f"  Regime   : {state.regime} | Vol: {state.volatility}", indent=1)
    _log(f"  Priority : {state.priority}", indent=1)
    _log("No search — designing from pipeline outputs.")

    out: QuantOut = _run_task("design_quant_strategy", state, QuantOut)

    _log("Quant brief ready:")
    _log(f"Strategy type : {out.strategy_type}", indent=1)
    _log(f"Time horizon  : {out.horizon}", indent=1)
    _log(f"Universe ({len(out.universe)} instruments):", indent=1)
    for inst in out.universe:
        _log(f"  {inst}", indent=2)
    _log(f"Signals ({len(out.signals)}):", indent=1)
    for sig in out.signals:
        _log(f"  {sig}", indent=2)

    return {
        "strategy_type": out.strategy_type,
        "signals":       out.signals,
        "universe":      out.universe,
        "horizon":       out.horizon,
        "run_complete":  True,
    }


# ── Routing ────────────────────────────────────────────────────────────────────

def _route_manager(state: PipelineState) -> str:
    print(f"\n{'─'*60}")
    print(f"  ROUTER — evaluating manager decision: {state.decision}")
    print(f"{'─'*60}")

    if state.decision == "PROCEED":
        _log("Routing to quant_node — idea approved for strategy design.")
        return "quant_node"

    if state.decision == "FIXABLE_REJECT":
        hard_cap = min(MAX_RETRIES, 2)
        if state.retry_count >= hard_cap:
            _log(f"retry_count={state.retry_count} has reached hard cap ({hard_cap}).")
            _log("Cannot loop again. Ending pipeline — user should review the rejection feedback.")
            return END
        _log(f"Routing back to market_node (attempt {state.retry_count} of {hard_cap} max).")
        _log(f"market_node will receive this feedback: [{state.rejection_feedback}]")
        return "market_node"

    _log("HARD_VETO confirmed. Terminating pipeline.")
    _log(f"Reason: {state.hard_veto_message}")
    return END


# ── Graph ──────────────────────────────────────────────────────────────────────

def _build_graph():
    g = StateGraph(PipelineState)

    g.add_node("market_node",       market_node)
    g.add_node("fundamental_node",  fundamental_node)
    g.add_node("regime_node",       regime_node)
    g.add_node("sentiment_node",    sentiment_node)
    g.add_node("risk_node",         risk_node)
    g.add_node("alpha_scorer_node", alpha_scorer_node)
    g.add_node("manager_node",      manager_node)
    g.add_node("quant_node",        quant_node)

    g.set_entry_point("market_node")

    g.add_edge("market_node",       "fundamental_node")
    g.add_edge("fundamental_node",  "regime_node")
    g.add_edge("regime_node",       "sentiment_node")
    g.add_edge("sentiment_node",    "risk_node")
    g.add_edge("risk_node",         "alpha_scorer_node")
    g.add_edge("alpha_scorer_node", "manager_node")
    g.add_edge("quant_node",        END)

    g.add_conditional_edges(
        "manager_node",
        _route_manager,
        {"market_node": "market_node", "quant_node": "quant_node", END: END},
    )

    return g.compile()


# ── Public interface ───────────────────────────────────────────────────────────

class QuantCrew:
    """
    Main crew object. Inspired by CrewAI's Crew class.

    Usage:
        crew = QuantCrew()
        result = crew.run()                                        # autonomous
        result = crew.run(asset_class="crypto")                    # asset-scoped
        result = crew.run(asset_class="equities", theme="AI")     # theme-guided
        result = crew.run(asset_class="mixed", tickers=["NVDA"])  # ticker-first
    """

    def __init__(self):
        self._graph = _build_graph()
        OUTPUT_DIR.mkdir(exist_ok=True)

    def run(
        self,
        asset_class: Optional[str]       = None,
        tickers:     Optional[list[str]] = None,
        theme:       Optional[str]       = None,
    ) -> PipelineState:

        print("\n" + "="*50)
        print("  QUANT CREW — STARTING")
        print(f"  asset_class : {asset_class or 'auto-detect'}")
        mode = (f"tickers={tickers}" if tickers
                else f"theme='{theme}'"  if theme
                else "autonomous")
        print(f"  mode        : {mode}")
        print("="*50)

        initial = PipelineState(
            asset_class=asset_class,
            user_tickers=tickers,
            user_theme=theme,
        )

        result = self._graph.invoke(initial)
        state  = PipelineState(**result)

        self._print_report(state)
        self._save(state)
        return state

    def _print_report(self, s: PipelineState):
        print("\n" + "="*50)
        print("  FINAL REPORT")
        print("="*50)

        if s.decision == "PROCEED":
            print(f"\n  DECISION  : PROCEED ({s.priority})")
            print(f"  Theme     : {s.theme}")
            print(f"  Tickers   : {s.tickers}")
            print(f"  Alpha     : {s.alpha_score:.1f}/10")
            print(f"  Regime    : {s.regime} | Vol: {s.volatility}")
            print(f"  Sentiment : {s.sentiment} | Risk: {s.sentiment_risk}")
            print(f"  Risk      : {s.risk_score}")
            print(f"\n  ── Quant Brief ──────────────────────")
            print(f"  Strategy  : {s.strategy_type}")
            print(f"  Horizon   : {s.horizon}")
            print(f"  Universe  :")
            for inst in (s.universe or []):
                print(f"    · {inst}")
            print(f"  Signals   :")
            for sig in (s.signals or []):
                print(f"    · {sig}")
            print(f"\n  ── Manager Reasoning ────────────────")
            print(f"  {s.manager_reasoning}")

        elif s.decision == "HARD_VETO":
            print(f"\n  DECISION  : HARD VETO")
            print(f"  Theme     : {s.theme}")
            print(f"  Reason    : {s.hard_veto_message}")

        else:
            print(f"\n  Pipeline ended after {s.retry_count} retries.")
            print(f"  Last feedback: {s.rejection_feedback}")

    def _save(self, s: PipelineState):
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = OUTPUT_DIR / f"run_{ts}.json"
        path.write_text(json.dumps(s.model_dump(), indent=2, default=str))
        print(f"\n  Saved → {path}")