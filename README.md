# Quant Idea Pipeline

An autonomous 8-agent LangGraph pipeline that scans live financial news, stress-tests ideas through a sequential research workflow, and produces a concrete quantitative research brief — or kills the idea before wasting a quant researcher's time.

---

## How it works

The pipeline runs 8 agents in sequence. Each agent does exactly one thing, writes its output to a shared state object, and passes it to the next. No agent knows what the others are doing — they only read what they need from state.

```
user input (optional)
      │
      ▼
 market agent          ← scans live news, finds catalyst-driven tickers
      │
      ▼
 fundamental agent     ← filters illiquid / distressed / untradeable assets
      │
      ▼
 regime agent          ← classifies trend direction + volatility environment
      │
      ▼
 sentiment agent       ← detects crowding, hype, and reversal risk
      │
      ▼
 risk agent            ← adversarially stress-tests the idea
      │
      ▼
 alpha scorer          ← scores alpha potential 0–10 across 4 dimensions
      │
      ▼
 manager agent         ← PROCEED / FIXABLE_REJECT / HARD_VETO
      │
      ├─── PROCEED ──────────► quant agent → research brief
      │
      ├─── FIXABLE_REJECT ───► loop back to market agent (max 2 retries)
      │                        with specific feedback injected into prompt
      │
      └─── HARD_VETO ────────► pipeline ends, reason surfaced to user
```

---

## Agents

| Agent | Role | Uses search |
|---|---|---|
| `market_agent` | Scans financial news (last 24–72h), finds a catalyst, selects 3–5 directly impacted tickers, forms a directional hypothesis | Yes |
| `fundamental_agent` | Checks each ticker for liquidity, market cap, earnings health, distress flags. Drops failed tickers, continues with clean ones | Yes |
| `regime_agent` | Classifies trend (UP / DOWN / RANGE) and volatility (HIGH / LOW / EXPANDING / CONTRACTING) using recent price action | Yes |
| `sentiment_agent` | Detects crowding and positioning risk. Classifies as UNDERCROWDED / NEUTRAL / CROWDED / HYPE | Yes |
| `risk_agent` | Adversarially stress-tests the idea. Must produce at least 3 specific risks. Cannot assign LOW if a binary tail risk exists | Yes |
| `alpha_scorer` | Scores 4 dimensions (regime alignment, sentiment edge, fundamental quality, expected persistence) 0–2.5 each. Total 0–10 | No |
| `manager_agent` | Makes the go/no-go call. PROCEED requires alpha ≥ 6.0, fund pass, risk LOW/MED. Retry_count ≥ 2 forces hard escalation | No |
| `quant_agent` | Translates approved idea into a backtestable research brief: strategy type, signals, universe, time horizon | No |

---

## Setup

**1. Clone and install dependencies**

```bash
pip install -r requirements.txt
```

**2. Set up environment variables**

```bash
cp .env.example .env
```

Edit `.env` and add your keys:

```
OPENAI_API_KEY=your_openai_key_here
TAVILY_API_KEY=your_tavily_key_here
```

Get a free Tavily API key at [tavily.com](https://tavily.com).

**3. Run**

```bash
python main.py
```

---

## Usage

```bash
# Autonomous — pipeline finds its own catalyst and tickers
python main.py

# Scoped to an asset class
python main.py --asset_class crypto
python main.py --asset_class equities

# Guided by a theme
python main.py --asset_class equities --theme "AI infrastructure"

# Ticker-first — pipeline infers the catalyst and theme from your tickers
python main.py --asset_class mixed --tickers NVDA AMD BTC
```

Or use programmatically:

```python
from dotenv import load_dotenv
from crew import QuantCrew

load_dotenv()

crew = QuantCrew()

# Autonomous
result = crew.run()

# With context
result = crew.run(asset_class="crypto")
result = crew.run(asset_class="equities", theme="semiconductor cycle")
result = crew.run(asset_class="mixed", tickers=["NVDA", "BTC", "SOL"])
```

---

## Output

Every run saves a full JSON log to `output/run_YYYYMMDD_HHMMSS.json` containing the complete `PipelineState` — every field from every node, the decision, reasoning, and if applicable the quant brief or veto message.

Example terminal output on a successful run:

```
════════════════════════════════════════════════════════════
  MARKET AGENT — News & Catalyst Scanner
════════════════════════════════════════════════════════════
  [14:32:01] Entry mode: autonomous — scanning live news for catalyst
  [14:32:02]   Search 1: [stock market news today]
  [14:32:03]   Search 2: [semiconductor earnings movers]
  [14:32:06] Catalyst identified.
  [14:32:06]   Theme    : Nvidia earnings beat driving AI chip momentum
  [14:32:06]   Tickers  : ['NVDA', 'AMD', 'AVGO', 'TSM']
  [14:32:06]   Hypothesis: Nvidia Q4 beat -> sector repricing upward -> 1-2 week momentum

...

  DECISION  : PROCEED (HIGH)
  Alpha     : 7.4/10
  Strategy  : cross-sectional momentum
  Horizon   : 5-10 days
  Universe  : NVDA, AMD, AVGO, TSM, SMH (benchmark ETF)
  Signals   :
    · 20-day price momentum rank within semiconductor universe
    · Earnings revision direction (last 30 days)
    · Relative strength vs SPY over 10 days
```

---

## Adding a new tool

1. Create `tools/your_tool.py` with a `build_your_tool()` function that returns a LangChain tool instance.
2. Add it to `TOOL_REGISTRY` in `tools/__init__.py`:
   ```python
   from tools.your_tool import build_your_tool
   TOOL_REGISTRY = {
       "tavily_search": build_search_tool,
       "your_tool":     build_your_tool,   # add here
   }
   ```
3. Reference it by name in `config/agents.yaml` under the agent's `tools:` list.

`crew.py` resolves names to instances automatically. No other changes needed.

---

## Adding a new agent

1. Add the agent definition to `config/agents.yaml`.
2. Add the task definition to `config/tasks.yaml`.
3. Add the output schema (Pydantic `BaseModel`) to `crew.py`.
4. Add the node function to `crew.py`.
5. Wire it into the graph with `g.add_node(...)` and `g.add_edge(...)`.

---
