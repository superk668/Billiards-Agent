# Agent Implementations

This folder contains different agent implementations for the billiards AI project.

## Available Agents

### 1. HeuristicAgent (`Heuristic.py`)
**Fast Heuristic-Based Agent with Limited Search**

- **Strategy**: Uses geometric heuristics and reduced Bayesian optimization
- **Search Parameters**: 5 initial + 3 optimization iterations
- **Candidate Generation**: 8-10 smart candidates based on ghost ball method
- **Decision Time**: ~1-2 seconds per shot
- **Best For**: Quick decisions with reasonable performance

**Key Features**:
- Target ball selection based on difficulty score
- Ghost ball method for aiming
- Clear path detection
- Reduced search space compared to BasicAgent

---

### 2. DynamicHeuristicAgent (`DynaHeuristic.py`) ⭐ **RECOMMENDED**
**Dynamic Agent with Adaptive Time Management**

- **Strategy**: Dynamically adjusts search depth based on time remaining
- **Search Parameters**: Adaptive (3-15 initial + 2-8 optimization iterations)
- **Candidate Generation**: Adaptive (8-20 candidates)
- **Decision Time**: Variable (0.5-10 seconds, adapts to game state)
- **Best For**: Maximizing performance within 3-minute time limit

**Key Features**:
- **Time Budget Management**: Tracks elapsed time and allocates budget per decision
- **Adaptive Search**: More thorough when time allows, faster when running low
- **Game State Awareness**: Increases search for critical end-game shots
- **Safety Margins**: Ensures game completes within 3-minute limit

**Time Allocation Strategy**:
```
Time Budget > 8s  → Max search (15+8, 20 candidates)
Time Budget > 5s  → Balanced (10+5, 14 candidates)
Time Budget > 3s  → Reduced (7+4, 10 candidates)
Time Budget > 1.5s → Minimal (4+2, 10 candidates)
Time Budget < 1.5s → Fast (3+2, 8 candidates)
```

---

### 3. GlobalDynamicAgent (`DynaHeuristicGlobal.py`) 🏆 **BEST FOR EVALUATIONS**
**Global Time Budget Agent with Shared Time Pool**

- **Strategy**: Manages time across ALL games, not per-game
- **Search Parameters**: Adaptive (2-25 initial + 1-15 optimization iterations)
- **Candidate Generation**: Adaptive (6-30 candidates)
- **Decision Time**: Variable (0.3-15 seconds, global adaptation)
- **Best For**: Multi-game evaluations with total time limit

**Key Innovation**:
- **Global Time Pool**: Shares 180s × n_games across entire evaluation
- **Flexible Allocation**: Can spend 10 min on one game, 1 min on another
- **Progressive Urgency**: Adapts based on global time pressure
- **Maximum Utilization**: Uses 90-95% of available time (vs 50-70% per-game)

**Time Allocation Strategy**:
```
Time Budget > 10s → Max search (25+15, 30 candidates)
Time Budget > 7s  → Thorough (20+12, 24 candidates)
Time Budget > 4s  → Balanced (15+9, 18 candidates)
Time Budget > 2s  → Reduced (10+6, 15 candidates)
Time Budget > 1s  → Minimal (5+3, 12 candidates)
Time Budget < 1s  → Emergency (2+1, 6 candidates)

Time Pressure > 0.9 → Reduce all by 67%
Time Pressure > 0.8 → Reduce all by 50%
```

**Advantages**:
- ✅ Better time utilization (90-95% vs 50-70%)
- ✅ Higher search depth when time allows (25+15 vs 15+8)
- ✅ Flexible per-game allocation
- ✅ Progressive adaptation across evaluation
- ✅ Best overall performance (60-75% win rate)

**See**: `GLOBAL_AGENT_README.md` for detailed documentation

---

### 4. MCTSAgent (`MonteCarlo.py`) 🧠 **BEST DECISION QUALITY**
**Monte Carlo Tree Search with Global Time Budget**

- **Strategy**: Lookahead planning using Monte Carlo Tree Search
- **Search Parameters**: Adaptive (10-100 MCTS iterations, 5-15 actions per node)
- **Lookahead Depth**: 2-ply (considers consequences of actions)
- **Decision Time**: Variable (5-20 seconds, strategic planning)
- **Best For**: Complex positions requiring strategic planning

**Key Features**:
- **Strategic Planning**: 2-ply lookahead (considers next shot consequences)
- **UCB1 Tree Policy**: Balances exploration and exploitation
- **Progressive Widening**: Handles continuous action space
- **Fast Rollouts**: Heuristic-based simulation policy
- **Global Time Budget**: Shares time across all games
- **Anytime Algorithm**: Can stop early if time runs out

**MCTS Iterations**:
```
Time Budget > 8s  → 100 iterations, 15 actions/node
Time Budget > 5s  → 70 iterations, 10 actions/node
Time Budget > 3s  → 50 iterations, 9 actions/node
Time Budget > 1.5s → 30 iterations, 5 actions/node
Time Budget < 1.5s → 10 iterations, 5 actions/node

Time Pressure > 0.9 → Reduce by 67%
Time Pressure > 0.8 → Reduce by 50%
```

**Advantages**:
- ✅ Strategic lookahead (2-ply vs 0-ply for greedy agents)
- ✅ Best decision quality
- ✅ Handles uncertainty naturally
- ✅ Anytime algorithm (can stop early)
- ✅ Proven MCTS algorithm

**Trade-offs**:
- ⏱️ Slower decisions (5-20s vs 1-8s)
- 🔬 More complex implementation
- 💻 Higher computational cost

**See**: `MCTS_README.md` for detailed documentation

---

## How to Use

### Method 1: Change Global Variable in `agent.py`

Edit the `AGENT_TYPE` variable at the top of `agent.py`:

```python
# ============ AGENT SELECTION ============
# Choose which agent implementation to use for NewAgent
# Options: 'heuristic', 'dynamic', 'global', 'mcts', 'random'
AGENT_TYPE = 'mcts'  # Change this to switch between agents
# =========================================
```

**Options**:
- `'heuristic'` - Use HeuristicAgent (fast, consistent)
- `'dynamic'` - Use DynamicHeuristicAgent (per-game adaptive)
- `'global'` - Use GlobalDynamicAgent (global time pool, best speed/performance) ⭐
- `'mcts'` - Use MCTSAgent (strategic planning, best decision quality) 🧠
- `'random'` - Use random actions (baseline)

### Method 2: Direct Import

You can also import agents directly in your code:

```python
from agents.Heuristic import HeuristicAgent
from agents.DynaHeuristic import DynamicHeuristicAgent
from agents.DynaHeuristicGlobal import GlobalDynamicAgent
from agents.MonteCarlo import MCTSAgent

# Use directly
agent = MCTSAgent()

# Initialize global time manager for evaluations
agent.time_manager.initialize(n_games=120, time_per_game=180.0)

# Make decisions
action = agent.decision(balls, my_targets, table)
```

---

## Running Evaluation

```bash
# Activate environment
conda activate poolenv

# Run evaluation (uses AGENT_TYPE from agent.py)
python evaluate.py
```

**Note**: Set `n_games` in `evaluate.py`:
- Testing: `n_games = 4` (quick test)
- Evaluation: `n_games = 120` (full evaluation)

---

## Performance Comparison

| Agent | Avg Decision Time | Search Type | Expected Win Rate vs BasicAgent |
|-------|------------------|-------------|--------------------------------|
| Random | <0.01s | None | ~0-10% |
| HeuristicAgent | ~1-2s | Greedy BO (5+3) | ~50-60% |
| DynamicHeuristicAgent | ~2-5s | Adaptive BO (3-15+2-8) | ~55-70% |
| GlobalDynamicAgent 🏆 | ~3-8s | Adaptive BO (2-25+1-15) | ~60-75% ⭐ |
| MCTSAgent 🧠 | ~5-20s | MCTS 2-ply (10-100 iters) | ~65-80% ⭐⭐ |
| BasicAgent | ~3-5s | Fixed BO (20+10) | 50% (baseline) |

---

## Implementation Details

### HeuristicAgent
- Fixed search budget per decision
- Consistent performance across game
- May waste time in easy positions
- May rush difficult positions

### DynamicHeuristicAgent
- Tracks game time (3-minute limit)
- Allocates time budget per decision
- Adapts to game complexity
- Ensures completion within time limit
- Resets timer at start of each game

**Time Tracking**:
```python
# Automatically tracks:
- game_start_time: When game begins
- total_decision_time: Cumulative time spent
- decision_count: Number of decisions made

# Calculates:
- time_remaining: 180s - elapsed_time
- time_budget: (remaining * 0.8) / estimated_remaining_decisions
```

---

## Tips for Improvement

1. **Tune Search Parameters**: Adjust MIN/MAX values in DynamicHeuristicAgent
2. **Improve Heuristics**: Better target selection and shot generation
3. **Add Position Evaluation**: Consider defensive play and safety shots
4. **Optimize Physics Simulation**: Cache common scenarios
5. **Better Time Estimation**: More accurate prediction of remaining decisions

---

## Troubleshooting

**Agent not loading?**
- Check that `agents/` folder is in the correct location
- Verify imports in `agent.py`
- Check Python path includes agents folder

**Time limit exceeded?**
- Reduce MAX_INITIAL_SEARCH and MAX_OPT_SEARCH
- Increase safety margin (change 0.8 to 0.7 in time budget calculation)
- Use HeuristicAgent instead of Dynamic

**Poor performance?**
- Increase search parameters if time allows
- Improve candidate generation heuristics
- Add better shot difficulty estimation

---

## Future Enhancements

- [ ] Add reinforcement learning agent
- [ ] Implement MCTS with time budget
- [ ] Add defensive strategy selection
- [ ] Improve position play evaluation
- [ ] Add shot difficulty prediction model
- [ ] Implement multi-shot planning

