# Candy Crush RL Agent — Project Q&A

---

## Slide 1: Problem Details

We aim to build a **Reinforcement Learning agent** that learns to play a **Candy Crush-style match-three puzzle game** optimally.

### The Game
- An **N×N grid** (e.g., 8×8) is filled with **colored tiles** (e.g., 6 different candy types).
- The player **swaps two adjacent tiles** (horizontally or vertically) to create a line of **3 or more matching tiles**.
- Matched tiles are **cleared**, tiles above **fall due to gravity**, and new random tiles fill in from the top.
- After tiles fall, **chain reactions (cascades)** can occur — new matches form automatically, yielding bonus points without consuming a move.
- The player has a **limited number of moves** (e.g., 30) to reach a **target score** or complete level-specific objectives.

### The Challenge
- The **state space is enormous**: an 8×8 grid with 6 colors has ~6⁶⁴ ≈ 10⁴⁹ possible configurations.
- Rewards are **delayed and stochastic** — a single swap can trigger unpredictable cascade chains worth far more than the initial match.
- The agent must learn **long-term strategic play**: sometimes a lower-scoring immediate move sets up a massive cascade later.
- **Greedy strategies fail** because they optimize for the current move without considering future board states.

---

## Slide 2: Why This Is a Good RL Problem & State/Action Space

### Why This Is a Strong RL Problem

| RL Property | How It Appears in Candy Crush |
|---|---|
| **Sequential decision-making** | Each swap changes the board; the agent must plan across 30 moves, not just one. |
| **Delayed rewards** | A swap may clear 3 tiles immediately but trigger a 5-level cascade worth 10× more — the agent must learn to value these setups. |
| **Stochastic transitions** | After tiles are cleared, **new tiles are randomly generated**, making the next state unpredictable. No two games play the same. |
| **Large state space** | The combinatorial grid makes tabular methods infeasible, requiring **function approximation** (neural networks). |
| **Sparse high-value events** | Cascades and special-tile combos are rare but extremely rewarding — the agent must explore to discover these strategies. |
| **Credit assignment problem** | Was the big score from the last swap, or from a setup move 5 turns ago? RL algorithms must learn this attribution. |
| **Finite horizon** | A fixed move budget creates natural episodes, fitting the episodic RL framework perfectly. |

### State Space

- **Representation**: The board is an `N × N` integer matrix where each cell holds a value in `{0, 1, ..., C-1}` representing the candy color.
- **Observation shape**: `(N, N)` — e.g., `(8, 8)` with values `0–5` for 6 colors.
- **Encoding**: Can be fed as a **flattened vector** (for MLP) or as a **2D grid / one-hot encoded channels** (for CNN, shape `(C, N, N)`).
- Additional features (optional): remaining moves, current score, move count.

### Action Space

- **Type**: `Discrete(2 × N × (N - 1))`
  - **Horizontal swaps**: `N` rows × `(N-1)` swap positions per row = `N(N-1)` actions
  - **Vertical swaps**: `(N-1)` swap positions per column × `N` columns = `N(N-1)` actions
  - **Total for 8×8 grid**: `2 × 8 × 7 = 112 possible actions`
- **Action decoding**: Each action index maps to a specific `(row, col, direction)` swap.
- **Invalid actions**: Swaps that produce no match are penalized (`-1` reward) and the board is reverted.

---

## Slide 3: How We Plan to Validate Results

### Algorithms Compared
We train and compare **three approaches** on identical seeds and conditions:
1. **PPO (Baseline)** — Proximal Policy Optimization with clipped surrogate objective
2. **GRPO** — Group Relative Policy Optimization (advantages computed relative to group trajectory means)
3. **PPO Variants** — Entropy-annealed PPO + Reward-normalized PPO

### Evaluation Metrics (over 10,000 episodes)

| Metric | What It Measures |
|---|---|
| **Average Final Score** | Overall agent performance across random seeds |
| **Score Per Move** | Move efficiency — how well the agent uses its limited budget |
| **Success Rate** | % of episodes where the agent reaches the score target |
| **Convergence Speed** | How many training episodes to reach stable performance |
| **Training Variance** | Stability of learning curves (lower = more reliable) |
| **Generalization** | Performance on unseen board layouts and larger grid sizes (e.g., train on 6×6, test on 8×8) |

### Validation Strategy

1. **Training Curves**: Plot reward vs. episode for all three agents. Expect upward trend, confirming the agent is learning (not random).
2. **Statistical Comparison**: Run each agent with **5 random seeds**, report **mean ± std** of all metrics. Use this to confirm results are reproducible.
3. **Baseline Comparison**: Compare all agents against a **random agent** and a **greedy heuristic** (always picks the highest immediate-reward swap). The RL agents should significantly outperform both.
4. **Cascade Analysis**: Track average cascade depth per episode over training. A good agent should learn to trigger deeper cascades over time.
5. **Ablation Studies**: Disable individual features (entropy annealing, reward normalization, group advantages) to measure each component's contribution.
6. **Generalization Test**: Train on `6×6` grid, evaluate on `8×8` and `10×10` to test whether learned strategies transfer to unseen board sizes.

### Tools
- **TensorBoard** for real-time training visualization
- **Matplotlib** for publication-quality comparison plots
- All experiments run on **standard consumer laptops** (CPU-only) to demonstrate feasibility
