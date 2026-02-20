# ThermoBTree — Hot/Cold B-tree Index with ML-Based Adaptive Sampling

> A self-tuning index structure that learns which keys deserve fast-path treatment — combining classical B-tree indexing with reinforcement learning to dynamically adapt to skewed workloads.

---

## Overview

Real-world database workloads are rarely uniform. A small subset of keys — the "hot" keys — typically receives a disproportionately large share of queries. **ThermoBTree** exploits this by maintaining two B-tree tiers and using machine learning to decide which keys should live in the fast tier.

The system maintains:

| Tier | Description |
|---|---|
| **Cold B-tree** | Contains *all* keys — the canonical fallback index |
| **Hot B-tree** | Contains frequently accessed keys promoted from the cold tier based on learned hit scores |

A key control parameter — the **sampling rate D** — governs the probability of promoting a key from cold to hot once it becomes sufficiently active. The core research question this project investigates is: *can the system learn the optimal D automatically, without manual tuning?*

---

## ML Adaptation Strategies

Three progressively more sophisticated approaches were implemented and evaluated for adaptive sampling rate control. All three are available as separate branches.

---

### Approach 1 — Heuristic Hill-Climbing

A simple rule-based controller that adjusts D after every fixed interval of queries based on whether the previous adjustment improved cost (measured in node visits per query).

**How it works:**
- If increasing D reduced cost → increase D again
- If increasing D raised cost → reverse direction

**Outcome:** Unstable under workload noise. The controller frequently adjusted D in unhelpful directions because it had no model of the underlying cost surface and reacted only to short-term fluctuations.

---

### Approach 2 — Online Linear Regression with SGD

A classical machine learning model that attempts to learn the relationship between sampling rate and query cost, then uses the learned model to drive D toward the optimal value.

**Model:**

```
cost_hat(D) = w₀ + w₁ · D
```

Weights `w₀` and `w₁` are updated online via stochastic gradient descent after each measurement interval. The sign of the learned slope `w₁` determines whether D is increased or decreased.

**Outcome:** The true cost surface is nonlinear and noisy, making it poorly modeled by a linear function. The system learned an incorrect relationship and converged to very small values of D, degrading performance.

---

### Approach 3 — Epsilon-Greedy Multi-Armed Bandit *(Best)*

A reinforcement learning approach that treats each candidate sampling rate as an independent "arm" of a bandit and directly estimates the reward (query cost) of each arm from observed data — without assuming any functional form.

**Candidate arms:** `D ∈ { 0.3, 0.5, 0.7, 1.0 }`

For each arm, the controller tracks the average cold-node cost observed during periods when that sampling rate was active. Using an **ε-greedy strategy**:
- With probability `ε` → **explore**: pick a random arm
- With probability `1 - ε` → **exploit**: pick the arm with the lowest observed average cost

**Outcome:** Stable under noise, makes no assumptions about the cost function, and reliably converges to a good sampling rate for the given workload distribution.

| Strategy | Stability | Accuracy | Assumptions |
|---|---|---|---|
| Hill-Climbing | ❌ Unstable | ❌ Poor | Monotone cost surface |
| Linear Regression + SGD | ⚠️ Moderate | ❌ Poor | Linear cost–D relationship |
| ε-Greedy Bandit | ✅ Stable | ✅ Good | None |

---

## Repository Structure

```
ThermoBTree/
├── Makefile                  # Build configuration
├── main.c                    # Workload generator, CLI, experiment harness
├── btree.c                   # Core in-memory B-tree implementation
├── btree.h
├── hctree.c                  # Hot/Cold index layer + ML adaptation logic
├── hctree.h
├── analyze_hctree.py         # Plotting and statistical analysis of results
└── results.csv               # Example benchmark output
```

### Module Responsibilities

| File | Responsibility |
|---|---|
| `btree.c / .h` | Core B-tree operations: insert, search, split, node management |
| `hctree.c / .h` | Hot/Cold tier logic, hit scoring, promotion policy, ML controllers |
| `main.c` | CLI argument parsing, workload generation, experiment orchestration |
| `analyze_hctree.py` | Post-run analysis: cost curves, promotion rates, D adaptation plots |
| `results.csv` | Benchmark data from sample runs |

---

## Building & Running

### Prerequisites

- GCC or Clang
- GNU Make
- Python 3 with `matplotlib` and `pandas` (for analysis only)

### Compile

```bash
make clean && make
```

### Run Modes

**Baseline HCIndex (no adaptation):**
```bash
./hctree_demo --mode hctree
```

**Fixed sampling rate:**
```bash
./hctree_demo --mode hctree --sample_init 0.5
```

**ML-adaptive sampling** *(switch branch for each approach)*:
```bash
./hctree_demo --mode hctree --sample_init 0.5 --adapt_sample
```

### Analyse Results

```bash
python analyze_hctree.py results.csv
```

Generates plots for node visit cost over time, sampling rate adaptation curves, and hot/cold tier promotion rates.

---

## Key Concepts

**Sampling Rate D**
The probability `[0.0, 1.0]` that a key crossing the promotion threshold is actually promoted to the hot tier. A value of `1.0` promotes every hot key; lower values reduce hot-tier churn at the cost of some missed promotions.

**Hit Score**
Each key in the cold tier accumulates a hit score as it is queried. Once the score exceeds a configurable threshold, the key becomes a candidate for promotion.

**Cold-Node Cost**
The primary performance metric: the number of cold B-tree node visits per query. Lower values indicate more queries are being served by the hot tier.

---

## License

For academic and educational use. See repository root for full license details.

