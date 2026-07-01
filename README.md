✂️ Cutting Stock Solver using Multi-Agent Meta-Heuristics

This repository features a Python-based optimization tool designed to solve the Cutting Stock Problem (CSP), a well-known NP-hard combinatorial optimization problem widely encountered in industries such as steel, paper, and textiles to minimize material waste.

The solver integrates three powerful meta-heuristic algorithms. Hill Climbing (HC), Simulated Annealing (SA), and Genetic Algorithm (GA) and orchestrates them using single-agent, collaborative, and hyper-metaheuristic multi-agent architectures.

---

🚀 Features

* 3 Meta-Heuristic Engines: Built-in support for Hill Climbing, Simulated Annealing, and Genetic Algorithms.
* Multi-Agent Collaboration: Supports N parallel agents executing independent searches to avoid local optima.
* Hyper-Metaheuristic Approach: Dynamically assigns different algorithms (HC ➡️ SA ➡️ GA) across the agent pool to leverage diverse search behaviors concurrently.
* Advanced Visualizations (Matplotlib & Pandas):
    * Detailed visual scheme of the optimal cutting pattern (including hatched blocks for waste).
    * Waste comparison bar charts across different agents.
    * Time-Efficiency scatter plots.
    * Algorithm performance distribution (Boxplots) with automated mean tracking.

---

🧠 Optimization Architectures

1. Single Algorithm
A single agent executes the chosen meta-heuristic algorithm (HC, SA, or GA) for a specified number of iterations to find a valid solution.

2. Collaborative Agent-Based Approach
A pool of N agents runs the same chosen algorithm independently with randomized starting points. The system aggregates all results and selects the global best solution based on minimum waste and minimum cuts.

3. Hyper Meta-Heuristic Approach
Agents are initialized with alternating algorithms (Agent 1: HC, Agent 2: SA, Agent 3: GA, etc.). This hybrid structure allows algorithms with different exploration/exploitation strengths to cooperatively find the absolute best pattern.

---

🛠️Simulation Workflow

  1.Stock Roll Length: Enter the total length of your raw material/stock roll (e.g., 123.45).

  2.Piece Lengths: Enter the desired lengths to cut one by one. Enter 0 when you are done (e.g., 12.5, 30.2, 15.0, 0).

  3.Number of Agents: Specify how many agents to deploy (1–10).

  4.Optimization Method: Choose between 1 (Single), 2 (Collaborative), or 3 (Hyper-Metaheuristic).
