#!/usr/bin/env python3
"""Generate figures for the README."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "font.size": 11})

# ── Figure 1: Win-rate comparison bar chart ──────────────────────────────
labels = ["Player A\n(p=0.1)", "Player B\n(p=0.8)", "Player C\n(p=0.9)"]
x = np.arange(len(labels))
w = 0.22

rl_sparse  = [0.0553, 0.4615, 0.4832]
rl_shaped  = [0.0544, 0.4594, 0.4862]
baseline1  = [0.1108, 0.7450, 0.1441]
baseline2  = [0.1210, 0.7154, 0.1636]

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - 1.5*w, rl_sparse, w, label="Q-learn (sparse)", color="#2563eb")
ax.bar(x - 0.5*w, rl_shaped, w, label="Q-learn (shaped)", color="#7c3aed")
ax.bar(x + 0.5*w, baseline1, w, label="Shoot-strongest", color="#dc2626")
ax.bar(x + 1.5*w, baseline2, w, label="A-pass + strongest", color="#ea580c")
ax.set_ylabel("Win Rate")
ax.set_title("Turn-Based Truel: Win Rates by Strategy")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc="upper left")
ax.set_ylim(0, 0.85)
ax.grid(axis="y", alpha=0.3)
for bars in ax.containers:
    ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
fig.savefig("figures/win_rates_comparison.png")
plt.close()

# ── Figure 2: Truel game-tree diagram ───────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-1, 11)
ax.set_ylim(-1, 7)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("Three-Player Truel: Game Structure", fontsize=14, fontweight="bold")

# Players as circles
players = {"A (p=0.1)": (5, 6), "B (p=0.8)": (1.5, 2), "C (p=0.9)": (8.5, 2)}
colors = {"A (p=0.1)": "#3b82f6", "B (p=0.8)": "#ef4444", "C (p=0.9)": "#22c55e"}
for name, (cx, cy) in players.items():
    circle = plt.Circle((cx, cy), 0.8, color=colors[name], alpha=0.85, zorder=3)
    ax.add_patch(circle)
    ax.text(cx, cy, name, ha="center", va="center", fontsize=10, fontweight="bold", color="white", zorder=4)

# Arrows between players (shoot edges)
from matplotlib.patches import FancyArrowPatch
arrow_kw = dict(arrowstyle="-|>", mutation_scale=18, lw=2, zorder=2)
pairs = [("A (p=0.1)", "B (p=0.8)"), ("A (p=0.1)", "C (p=0.9)"),
         ("B (p=0.8)", "A (p=0.1)"), ("B (p=0.8)", "C (p=0.9)"),
         ("C (p=0.9)", "A (p=0.1)"), ("C (p=0.9)", "B (p=0.8)")]
offsets = [(0.15, 0), (-0.15, 0), (0.15, 0), (-0.15, 0), (0.15, 0), (-0.15, 0)]
for (src, dst), (ox, oy) in zip(pairs, offsets):
    sx, sy = players[src]
    dx, dy = players[dst]
    vec = np.array([dx-sx, dy-sy])
    norm = np.linalg.norm(vec)
    uv = vec / norm
    start = np.array([sx, sy]) + uv * 0.85 + np.array([ox, oy])
    end = np.array([dx, dy]) - uv * 0.85 + np.array([ox, oy])
    arrow = FancyArrowPatch(start, end, color=colors[src], alpha=0.5, **arrow_kw)
    ax.add_patch(arrow)

ax.text(5, 0.2, "Each player: SHOOT a rival  or  deliberately MISS\n"
        "Hit probability = player accuracy p\n"
        "Last player standing wins", ha="center", va="center",
        fontsize=10, style="italic",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f1f5f9", edgecolor="#94a3b8"))
fig.savefig("figures/truel_game_structure.png")
plt.close()

# ── Figure 3: Q-learning convergence curve ──────────────────────────────
from collections import defaultdict
import random

class TruelEnv:
    def __init__(self, accuracies, seed=0):
        self.acc = list(accuracies)
        self.rng = random.Random(seed)
        self.reset()
    def reset(self, start_turn=0):
        self.alive = [True, True, True]
        self.turn = start_turn
        return self._state()
    def _alive_mask(self):
        m = 0
        for i in range(3):
            if self.alive[i]: m |= (1 << i)
        return m
    def _state(self):
        return (self._alive_mask(), self.turn)
    def _next_alive(self, i):
        for k in range(1, 4):
            j = (i + k) % 3
            if self.alive[j]: return j
        return i
    def is_terminal(self):
        return sum(self.alive) == 1
    def winner(self):
        for i in range(3):
            if self.alive[i]: return i
        return None
    def valid_actions(self, player):
        if not self.alive[player] or self.is_terminal(): return []
        acts = [("miss", None)]
        for t in range(3):
            if t != player and self.alive[t]: acts.append(("shoot", t))
        return acts
    def step(self, action):
        if self.is_terminal(): return self._state(), True
        p = self.turn
        kind, target = action
        if kind == "shoot":
            if self.rng.random() < self.acc[p]: self.alive[target] = False
        if not self.is_terminal(): self.turn = self._next_alive(p)
        return self._state(), self.is_terminal()

class QAgent:
    def __init__(self, pid, alpha=0.2, gamma=0.98, eps_start=1.0, eps_end=0.05, eps_decay=0.99995):
        self.id = pid; self.Q = defaultdict(float)
        self.alpha = alpha; self.gamma = gamma
        self.eps = eps_start; self.eps_end = eps_end; self.eps_decay = eps_decay
    def choose_action(self, env, state):
        valid = env.valid_actions(self.id)
        if not valid: return None
        if random.random() < self.eps: return random.choice(valid)
        qs = [self.Q[(state, a)] for a in valid]
        mx = max(qs); best = [a for a, q in zip(valid, qs) if q == mx]
        return random.choice(best)
    def update(self, state, action, reward, next_state, nva):
        if action is None: return
        nq = max((self.Q[(next_state, a)] for a in nva), default=0.0)
        self.Q[(state, action)] += self.alpha * (reward + self.gamma * nq - self.Q[(state, action)])
    def decay_eps(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

random.seed(42); np.random.seed(42)
acc = (0.1, 0.8, 0.9)
env = TruelEnv(acc, seed=42)
agents = [QAgent(i) for i in range(3)]
checkpoints = list(range(1000, 200001, 1000))
history = {0: [], 1: [], 2: []}

for ep in range(200000):
    state = env.reset(start_turn=0)
    done = False
    while not done:
        turn = state[1]; agent = agents[turn]
        action = agent.choose_action(env, state)
        ns, done = env.step(action)
        r = [0.0]*3
        if done:
            w = env.winner()
            for i in range(3): r[i] = 1.0 if i == w else -1.0
        nv = env.valid_actions(turn) if (not done and env.alive[turn]) else []
        agent.update(state, action, r[turn], ns, nv)
        state = ns
    for a in agents: a.decay_eps()
    if (ep + 1) in checkpoints:
        eval_env = TruelEnv(acc, seed=7)
        wins = [0]*3
        old = [a.eps for a in agents]
        for a in agents: a.eps = 0.0
        for _ in range(5000):
            s = eval_env.reset(start_turn=0); d = False
            while not d:
                t = s[1]; act = agents[t].choose_action(eval_env, s)
                s, d = eval_env.step(act)
            wins[eval_env.winner()] += 1
        for a, e in zip(agents, old): a.eps = e
        for i in range(3): history[i].append(wins[i]/5000)

fig, ax = plt.subplots(figsize=(10, 5))
names = ["Player A (p=0.1)", "Player B (p=0.8)", "Player C (p=0.9)"]
clrs = ["#3b82f6", "#ef4444", "#22c55e"]
for i in range(3):
    ax.plot(checkpoints, history[i], label=names[i], color=clrs[i], linewidth=2)
ax.set_xlabel("Training Episodes")
ax.set_ylabel("Win Rate")
ax.set_title("Q-Learning Convergence: Win Rate During Training")
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlim(0, 200000)
ax.set_ylim(0, 0.7)
fig.savefig("figures/qlearning_convergence.png")
plt.close()

# ── Figure 4: Strategy heatmap ──────────────────────────────────────────
# Extract learned policy for the full-alive state (mask=7)
state_labels = ["A's turn\n(all alive)", "B's turn\n(all alive)", "C's turn\n(all alive)"]
# For each player at their turn in the all-alive state
action_names = ["Miss", "Shoot Left", "Shoot Right"]
policy_probs = np.zeros((3, 3))

for turn in range(3):
    state = (7, turn)
    valid = [("miss", None)]
    targets = [t for t in range(3) if t != turn]
    for t in targets:
        valid.append(("shoot", t))
    qs = [agents[turn].Q[(state, a)] for a in valid]
    mx = max(qs)
    for j, q in enumerate(qs):
        policy_probs[turn, j] = 1.0 if q == mx else 0.0
    s = policy_probs[turn].sum()
    if s > 0:
        policy_probs[turn] /= s

# Relabel columns per player
fig, ax = plt.subplots(figsize=(7, 4))
row_labels = ["A's turn", "B's turn", "C's turn"]
col_labels_per_row = [
    ["Miss", f"Shoot B", f"Shoot C"],
    ["Miss", f"Shoot A", f"Shoot C"],
    ["Miss", f"Shoot A", f"Shoot B"],
]

im = ax.imshow(policy_probs, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(row_labels)
ax.set_xticklabels(["Action 1", "Action 2", "Action 3"])
ax.set_title("Learned Greedy Policy (All Players Alive, state=7)")

for i in range(3):
    for j in range(3):
        label = col_labels_per_row[i][j]
        val = policy_probs[i, j]
        ax.text(j, i, f"{label}\n{val:.0%}", ha="center", va="center",
                fontsize=9, fontweight="bold" if val > 0.5 else "normal",
                color="white" if val > 0.5 else "black")

fig.colorbar(im, ax=ax, label="Selection Probability")
fig.savefig("figures/policy_heatmap.png")
plt.close()

print("All figures saved to figures/")
