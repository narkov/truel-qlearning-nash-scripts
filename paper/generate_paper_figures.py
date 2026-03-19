#!/usr/bin/env python3
"""
Generate all figures for the truel paper.
Random truel: at each step a random alive player acts.
Exact absorption probabilities via Markov-chain value iteration.
Exploitability via best-response value iteration.
"""
import os, itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

os.makedirs("figs", exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "font.size": 10})

# ─── Random Truel Environment ───────────────────────────────────────────────

def enum_states():
    """States: frozenset of alive players. Terminal if |alive|<=1."""
    players = [0, 1, 2]
    states = []
    for r in range(1, 4):
        for combo in itertools.combinations(players, r):
            states.append(frozenset(combo))
    return states

ALL_STATES = enum_states()
STATE_IDX = {s: i for i, s in enumerate(ALL_STATES)}
N_STATES = len(ALL_STATES)
INIT_STATE = frozenset([0, 1, 2])

def is_terminal(s):
    return len(s) <= 1

def actions_for(player, s):
    """Returns list of actions: 0=abstain, 1+j=shoot player j."""
    if player not in s or is_terminal(s):
        return [0]
    acts = [0]
    for j in sorted(s):
        if j != player:
            acts.append(1 + j)
    return acts

def transition(s, player, action, acc):
    """Returns list of (prob, next_state)."""
    if action == 0 or is_terminal(s):
        return [(1.0, s)]
    target = action - 1
    if target not in s or target == player:
        return [(1.0, s)]
    p_hit = acc[player]
    hit_state = s - {target}
    return [(p_hit, hit_state), (1.0 - p_hit, s)]


# ─── Policy Definitions ─────────────────────────────────────────────────────

def policy_shoot_strongest(player, s, acc):
    """Deterministic: shoot alive opponent with highest accuracy."""
    opps = sorted(s - {player})
    if not opps:
        return 0
    target = max(opps, key=lambda j: acc[j])
    return 1 + target

def policy_A_pass(player, s, acc):
    """Player 0 abstains when all 3 alive; otherwise shoot strongest."""
    if player == 0 and s == INIT_STATE:
        return 0
    return policy_shoot_strongest(player, s, acc)

def policy_nash_approx(player, s, acc):
    """
    Simple Nash-approximation heuristic:
    - Weakest player abstains when all 3 alive
    - Otherwise shoot strongest opponent
    """
    if s == INIT_STATE:
        weakest = min(s, key=lambda j: acc[j])
        if player == weakest:
            return 0
    return policy_shoot_strongest(player, s, acc)


POLICIES = {
    "shoot_strongest": policy_shoot_strongest,
    "A_pass": policy_A_pass,
    "nash_approx": policy_nash_approx,
}


# ─── Exact Win Probabilities (Value Iteration) ──────────────────────────────

def compute_win_probs(acc, policy_fn):
    """
    Compute win probability for each player from each state.
    Random truel: uniform random turn selection among alive players.
    Returns V[state_idx, player] = win probability.
    """
    V = np.zeros((N_STATES, 3), dtype=float)

    # Terminal states
    for si, s in enumerate(ALL_STATES):
        if len(s) == 1:
            winner = list(s)[0]
            V[si, winner] = 1.0
        elif len(s) == 0:
            pass  # all zero

    for _ in range(50000):
        delta = 0.0
        for si, s in enumerate(ALL_STATES):
            if is_terminal(s):
                continue
            alive = sorted(s)
            n_alive = len(alive)
            nv = np.zeros(3, dtype=float)

            for player in alive:
                action = policy_fn(player, s, acc)
                outcomes = transition(s, player, action, acc)
                player_contrib = np.zeros(3, dtype=float)
                for prob, ns in outcomes:
                    nsi = STATE_IDX[ns]
                    player_contrib += prob * V[nsi]
                nv += player_contrib / n_alive

            delta = max(delta, np.max(np.abs(nv - V[si])))
            V[si] = nv

        if delta < 1e-12:
            break

    return V


def win_probs_from_start(acc, policy_fn):
    V = compute_win_probs(acc, policy_fn)
    return V[STATE_IDX[INIT_STATE]]


# ─── Best Response & Exploitability ─────────────────────────────────────────

def best_response_value(acc, policy_fn, br_player):
    """
    Compute BR value for br_player: they optimize, others follow policy_fn.
    Returns V[state_idx, player].
    """
    V = np.zeros((N_STATES, 3), dtype=float)

    for si, s in enumerate(ALL_STATES):
        if len(s) == 1:
            V[si, list(s)[0]] = 1.0

    for _ in range(50000):
        delta = 0.0
        for si, s in enumerate(ALL_STATES):
            if is_terminal(s):
                continue
            alive = sorted(s)
            n_alive = len(alive)
            nv = np.zeros(3, dtype=float)

            for player in alive:
                if player == br_player:
                    # maximize over actions
                    best_val = -1.0
                    best_contrib = np.zeros(3)
                    for action in actions_for(player, s):
                        outcomes = transition(s, player, action, acc)
                        contrib = np.zeros(3, dtype=float)
                        for prob, ns in outcomes:
                            contrib += prob * V[STATE_IDX[ns]]
                        if contrib[br_player] > best_val + 1e-15:
                            best_val = contrib[br_player]
                            best_contrib = contrib
                    nv += best_contrib / n_alive
                else:
                    action = policy_fn(player, s, acc)
                    outcomes = transition(s, player, action, acc)
                    contrib = np.zeros(3, dtype=float)
                    for prob, ns in outcomes:
                        contrib += prob * V[STATE_IDX[ns]]
                    nv += contrib / n_alive

            delta = max(delta, np.max(np.abs(nv - V[si])))
            V[si] = nv

        if delta < 1e-12:
            break

    return V


def compute_exploitability(acc, policy_fn):
    """Returns (total_exploit, [delta_A, delta_B, delta_C])."""
    base_V = compute_win_probs(acc, policy_fn)
    base_u = base_V[STATE_IDX[INIT_STATE]]

    deltas = []
    for i in range(3):
        br_V = best_response_value(acc, policy_fn, i)
        br_u = br_V[STATE_IDX[INIT_STATE]]
        deltas.append(max(0.0, br_u[i] - base_u[i]))

    return sum(deltas), deltas


# ─── Grid Scan ──────────────────────────────────────────────────────────────

def grid_scan(policy_fn, a_fixed, grid_vals):
    """Scan (b, c) grid at fixed a. Returns winner_grid, exploit_grid, per_player_exploit."""
    n = len(grid_vals)
    winner_grid = np.full((n, n), np.nan)
    exploit_grid = np.full((n, n), np.nan)
    exploit_per = [np.full((n, n), np.nan) for _ in range(3)]

    for bi, b in enumerate(grid_vals):
        for ci, c in enumerate(grid_vals):
            acc = [a_fixed, b, c]
            wp = win_probs_from_start(acc, policy_fn)
            winner_grid[bi, ci] = np.argmax(wp)

            total_ex, deltas = compute_exploitability(acc, policy_fn)
            exploit_grid[bi, ci] = total_ex
            for p in range(3):
                exploit_per[p][bi, ci] = deltas[p]

    return winner_grid, exploit_grid, exploit_per


# ─── Plotting ───────────────────────────────────────────────────────────────

WINNER_CMAP = ListedColormap(["#3b82f6", "#ef4444", "#22c55e"])

def plot_winner(grid, grid_vals, title, fname):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(grid, origin='lower', aspect='auto',
                   extent=[grid_vals[0], grid_vals[-1], grid_vals[0], grid_vals[-1]],
                   cmap=WINNER_CMAP, vmin=0, vmax=2)
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["A", "B", "C"])
    ax.set_xlabel("c (Player C accuracy)")
    ax.set_ylabel("b (Player B accuracy)")
    ax.set_title(title)
    fig.savefig(fname)
    plt.close()

def plot_exploit(grid, grid_vals, title, fname, vmax=None):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(grid, origin='lower', aspect='auto',
                   extent=[grid_vals[0], grid_vals[-1], grid_vals[0], grid_vals[-1]],
                   cmap='YlOrRd', vmin=0, vmax=vmax)
    fig.colorbar(im, ax=ax, label="Exploitability")
    ax.set_xlabel("c (Player C accuracy)")
    ax.set_ylabel("b (Player B accuracy)")
    ax.set_title(title)
    fig.savefig(fname)
    plt.close()


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    grid_vals = np.arange(0.1, 1.0, 0.1)  # 0.1 .. 0.9
    a_slices = [0.1, 0.5, 0.9]

    for pol_name, pol_fn in POLICIES.items():
        print(f"\n=== Policy: {pol_name} ===")
        for a_val in a_slices:
            print(f"  a = {a_val} ...")
            winner, exploit, exploit_per = grid_scan(pol_fn, a_val, grid_vals)

            a_str = f"{a_val:.1f}".replace('.', 'p' if False else '.')
            tag = f"policy_{pol_name}_a_{a_val}"

            plot_winner(winner, grid_vals, f"Winner: {pol_name}, a={a_val}",
                        f"figs/winner_{tag}.pdf")
            plot_exploit(exploit, grid_vals, f"Exploitability: {pol_name}, a={a_val}",
                         f"figs/exploit_{tag}.pdf")

            # Per-player exploitability (only for a=0.1)
            if abs(a_val - 0.1) < 0.01:
                for p, pname in enumerate(["A", "B", "C"]):
                    plot_exploit(exploit_per[p], grid_vals,
                                 f"Exploit {pname}: {pol_name}, a={a_val}",
                                 f"figs/exploit_{pname}_{tag}.pdf",
                                 vmax=None)

    print("\nAll figures saved to figs/")


if __name__ == "__main__":
    main()
