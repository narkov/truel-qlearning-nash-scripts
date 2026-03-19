#!/usr/bin/env python3
"""
Generate all figures for the truel paper — matching Overleaf style.
Random truel: at each step a random alive player acts.
Exact absorption probabilities via Markov-chain value iteration.
Exploitability via best-response value iteration.
"""
import os, itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

os.makedirs("figs", exist_ok=True)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 9,
})

# ─── Random Truel Environment ───────────────────────────────────────────────

def enum_states():
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
    if player not in s or is_terminal(s):
        return [0]
    acts = [0]
    for j in sorted(s):
        if j != player:
            acts.append(1 + j)
    return acts

def transition(s, player, action, acc):
    if action == 0 or is_terminal(s):
        return [(1.0, s)]
    target = action - 1
    if target not in s or target == player:
        return [(1.0, s)]
    p_hit = acc[player]
    hit_state = s - {target}
    return [(p_hit, hit_state), (1.0 - p_hit, s)]


# ─── Policies ────────────────────────────────────────────────────────────────

def policy_shoot_strongest(player, s, acc):
    opps = sorted(s - {player})
    if not opps:
        return 0
    target = max(opps, key=lambda j: acc[j])
    return 1 + target

def policy_A_pass(player, s, acc):
    if player == 0 and s == INIT_STATE:
        return 0
    return policy_shoot_strongest(player, s, acc)

def policy_nash_approx(player, s, acc):
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


# ─── Value Iteration ─────────────────────────────────────────────────────────

def compute_win_probs(acc, policy_fn):
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

def win_probs_from_start(acc, policy_fn):
    V = compute_win_probs(acc, policy_fn)
    return V[STATE_IDX[INIT_STATE]]

def best_response_value(acc, policy_fn, br_player):
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


# ─── Plotting (Overleaf-matching style) ──────────────────────────────────────

def plot_winner(grid, grid_vals, policy_name, a_val, fname):
    """Discrete viridis 3-level colormap matching Overleaf style."""
    fig, ax = plt.subplots(figsize=(4.2, 3.5))
    cmap = plt.cm.viridis
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = BoundaryNorm(bounds, cmap.N)

    im = ax.pcolormesh(grid_vals, grid_vals, grid, cmap=cmap, norm=norm,
                       shading='nearest')
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["0.00", "1.00", "2.00"])
    ax.set_xlabel("c (accuracy of C)")
    ax.set_ylabel("b (accuracy of B)")
    ax.set_title(f"Winner (policy={policy_name}, a={a_val}) (0=A,1=B,2=C)")
    ax.set_xticks(grid_vals[::2])
    ax.set_yticks(grid_vals[::2])
    fig.tight_layout()
    fig.savefig(fname)
    plt.close()

def plot_exploit(grid, grid_vals, policy_name, a_val, fname, label="Total exploitability"):
    """Continuous viridis colormap matching Overleaf style."""
    fig, ax = plt.subplots(figsize=(4.2, 3.5))
    im = ax.pcolormesh(grid_vals, grid_vals, grid, cmap='viridis',
                       shading='nearest')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(label, fontsize=8)
    ax.set_xlabel("c (accuracy of C)")
    ax.set_ylabel("b (accuracy of B)")
    ax.set_title(f"{label} (policy={policy_name}, a={a_val})")
    ax.set_xticks(grid_vals[::2])
    ax.set_yticks(grid_vals[::2])
    fig.tight_layout()
    fig.savefig(fname)
    plt.close()

def plot_exploit_per_player(grids, grid_vals, policy_name, a_val, fnames):
    """Per-player exploit as individual files matching Overleaf appendix style."""
    player_names = ["A", "B", "C"]
    for p in range(3):
        fig, ax = plt.subplots(figsize=(3.8, 3.2))
        im = ax.pcolormesh(grid_vals, grid_vals, grids[p], cmap='viridis',
                           shading='nearest')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(f"Exploit_{player_names[p]}", fontsize=8)
        ax.set_xlabel("c (accuracy of C)")
        ax.set_ylabel("b (accuracy of B)")
        ax.set_title(f"Exploitability of {player_names[p]} (policy={policy_name}, a={a_val})")
        ax.set_xticks(grid_vals[::2])
        ax.set_yticks(grid_vals[::2])
        fig.tight_layout()
        fig.savefig(fnames[p])
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

            tag = f"policy_{pol_name}_a_{a_val}"

            plot_winner(winner, grid_vals, pol_name, a_val,
                        f"figs/winner_{tag}.pdf")
            plot_exploit(exploit, grid_vals, pol_name, a_val,
                         f"figs/exploit_{tag}.pdf",
                         label="Total exploitability")

            # Per-player exploitability (only for a=0.1)
            if abs(a_val - 0.1) < 0.01:
                player_names = ["A", "B", "C"]
                fnames = [f"figs/exploit_{pn}_{tag}.pdf" for pn in player_names]
                plot_exploit_per_player(exploit_per, grid_vals, pol_name, a_val, fnames)

    print("\nAll figures saved to figs/")


if __name__ == "__main__":
    main()
