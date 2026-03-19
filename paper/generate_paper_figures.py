#!/usr/bin/env python3
"""
Generate all figures for the truel paper — matching Overleaf style.
Turn-based cyclic truel: players act in order A->B->C->A->... (skipping dead).
State = (alive_mask, turn). Exact via DP value iteration.
"""
import os
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

# ─── Turn-based Truel Environment ────────────────────────────────────────────

def is_alive(mask, p):
    return (mask >> p) & 1

def popcount(mask):
    return bin(mask).count('1')

def kill(mask, p):
    return mask & ~(1 << p)

def next_turn(mask, cur, n=3):
    for k in range(1, n + 1):
        c = (cur + k) % n
        if is_alive(mask, c):
            return c
    return cur

def enum_states(n=3):
    """State = (alive_mask, turn) where turn is alive."""
    states = []
    for mask in range(1, 1 << n):
        for turn in range(n):
            if is_alive(mask, turn):
                states.append((mask, turn))
    return states

ALL_STATES = enum_states()
STATE_IDX = {s: i for i, s in enumerate(ALL_STATES)}
N_STATES = len(ALL_STATES)

def init_state(turn=0):
    return (0b111, turn)

def is_terminal(mask):
    return popcount(mask) <= 1

def winner(mask):
    if popcount(mask) != 1:
        return None
    for p in range(3):
        if is_alive(mask, p):
            return p
    return None

def legal_actions(mask, turn):
    if is_terminal(mask):
        return [0]
    acts = [0]  # abstain
    for t in range(3):
        if t != turn and is_alive(mask, t):
            acts.append(1 + t)
    return acts

def transition(mask, turn, action, acc):
    """Returns list of (prob, (new_mask, new_turn))."""
    if is_terminal(mask):
        return [(1.0, (mask, turn))]
    if action == 0:
        nt = next_turn(mask, turn)
        return [(1.0, (mask, nt))]
    target = action - 1
    if not is_alive(mask, target) or target == turn:
        nt = next_turn(mask, turn)
        return [(1.0, (mask, nt))]
    p_hit = acc[turn]
    hit_mask = kill(mask, target)
    outcomes = []
    # Hit
    if is_terminal(hit_mask):
        outcomes.append((p_hit, (hit_mask, turn)))
    else:
        nt = next_turn(hit_mask, turn)
        outcomes.append((p_hit, (hit_mask, nt)))
    # Miss
    nt = next_turn(mask, turn)
    outcomes.append((1.0 - p_hit, (mask, nt)))
    return outcomes


# ─── Policies ────────────────────────────────────────────────────────────────

def policy_shoot_strongest(turn, mask, acc):
    opps = [p for p in range(3) if p != turn and is_alive(mask, p)]
    if not opps:
        return 0
    target = max(opps, key=lambda p: acc[p])
    return 1 + target

def policy_A_pass(turn, mask, acc):
    if turn == 0 and mask == 0b111:
        return 0
    return policy_shoot_strongest(turn, mask, acc)

def policy_nash_approx(turn, mask, acc):
    if mask == 0b111:
        weakest = min([p for p in range(3) if is_alive(mask, p)], key=lambda p: acc[p])
        if turn == weakest:
            return 0
    return policy_shoot_strongest(turn, mask, acc)

POLICIES = {
    "shoot_strongest": policy_shoot_strongest,
    "A_pass": policy_A_pass,
    "nash_approx": policy_nash_approx,
}


# ─── Value Iteration ─────────────────────────────────────────────────────────

def compute_win_probs(acc, policy_fn, start_turn=0):
    """Returns V[state_idx, player]."""
    V = np.zeros((N_STATES, 3), dtype=float)
    for si, (mask, turn) in enumerate(ALL_STATES):
        if is_terminal(mask):
            w = winner(mask)
            if w is not None:
                V[si, w] = 1.0

    for _ in range(100000):
        delta = 0.0
        for si, (mask, turn) in enumerate(ALL_STATES):
            if is_terminal(mask):
                continue
            action = policy_fn(turn, mask, acc)
            nv = np.zeros(3, dtype=float)
            for prob, ns in transition(mask, turn, action, acc):
                nsi = STATE_IDX[ns]
                nv += prob * V[nsi]
            delta = max(delta, np.max(np.abs(nv - V[si])))
            V[si] = nv
        if delta < 1e-13:
            break
    return V

def best_response_value(acc, policy_fn, br_player, start_turn=0):
    """BR for br_player, others follow policy_fn."""
    V = np.zeros((N_STATES, 3), dtype=float)
    for si, (mask, turn) in enumerate(ALL_STATES):
        if is_terminal(mask):
            w = winner(mask)
            if w is not None:
                V[si, w] = 1.0

    for _ in range(100000):
        delta = 0.0
        for si, (mask, turn) in enumerate(ALL_STATES):
            if is_terminal(mask):
                continue
            if turn == br_player:
                best_nv = None
                best_val = -1e30
                for a in legal_actions(mask, turn):
                    exp = np.zeros(3, dtype=float)
                    for prob, ns in transition(mask, turn, a, acc):
                        exp += prob * V[STATE_IDX[ns]]
                    if exp[br_player] > best_val + 1e-15:
                        best_val = exp[br_player]
                        best_nv = exp
                nv = best_nv
            else:
                action = policy_fn(turn, mask, acc)
                nv = np.zeros(3, dtype=float)
                for prob, ns in transition(mask, turn, action, acc):
                    nv += prob * V[STATE_IDX[ns]]
            delta = max(delta, np.max(np.abs(nv - V[si])))
            V[si] = nv
        if delta < 1e-13:
            break
    return V

def compute_exploitability(acc, policy_fn):
    s0 = init_state(0)
    s0_idx = STATE_IDX[s0]
    base_V = compute_win_probs(acc, policy_fn)
    base_u = base_V[s0_idx]
    deltas = []
    for i in range(3):
        br_V = best_response_value(acc, policy_fn, i)
        br_u = br_V[s0_idx]
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
            acc = [a_fixed, float(b), float(c)]
            V = compute_win_probs(acc, policy_fn)
            s0 = init_state(0)
            wp = V[STATE_IDX[s0]]
            winner_grid[bi, ci] = np.argmax(wp)

            total_ex, deltas = compute_exploitability(acc, policy_fn)
            exploit_grid[bi, ci] = total_ex
            for p in range(3):
                exploit_per[p][bi, ci] = deltas[p]

    return winner_grid, exploit_grid, exploit_per


# ─── Plotting ───────────────────────────────────────────────────────────────

def plot_winner(grid, grid_vals, policy_name, a_val, fname):
    fig, ax = plt.subplots(figsize=(4.2, 3.5))
    cmap = plt.cm.viridis
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = BoundaryNorm(bounds, cmap.N)
    im = ax.pcolormesh(grid_vals, grid_vals, grid, cmap=cmap, norm=norm, shading='nearest')
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
    fig, ax = plt.subplots(figsize=(4.2, 3.5))
    im = ax.pcolormesh(grid_vals, grid_vals, grid, cmap='viridis', shading='nearest', vmin=0)
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


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    grid_vals = np.arange(0.1, 1.0, 0.05)  # finer grid: 0.10, 0.15, ..., 0.95
    a_slices = [0.1, 0.5, 0.9]

    for pol_name, pol_fn in POLICIES.items():
        print(f"\n=== Policy: {pol_name} ===")
        for a_val in a_slices:
            print(f"  a = {a_val} ...", end=" ", flush=True)
            winner, exploit, exploit_per = grid_scan(pol_fn, a_val, grid_vals)
            tag = f"policy_{pol_name}_a_{a_val}"
            plot_winner(winner, grid_vals, pol_name, a_val, f"figs/winner_{tag}.pdf")
            plot_exploit(exploit, grid_vals, pol_name, a_val, f"figs/exploit_{tag}.pdf")

            if abs(a_val - 0.1) < 0.01:
                for p, pn in enumerate(["A", "B", "C"]):
                    plot_exploit(exploit_per[p], grid_vals, pol_name, a_val,
                                 f"figs/exploit_{pn}_{tag}.pdf",
                                 label=f"Exploitability of {pn}")
            print("done")

    print("\nAll figures saved to figs/")


if __name__ == "__main__":
    main()
