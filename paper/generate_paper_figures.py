#!/usr/bin/env python3
"""
Generate all figures for the truel paper.

This script matches the model described in main.tex:
- random turn selection among currently alive players
- state = alive_mask only
- exact policy evaluation via linear solves
- exact one-player best responses via deterministic policy enumeration
"""
import itertools
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm


os.makedirs("figs", exist_ok=True)
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 9,
    }
)


def is_alive(mask, p):
    return (mask >> p) & 1


def popcount(mask):
    return bin(mask).count("1")


def kill(mask, p):
    return mask & ~(1 << p)


def alive_players(mask):
    return [p for p in range(3) if is_alive(mask, p)]


def is_terminal(mask):
    return popcount(mask) <= 1


def winner(mask):
    if popcount(mask) != 1:
        return None
    for p in range(3):
        if is_alive(mask, p):
            return p
    return None


ALL_STATES = [mask for mask in range(1, 1 << 3)]
STATE_IDX = {mask: i for i, mask in enumerate(ALL_STATES)}
NONTERMINAL_STATES = [mask for mask in ALL_STATES if not is_terminal(mask)]
TERMINAL_STATES = [mask for mask in ALL_STATES if is_terminal(mask)]


def init_state():
    return 0b111


def legal_actions(mask, actor):
    if is_terminal(mask) or not is_alive(mask, actor):
        return [0]
    acts = [0]
    for target in alive_players(mask):
        if target != actor:
            acts.append(1 + target)
    return acts


def action_transition(mask, actor, action, acc):
    """
    Return a list of (prob, next_mask) for a chosen actor/action.
    """
    if is_terminal(mask) or not is_alive(mask, actor):
        return [(1.0, mask)]

    if action == 0:
        return [(1.0, mask)]

    target = action - 1
    if target == actor or not is_alive(mask, target):
        return [(1.0, mask)]

    p_hit = float(acc[actor])
    hit_mask = kill(mask, target)
    return [(p_hit, hit_mask), (1.0 - p_hit, mask)]


def random_turn_transition(mask, actions_by_actor, acc):
    """
    Return a list of (prob, next_mask) after uniformly sampling the acting player.
    """
    if is_terminal(mask):
        return [(1.0, mask)]

    probs = {}
    actors = alive_players(mask)
    actor_weight = 1.0 / len(actors)
    for actor in actors:
        action = actions_by_actor[actor]
        for prob, next_mask in action_transition(mask, actor, action, acc):
            probs[next_mask] = probs.get(next_mask, 0.0) + actor_weight * prob
    return list(probs.items())


def policy_shoot_strongest(actor, mask, acc):
    opps = [p for p in alive_players(mask) if p != actor]
    if not opps:
        return 0
    target = max(opps, key=lambda p: (acc[p], -p))
    return 1 + target


def policy_A_pass(actor, mask, acc):
    if actor == 0 and mask == 0b111:
        return 0
    return policy_shoot_strongest(actor, mask, acc)


def policy_nash_approx(actor, mask, acc):
    if mask == 0b111:
        weakest = min(alive_players(mask), key=lambda p: (acc[p], p))
        if actor == weakest:
            return 0
    return policy_shoot_strongest(actor, mask, acc)


POLICIES = {
    "shoot_strongest": policy_shoot_strongest,
    "A_pass": policy_A_pass,
    "nash_approx": policy_nash_approx,
}


def actions_for_profile(mask, acc, policy_fn):
    return {actor: policy_fn(actor, mask, acc) for actor in alive_players(mask)}


def evaluate_transition_kernel(acc, action_selector):
    """
    Build the exact transition kernel under a fixed policy profile.

    action_selector(mask) must return {actor: action}.
    """
    n_states = len(ALL_STATES)
    P = np.zeros((n_states, n_states), dtype=float)
    for mask in ALL_STATES:
        row = STATE_IDX[mask]
        if is_terminal(mask):
            P[row, row] = 1.0
            continue
        for next_mask, prob in random_turn_transition(mask, action_selector(mask), acc):
            P[row, STATE_IDX[next_mask]] += prob
    return P


def solve_win_probs_from_kernel(P):
    """
    Solve exact absorption probabilities for each winner.
    """
    nt_idx = [STATE_IDX[mask] for mask in NONTERMINAL_STATES]
    t_idx = [STATE_IDX[mask] for mask in TERMINAL_STATES]

    Q = P[np.ix_(nt_idx, nt_idx)]
    R = P[np.ix_(nt_idx, t_idx)]
    I = np.eye(len(nt_idx))

    terminal_payoffs = np.zeros((len(t_idx), 3), dtype=float)
    for row, mask in enumerate(TERMINAL_STATES):
        terminal_payoffs[row, winner(mask)] = 1.0

    try:
        nonterminal_values = np.linalg.solve(I - Q, R @ terminal_payoffs)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Policy profile does not induce a proper absorbing Markov chain."
        ) from exc

    V = np.zeros((len(ALL_STATES), 3), dtype=float)
    for row, mask in enumerate(NONTERMINAL_STATES):
        V[STATE_IDX[mask]] = nonterminal_values[row]
    for row, mask in enumerate(TERMINAL_STATES):
        V[STATE_IDX[mask]] = terminal_payoffs[row]
    return V


def compute_win_probs(acc, policy_fn):
    P = evaluate_transition_kernel(
        acc, lambda mask: actions_for_profile(mask, acc, policy_fn)
    )
    return solve_win_probs_from_kernel(P)


def best_response_value(acc, policy_fn, br_player):
    """
    Compute an exact pure best response for br_player.

    Because the state space is tiny, enumerating deterministic policies is
    simpler and more reliable than undiscounted value iteration with self-loops.
    """
    decision_states = [
        mask for mask in NONTERMINAL_STATES if is_alive(mask, br_player)
    ]
    legal_by_state = {
        mask: legal_actions(mask, br_player) for mask in decision_states
    }

    best_V = None
    best_payoff = -np.inf
    for action_tuple in itertools.product(
        *(legal_by_state[mask] for mask in decision_states)
    ):
        br_actions = dict(zip(decision_states, action_tuple))

        def selector(mask):
            actions = {}
            for actor in alive_players(mask):
                if actor == br_player:
                    actions[actor] = br_actions[mask]
                else:
                    actions[actor] = policy_fn(actor, mask, acc)
            return actions

        P = evaluate_transition_kernel(acc, selector)
        try:
            V = solve_win_probs_from_kernel(P)
        except ValueError:
            continue
        payoff = V[STATE_IDX[init_state()], br_player]
        if payoff > best_payoff + 1e-15:
            best_payoff = payoff
            best_V = V

    if best_V is None:
        raise ValueError("No proper absorbing best-response policy was found.")

    return best_V


NOISE_THRESHOLD = 1e-10


def compute_exploitability(acc, policy_fn):
    s0_idx = STATE_IDX[init_state()]
    base_V = compute_win_probs(acc, policy_fn)
    base_u = base_V[s0_idx]
    deltas = []
    for i in range(3):
        br_V = best_response_value(acc, policy_fn, i)
        br_u = br_V[s0_idx]
        raw = br_u[i] - base_u[i]
        # Clamp: exploitability is non-negative by definition;
        # zero out numerical noise below threshold.
        d = max(0.0, raw)
        if d < NOISE_THRESHOLD:
            d = 0.0
        deltas.append(d)
    return float(sum(deltas)), deltas


def grid_scan(policy_fn, a_fixed, grid_vals):
    n = len(grid_vals)
    winner_grid = np.full((n, n), np.nan)
    exploit_grid = np.full((n, n), np.nan)
    exploit_per = [np.full((n, n), np.nan) for _ in range(3)]

    for bi, b in enumerate(grid_vals):
        for ci, c in enumerate(grid_vals):
            acc = [float(a_fixed), float(b), float(c)]
            V = compute_win_probs(acc, policy_fn)
            wp = V[STATE_IDX[init_state()]]
            winner_grid[bi, ci] = np.argmax(wp)

            total_ex, deltas = compute_exploitability(acc, policy_fn)
            exploit_grid[bi, ci] = total_ex
            for p in range(3):
                exploit_per[p][bi, ci] = deltas[p]

    return winner_grid, exploit_grid, exploit_per


def plot_winner(grid, grid_vals, policy_name, a_val, fname):
    fig, ax = plt.subplots(figsize=(4.2, 3.5))
    cmap = plt.cm.viridis
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = BoundaryNorm(bounds, cmap.N)
    im = ax.pcolormesh(
        grid_vals, grid_vals, grid, cmap=cmap, norm=norm, shading="nearest"
    )
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
    grid_max = float(np.nanmax(grid))
    if grid_max < NOISE_THRESHOLD:
        # Grid is effectively zero everywhere — show uniform color + annotation
        im = ax.pcolormesh(grid_vals, grid_vals, np.zeros_like(grid),
                           cmap="viridis", shading="nearest", vmin=0, vmax=1)
        ax.text(0.5, 0.5, "≈ 0 everywhere\n(within numerical precision)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.7))
    else:
        im = ax.pcolormesh(grid_vals, grid_vals, grid, cmap="viridis",
                           shading="nearest", vmin=0)
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


def main():
    grid_vals = np.arange(0.1, 1.0, 0.05)
    a_slices = [0.1, 0.5, 0.9]

    for pol_name, pol_fn in POLICIES.items():
        print(f"\n=== Policy: {pol_name} ===")
        for a_val in a_slices:
            print(f"  a = {a_val} ...", end=" ", flush=True)
            winner, exploit, exploit_per = grid_scan(pol_fn, a_val, grid_vals)
            tag = f"policy_{pol_name}_a_{a_val}"
            plot_winner(winner, grid_vals, pol_name, a_val, f"figs/winner_{tag}.pdf")
            plot_exploit(exploit, grid_vals, pol_name, a_val, f"figs/exploit_{tag}.pdf")

            # Sanity check: verify total == sum of per-player
            diff = np.abs(exploit - sum(exploit_per[p] for p in range(3)))
            max_diff = np.nanmax(diff)
            if max_diff > NOISE_THRESHOLD:
                print(f"WARNING: total != sum(per-player), max diff = {max_diff:.2e}")

            # Report stats
            ex_max = np.nanmax(exploit)
            print(f"exploit max={ex_max:.6f}", end=" ")

            if abs(a_val - 0.1) < 0.01:
                for p, pn in enumerate(["A", "B", "C"]):
                    plot_exploit(
                        exploit_per[p],
                        grid_vals,
                        pol_name,
                        a_val,
                        f"figs/exploit_{pn}_{tag}.pdf",
                        label=f"Exploitability of {pn}",
                    )
                    pp_max = np.nanmax(exploit_per[p])
                    pp_min = np.nanmin(exploit_per[p])
                    print(f"[{pn}: {pp_min:.2e}..{pp_max:.2e}]", end=" ")
            print("done")

    print("\nAll figures saved to figs/")


if __name__ == "__main__":
    main()
