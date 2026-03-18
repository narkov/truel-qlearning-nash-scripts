#!/usr/bin/env python3
"""
Turn-based truel + Q-learning with shaped reward (per-step cost) + baseline comparisons.

Usage:
  python 02_truel_turn_based_qlearning_shaped.py
"""
import random
from collections import defaultdict
import numpy as np

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
            if self.alive[i]:
                m |= (1 << i)
        return m

    def _state(self):
        return (self._alive_mask(), self.turn)

    def _next_alive(self, i):
        for k in range(1, 4):
            j = (i + k) % 3
            if self.alive[j]:
                return j
        return i

    def is_terminal(self):
        return sum(self.alive) == 1

    def winner(self):
        for i in range(3):
            if self.alive[i]:
                return i
        return None

    def valid_actions(self, player):
        if (not self.alive[player]) or self.is_terminal():
            return []
        acts = [("miss", None)]
        for t in range(3):
            if t != player and self.alive[t]:
                acts.append(("shoot", t))
        return acts

    def step(self, action):
        if self.is_terminal():
            return self._state(), True
        p = self.turn
        kind, target = action
        if kind == "shoot":
            hit = (self.rng.random() < self.acc[p])
            if hit:
                self.alive[target] = False
        elif kind == "miss":
            pass
        else:
            raise ValueError(action)

        if not self.is_terminal():
            self.turn = self._next_alive(p)
        return self._state(), self.is_terminal()

class QAgent:
    def __init__(self, player_id, alpha=0.25, gamma=0.98, eps_start=1.0, eps_end=0.05, eps_decay=0.99995):
        self.id = player_id
        self.Q = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

    def choose_action(self, env, state):
        valid = env.valid_actions(self.id)
        if not valid:
            return None
        if random.random() < self.eps:
            return random.choice(valid)
        qs = [self.Q[(state, a)] for a in valid]
        mx = max(qs)
        best = [a for a, q in zip(valid, qs) if q == mx]
        return random.choice(best)

    def update(self, state, action, reward, next_state, next_valid_actions):
        if action is None:
            return
        key = (state, action)
        next_q = max((self.Q[(next_state, a)] for a in next_valid_actions), default=0.0)
        td_target = reward + self.gamma * next_q
        self.Q[key] += self.alpha * (td_target - self.Q[key])

    def decay_eps(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

def act_shoot_strongest(env, player_id):
    valid = env.valid_actions(player_id)
    if not valid:
        return None
    shoot = [a for a in valid if a[0] == "shoot"]
    if not shoot:
        return ("miss", None)
    return max(shoot, key=lambda a: env.acc[a[1]])

def act_A_pass_when_3_alive_else_shoot_strongest(env, player_id):
    if player_id == 0 and sum(env.alive) == 3:
        return ("miss", None)
    return act_shoot_strongest(env, player_id)

def eval_baseline(policy_fn, accuracies, games=50_000, seed=101, start_turn=0):
    env = TruelEnv(accuracies, seed=seed)
    wins = [0, 0, 0]
    for _ in range(games):
        env.reset(start_turn=start_turn)
        done = False
        while not done:
            p = env.turn
            action = policy_fn(env, p)
            _, done = env.step(action)
        wins[env.winner()] += 1
    return [w / games for w in wins]

def train_shaped(accuracies=(0.1, 0.8, 0.9), episodes=150_000, seed=42, start_turn=0, step_cost=0.002, max_steps=200):
    random.seed(seed)
    np.random.seed(seed)
    env = TruelEnv(accuracies, seed=seed)
    agents = [QAgent(i) for i in range(3)]

    for ep in range(episodes):
        state = env.reset(start_turn=start_turn)
        done = False
        steps = 0
        while not done and steps < max_steps:
            turn = state[1]
            agent = agents[turn]
            action = agent.choose_action(env, state)
            next_state, done = env.step(action)

            reward = -step_cost
            if done:
                w = env.winner()
                reward += (1.0 if turn == w else -1.0)

            next_valid = env.valid_actions(turn) if (not done and env.alive[turn]) else []
            agent.update(state, action, reward, next_state, next_valid)

            state = next_state
            steps += 1

        for a in agents:
            a.decay_eps()

        if (ep + 1) % 50_000 == 0:
            print(f"Episode {ep+1}/{episodes} eps~{agents[0].eps:.3f}")

    return agents

def evaluate_agents(agents, accuracies, games=50_000, seed=7, start_turn=0):
    env = TruelEnv(accuracies, seed=seed)
    wins = [0, 0, 0]
    old_eps = [a.eps for a in agents]
    for a in agents:
        a.eps = 0.0
    for _ in range(games):
        state = env.reset(start_turn=start_turn)
        done = False
        steps = 0
        while not done and steps < 200:
            p = state[1]
            action = agents[p].choose_action(env, state)
            state, done = env.step(action)
            steps += 1
        wins[env.winner()] += 1
    for a, e in zip(agents, old_eps):
        a.eps = e
    return [w / games for w in wins]

if __name__ == "__main__":
    acc = (0.1, 0.8, 0.9)
    agents = train_shaped(acc, episodes=150_000, seed=42, step_cost=0.002)
    wr_rl = evaluate_agents(agents, acc, games=50_000, seed=7)
    wr_b1 = eval_baseline(act_shoot_strongest, acc, games=50_000, seed=101)
    wr_b2 = eval_baseline(act_A_pass_when_3_alive_else_shoot_strongest, acc, games=50_000, seed=202)
    print("RL shaped win rates (A,B,C):", [round(x,4) for x in wr_rl])
    print("Baseline shoot-strongest:", [round(x,4) for x in wr_b1])
    print("Baseline A-pass then strongest:", [round(x,4) for x in wr_b2])
