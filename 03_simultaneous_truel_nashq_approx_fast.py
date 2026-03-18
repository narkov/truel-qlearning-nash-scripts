#!/usr/bin/env python3
"""
Simultaneous-fire truel + Nash-Q (approx) using a fast logit best-response fixed point
for product mixed strategies at each state.

This is experimental: not a guaranteed Nash solver for 3-player general-sum games.
Useful as a starting point.

Usage:
  python 03_simultaneous_truel_nashq_approx_fast.py
"""
import random
from collections import defaultdict
import numpy as np

class SimultaneousTruelEnv:
    def __init__(self, accuracies=(0.1, 0.8, 0.9), step_cost=0.002, seed=0):
        self.acc = list(accuracies)
        self.step_cost = step_cost
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.alive = [True, True, True]
        return self.state()

    def state(self):
        m = 0
        for i in range(3):
            if self.alive[i]:
                m |= (1 << i)
        return m

    def is_terminal(self):
        return sum(self.alive) <= 1

    def winner(self):
        if sum(self.alive) == 1:
            return self.alive.index(True)
        return None

    def action_set(self, player):
        if not self.alive[player]:
            return [("dead", None)]
        acts = [("miss", None)]
        for t in range(3):
            if t != player and self.alive[t]:
                acts.append(("shoot", t))
        return acts

    def step(self, joint_action):
        if self.is_terminal():
            return self.state(), [0.0, 0.0, 0.0], True

        rewards = [0.0, 0.0, 0.0]
        for i in range(3):
            if self.alive[i]:
                rewards[i] -= self.step_cost

        hits_on = [False, False, False]
        for shooter in range(3):
            if not self.alive[shooter]:
                continue
            kind, target = joint_action[shooter]
            if kind == "shoot" and target is not None and self.alive[target]:
                if self.rng.random() < self.acc[shooter]:
                    hits_on[target] = True

        for t in range(3):
            if hits_on[t]:
                self.alive[t] = False

        done = self.is_terminal()
        if done:
            w = self.winner()
            if w is None:
                rewards = [r - 1.0 for r in rewards]
            else:
                for i in range(3):
                    rewards[i] += (1.0 if i == w else -1.0)

        return self.state(), rewards, done

def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)

def approx_eq_fast(Qi, iters=40, tau=0.18):
    Q0, Q1, Q2 = Qi
    A0, A1, A2 = Q0.shape
    pi0 = np.ones(A0) / A0
    pi1 = np.ones(A1) / A1
    pi2 = np.ones(A2) / A2

    for _ in range(iters):
        u0 = np.tensordot(Q0, np.outer(pi1, pi2), axes=([1, 2], [0, 1]))
        pi0 = softmax(u0 / tau)
        u1 = np.tensordot(Q1, np.outer(pi0, pi2), axes=([0, 2], [0, 1]))
        pi1 = softmax(u1 / tau)
        u2 = np.tensordot(Q2, np.outer(pi0, pi1), axes=([0, 1], [0, 1]))
        pi2 = softmax(u2 / tau)

    def expected_u(Q):
        return np.sum(Q * pi0[:, None, None] * pi1[None, :, None] * pi2[None, None, :])
    u0 = expected_u(Q0); u1 = expected_u(Q1); u2 = expected_u(Q2)
    br0 = np.max(np.tensordot(Q0, np.outer(pi1, pi2), axes=([1,2],[0,1])))
    br1 = np.max(np.tensordot(Q1, np.outer(pi0, pi2), axes=([0,2],[0,1])))
    br2 = np.max(np.tensordot(Q2, np.outer(pi0, pi1), axes=([0,1],[0,1])))
    regret = float((br0-u0)+(br1-u1)+(br2-u2))
    return [pi0, pi1, pi2], regret

class NashQApprox:
    def __init__(self, gamma=0.98, alpha=0.25, eps_start=1.0, eps_end=0.05, eps_decay=0.9993,
                 eq_refresh_every=1000, eq_iters=35, tau=0.18):
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.eq_refresh_every = eq_refresh_every
        self.eq_iters = eq_iters
        self.tau = tau

        self.Q = {i: {} for i in range(3)}
        self.visit = defaultdict(int)
        self.cache = {}

    def _init_state(self, env, s):
        if s in self.Q[0]:
            return
        action_lists = [env.action_set(i) for i in range(3)]
        sizes = [len(al) for al in action_lists]
        for i in range(3):
            self.Q[i][s] = np.zeros(sizes, dtype=np.float64)
        self.cache[s] = {"pi": None, "action_lists": action_lists, "regret": None}

    def _compute_eq(self, s):
        pi, reg = approx_eq_fast([self.Q[i][s] for i in range(3)], iters=self.eq_iters, tau=self.tau)
        self.cache[s]["pi"] = pi
        self.cache[s]["regret"] = reg

    def _get_eq(self, s):
        if self.cache[s]["pi"] is None:
            self._compute_eq(s)
        return self.cache[s]["pi"], self.cache[s]["action_lists"], self.cache[s]["regret"]

    def _sample_joint(self, pi, action_lists):
        ja = []
        for i in range(3):
            idx = int(np.random.choice(len(action_lists[i]), p=pi[i]))
            ja.append(action_lists[i][idx])
        return tuple(ja)

    def _V(self, s):
        pi, _, _ = self._get_eq(s)
        p0,p1,p2 = pi
        return [float(np.sum(self.Q[i][s] * p0[:,None,None] * p1[None,:,None] * p2[None,None,:])) for i in range(3)]

    def decay_eps(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def train(self, accuracies=(0.1,0.8,0.9), episodes=12_000, seed=42, step_cost=0.002, max_steps=60):
        random.seed(seed); np.random.seed(seed)
        env = SimultaneousTruelEnv(accuracies=accuracies, step_cost=step_cost, seed=seed)

        for ep in range(episodes):
            s = env.reset()
            done = False
            steps = 0
            while not done and steps < max_steps:
                self._init_state(env, s)
                self.visit[s] += 1
                if self.cache[s]["pi"] is None or (self.visit[s] % self.eq_refresh_every == 0):
                    self._compute_eq(s)

                pi, action_lists, _ = self._get_eq(s)
                if random.random() < self.eps:
                    ja = tuple(random.choice(action_lists[i]) for i in range(3))
                else:
                    ja = self._sample_joint(pi, action_lists)

                s2, r, done = env.step(ja)
                self._init_state(env, s2)
                self.visit[s2] += 1
                if self.cache[s2]["pi"] is None or (self.visit[s2] % self.eq_refresh_every == 0):
                    self._compute_eq(s2)

                V2 = self._V(s2)
                idxs = tuple(action_lists[i].index(ja[i]) for i in range(3))
                for i in range(3):
                    target = r[i] + (0.0 if done else self.gamma * V2[i])
                    self.Q[i][s][idxs] = (1-self.alpha)*self.Q[i][s][idxs] + self.alpha*target

                s = s2
                steps += 1

            self.decay_eps()
            if (ep + 1) % 3000 == 0:
                self._init_state(env, 7)
                self._compute_eq(7)
                print(f"ep={ep+1} eps={self.eps:.3f} regret(ABC)={self.cache[7]['regret']:.3f}")

if __name__ == "__main__":
    nq = NashQApprox()
    nq.train(accuracies=(0.1,0.8,0.9), episodes=12_000)
