# Overview

Rather than assuming Rainbow's component-wise improvements transfer uniformly across settings, this project runs individual and combined DQN enhancements under a matched compute budget and asks: does each component's benefit hold up outside the conditions it was originally validated in?

The study spans two observation modalities:

Vector (1D): CartPole-v1, using the raw 4-dimensional physical state.
Pixel (2D): Four Atari 2600 environments — ALE/Freeway-v5, ALE/DemonAttack-v5, ALE/Frostbite-v5, and ALE/Krull-v5 — using stacked, grayscale, downsampled frames.

Rather than only comparing final scores, the pixel-modality analysis pairs each architectural component with a diagnostic that targets its specific mechanism, then checks whether the diagnostic evidence actually explains the resulting score difference — since the two don't always agree.

## What's implemented
Component	Mechanism	Diagnostic used
Vanilla DQN	Baseline, 1-step bootstrapped target	—
Double DQN	Decouples action selection (online net) from evaluation (target net) to reduce maximization bias	Holdout Q-value drift over a fixed state set
Dueling DQN	Decomposes Q(s,a) into state-value and advantage streams	Value/Advantage stream magnitude decomposition
N-Step DQN (n=20)	Multi-step return target, trading bootstrapping frequency for compounded variance	1-step vs. n-step TD-error magnitude distribution over training

Each configuration is trained across 3 seeds under a matched 100,000-step (400,000-frame) budget per Atari environment, with holdout evaluation on a fixed state set sampled at initialization.

## Environments
Below is the complete formal breakdown of how each game is modelled as a **Markov Decision Process (MDP)** 

```
+-----------------------------------------------------------------------------------------+
|                                    GAME MDP COMPARISON                                  |
+-------------------+-------------------+------------------------+------------------------+
| Game              | Action Space |A|  | Reward Distribution    | Key MDP Challenge      |
+-------------------+-------------------+------------------------+------------------------+
| ALE/Freeway-v5    | 3 Actions         | Extremely Sparse (+1)  | Credit assignment gap  |
| ALE/DemonAttack-v5| 6 Actions         | Very Dense (+1 hits)   | High-frequency timing  |
| ALE/Frostbite-v5  | 18 Actions        | Structured Hierarchical| Sub-goals & temporal decay
| ALE/Krull-v5      | 18 Actions        | Multi-Phase Composite  | Non-stationary regimes |
+-------------------+-------------------+------------------------+------------------------+
```

---

#### 1. ALE/Freeway-v5

* **State Space $\mathcal{S}$:**
  Visual positioning of the chicken (player), the opponent chicken, and 10 horizontal traffic lanes where cars travel at different fixed velocities (some left-to-right, some right-to-left).
* **Action Space $\mathcal{A}$ (3 Discrete Actions):**
  $$\mathcal{A} = \{\text{NOOP (0)}, \, \text{UP (1)}, \, \text{DOWN (2)}\}$$
* **Transition Dynamics $\mathcal{P}(s' \mid s, a)$:**
  * Action `UP` increments vertical position; `DOWN` decrements it.
  * Car collision does *not* lose a life; it induces a negative deterministic displacement (pushes the chicken backwards down several lanes).
  * Crossing the top lane teleports the chicken back to the bottom lane.
* **Reward Function $\mathcal{R}$:**
  $$\mathcal{R}(s, a, s') = \begin{cases} +1.0 & \text{if chicken crosses the top highway boundary} \\ 0.0 & \text{otherwise} \end{cases}$$
* **MDP Characteristic:** **Severe reward sparsity.** The agent must execute roughly 30–40 coordinated `UP` decisions while dodging traffic before receiving a single non-zero reward. Without intermediate rewards, early Q-values remain near zero.

---

#### 2. ALE/DemonAttack-v5

* **State Space $\mathcal{S}$:**
  Horizontal position of the bottom laser cannon, flying demon formations at the top, demonic laser trajectories dropping downwards, and ascending player missiles.
* **Action Space $\mathcal{A}$ (6 Discrete Actions):**
  $$\mathcal{A} = \{\text{NOOP (0)}, \, \text{FIRE (1)}, \, \text{RIGHT (2)}, \, \text{LEFT (3)}, \, \text{RIGHTFIRE (4)}, \, \text{LEFTFIRE (5)}\}$$
* **Transition Dynamics $\mathcal{P}(s' \mid s, a)$:**
  * Demons follow sinusoidal hovering trajectories and fire randomized downward lasers.
  * In later waves, shooting a demon splits it into two smaller, rapidly diving sub-demons.
  * Laser impact with player cannon decreases life count $\implies$ triggers `terminated = True`.
* **Reward Function $\mathcal{R}$:**
  * *Raw Rewards:* $+10$ to $+35$ per demon, bonuses per wave cleared.
  * *Clipped Rewards (Agent's perspective):*
    $$\mathcal{R}(s, a, s') = \begin{cases} +1.0 & \text{on any demon hit / destruction event} \\ 0.0 & \text{otherwise} \end{cases}$$
* **MDP Characteristic:** **Dense, high-frequency credit assignment.** The agent receives frequent immediate positive reinforcement, leading to rapid Q-value escalation and early policy divergence.

---

#### 3. ALE/Frostbite-v5

* **State Space $\mathcal{S}$:**
  Frostbite Bailey's position, 4 rows of floating ice blocks moving horizontally in alternating directions, the igloo structure status (15 sequential blocks), temperature countdown bar, and obstacles (birds, clams, crabs, polar bear).
* **Action Space $\mathcal{A}$ (18 Discrete Actions):**
  Full 18-action Atari joystick space (all 8 directional movements $\times$ with/without `FIRE`).
* **Transition Dynamics $\mathcal{P}(s' \mid s, a)$:**
  * Hopping onto an unvisited white ice row changes it to blue and adds 1 block to the igloo.
  * Stepping into water or touching moving hazards decrements life $\implies$ triggers `terminated = True`.
  * After all 15 igloo blocks are constructed, jumping into the igloo entrance terminates the level successfully.
* **Reward Function $\mathcal{R}$:**
  * *Raw Rewards:* $+10$ per active ice block hopped, $+100$ to $+1000$ upon entering completed igloo, plus remaining temperature bonus.
  * *Clipped Rewards (Agent's perspective):*
    $$\mathcal{R}(s, a, s') = \begin{cases} +1.0 & \text{on each newly activated ice floe or stage completion} \\ 0.0 & \text{otherwise} \end{cases}$$
* **MDP Characteristic:** **Hierarchical / Sequential sub-goals.** The policy must balance short-term reward (hopping on floes) with terminal planning (returning safely to the top igloo before temperature hits zero).

---

#### 4. ALE/Krull-v5

* **State Space $\mathcal{S}$:**
  A complex, **multi-phase state distribution** representing 4 distinct screen environments:
  1. *The Wedding / Citadel:* Navigating a hall dodging fireballs.
  2. *Iron Valley:* Riding a horse across scrolling plains while avoiding obstacles.
  3. *The Swamp:* Rescuing trapped prisoners while fighting Slayer troops.
  4. *The Black Fortress:* Final battle hurling the "Glaive" weapon at the Beast.
* **Action Space $\mathcal{A}$ (18 Discrete Actions):**
  Full 18-action Atari joystick space.
* **Transition Dynamics $\mathcal{P}(s' \mid s, a)$:**
  * Non-standard macro-transitions: completing a mini-goal triggers a phase shift into an entirely different screen layout with different movement rules.
  * Life loss in any phase triggers `terminated = True`.
* **Reward Function $\mathcal{R}$:**
  * *Raw Rewards:* Weapon collection, defeating slayers, completing stages.
  * *Clipped Rewards (Agent's perspective):*
    $$\mathcal{R}(s, a, s') = \begin{cases} +1.0 & \text{for combat hits, pickups, or phase advancements} \\ 0.0 & \text{otherwise} \end{cases}$$
* **MDP Characteristic:** **High structural variance / Multi-modal dynamics.** The feature representations learned for one scene (e.g. dodging in the Citadel) must not interfere destructively with the representations required for other scenes (e.g. throwing weapons at the Beast).

---

