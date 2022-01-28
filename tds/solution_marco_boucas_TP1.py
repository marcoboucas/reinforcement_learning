"""Boucas Marco."""
# pylint:: disable=invalid-name

import pdb
from time import perf_counter
from typing import Dict

import gym.envs.toy_text.frozen_lake as fl
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=3)
NBR_EPISODES = 50000
HORIZON = 200
GAMMA = 0.9
SEED = int("whatAnAwesomePracticalSession", base=36) % 2 ** 31
Policy = Dict[str, int]

# Create environment
env = fl.FrozenLakeEnv()  # gym.make('FrozenLake-v1')
# env.seed(SEED)


def to_s(row, col):
    return row * env.ncol + col


def to_row_col(s):
    col = s % env.ncol
    row = s // env.ncol
    return row, col


def print_values(v):
    for row in range(env.nrow):
        for col in range(env.ncol):
            s = f"{v[to_s(row, col)]:.3}"
            print(s, end=" " * (8 - len(s)))
        print("")


def convert_time(t1, t2):
    return f"Running time: {t2 - t1:.4f} sec\n"


# Question 3
def get_random_policy(nA: int, nS: int) -> Policy:
    """Get random policy."""
    return np.random.randint(0, nA - 1, nS)


def question_3():
    """Evaluate one policy using Monte Carlo estimation."""
    policy = get_random_policy(env.nA, env.nS)
    V = np.zeros((NBR_EPISODES,))
    for i in range(NBR_EPISODES):
        state = env.reset()
        done = False
        t = 0
        discount = 1
        while (not done) and (t < HORIZON):
            state, reward, done, _ = env.step(policy[state])
            V[i] += discount * reward
            discount *= GAMMA
            t += 1
    print(f"Value estimate of the starting point: {np.mean(V):.4f}")

    offset = 10
    plt.figure()
    plt.title(
        "Convergence of the Monte Carlo estimation\nof the value of the \
    starting point"
    )
    plt.plot((np.cumsum(V) / (np.arange(NBR_EPISODES) + 1))[offset:])
    plt.show()


if True:
    question_3()
pdb.set_trace()

# Question 4


def value_function_expected(policy: Policy):
    V = np.zeros(env.nS)
    for init_state in range(env.nS):
        # Change the starting point of the env
        state_value = 0
        state = init_state
        env.isd = np.zeros(env.nS)
        env.isd[init_state] = 1
        env.reset()

        # Monte Carlo to estimate the value
        for _ in range(NBR_EPISODES):
            env.reset()
            done = False
            t = 0
            discount = 1
            while not done and t < HORIZON:

                state, reward, done, _ = env.step(policy[state])
                state_value += discount * reward
                discount *= GAMMA
                t += 1
        V[init_state] = state_value / NBR_EPISODES
    return V


def question_4():
    """Question 4."""
    print("\n\n######################")
    print("##### Question 4 #####")
    print("######################\n")
    print("EXPECTED VALUE METHOD\n")

    simple_pi = fl.RIGHT * np.ones(env.nS)
    starting_time = perf_counter()
    V_simple_pi = value_function_expected(simple_pi)
    print(convert_time(starting_time, perf_counter()))
    print(f"Value estimate of the starting point: {V_simple_pi[0]:.3f}")
    print(f"Value function of the always RIGHT policy:\n")
    print_values(V_simple_pi)

    # reset the original isd
    env.isd = np.zeros(env.nS)
    env.isd[0] = 1


if True:
    question_4()
"""
Les valeurs de V semblent similaires (en réalité, même si les valeurs
 sont différents, on peut comparer les valeurs aux différents états,
 mais les comparaisons inter-états semblent les mêmes, et c'est surtout cela qui importe)
"""


pdb.set_trace()
# Question 5


def value_function(policy: Policy):
    """
    pi : int array
    For each index i, pi[i] is the action (int) chosen in state i

    return:
    ------
    V_pi : float array
    For each index i, V_pi[i] is the value (float) of the state i
    """
    # Compute both the reward vector r_pi and
    # transition matrix P_pi associated to the policy on the given env
    reward_pi = np.zeros(env.nS)
    P_pi = np.zeros((env.nS, env.nS))
    for state in range(env.nS):
        for proba, next_state, reward, _ in env.P[state][policy[state]]:
            P_pi[state, next_state] += proba
            reward_pi[state] += reward * proba
    # Compute the value function of the policy pi
    return np.linalg.inv(np.eye(env.nS) - GAMMA * P_pi) @ reward_pi


def question_5():
    """Question 5: Linear System method.

    We use the Bellman operator:
    v^\pi = T_\pi [v_\pi]
    with T_\pi [v] = r_\pi + \gamma * P_\pi * v

    So we get V^\pi = (I - \gamma * P_\pi)^-1 * r_\pi
    """
    print("\n######################")
    print("##### Question 5 #####")
    print("######################\n")
    print("LINEAR SYSTEM METHOD\n")
    simple_pi = fl.RIGHT * np.ones(env.nS)
    starting_time = perf_counter()
    V_simple_pi = value_function(simple_pi)
    print(convert_time(starting_time, perf_counter()))
    print(f"Value estimate of the starting point: {V_simple_pi[0]:.3f}")
    print(f"Value function of the always RIGHT policy:\n")
    print_values(V_simple_pi)


if True:
    question_5()


pdb.set_trace()
# Question 6


def value_function_2(policy, epsilon, max_iter):
    """
    pi : int array
    For each index i, pi[i] is the action (int) chosen in state i

    epsilon : float
    Used as a threshold for the stopping rule

    max_iter : int
    Hard threshold on the number of loops

    return:
    ------
    V_pi : float array
    For each index i, V_pi[i] is the value (float) of the state i
    """
    # Compute both the reward vector r_pi and
    # transition matrix P_pi associated to the policy on the given env
    r_pi = np.zeros(env.nS)
    P_pi = np.zeros((env.nS, env.nS))
    for state in range(env.nS):
        for proba, next_state, reward, _ in env.P[state][policy[state]]:
            P_pi[state, next_state] += proba
            r_pi[state] += reward * proba
    # Compute the value function V_pi of the policy pi
    v_pi = np.zeros(env.nS)
    v_pi_old = np.zeros(env.nS)
    delta_inf = np.zeros(max_iter)
    stop = False
    i = 0
    while (not stop) and (i < max_iter):
        v_pi = r_pi + GAMMA * (P_pi @ v_pi_old)
        delta_inf[i] = np.max(np.abs(v_pi - v_pi_old))
        v_pi_old[:] = v_pi
        if delta_inf[i] < epsilon:
            stop = True
            delta_inf = delta_inf[: i + 1]
        i += 1
    return v_pi, delta_inf


def question_6():
    simple_pi = fl.RIGHT * np.ones(env.nS)
    starting_time = perf_counter()
    V_simple_pi, Delta_inf = value_function_2(simple_pi, 1e-4, 10000)
    print(convert_time(starting_time, perf_counter()))
    print(f"Value function of the always RIGHT policy:\n")
    print_values(V_simple_pi)

    plt.figure()
    plt.title(
        "Semi-log graph of $n \mapsto || V_{n+1} - V_n ||_\infty $ \n\
    The Linearity of this graph proves exponential convergence"
    )
    plt.semilogy(Delta_inf)
    plt.xlabel("Iterate")
    plt.ylabel(r"$|| V_{n+1} - V_n ||_\infty$")
    # plt.savefig("question6.png")
    print(f"\nNumber of iterations: {Delta_inf.size}")
    print(f"Last residual {Delta_inf[-1]:.6f}")
    plt.show()


if True:
    question_6()

"""
Je ne sais pas si cela pourrait être pertinent, mais ne peut-on pas
regarder si la policy générée par la value est identique à celle que l'on
a en entrée ?
"""

pdb.set_trace()
# Question 7


def value_function_optimal(epsilon, max_iter):
    """
    epsilon : float
    Used as a threshold for the stopping rule

    max_iter : int
    Hard threshold on the number of loops

    returns:
    -------
    V_opt : float array, (env.nS,) size
    Optimal value function on the FrozenLake MDP given a discount GAMMA
    V_opt[state index] = Value of that state
    """
    r_pi_a = np.zeros((env.nA, env.nS))
    P_pi_a = np.zeros((env.nA, env.nS, env.nS))
    for action in range(env.nA):
        for state in range(env.nS):
            for proba, next_state, reward, _ in env.P[state][action]:
                P_pi_a[action, state, next_state] += proba
                r_pi_a[action, state] += reward * proba
    # Compute the value function V_pi of the policy pi
    v_opt = np.zeros(env.nS)
    v_opt_old = np.zeros(env.nS)
    delta_inf = np.zeros(max_iter)
    stop = False
    i = 0
    while (not stop) and (i < max_iter):
        v_opt_a = []
        for action in range(env.nA):
            x = r_pi_a[action] + GAMMA * (P_pi_a[action] @ v_opt_old)
            v_opt_a.append(np.expand_dims(x, 0))
        v_opt = np.max(np.concatenate(v_opt_a, axis=0), axis=0)
        delta_inf[i] = np.max(np.abs(v_opt - v_opt_old))
        v_opt_old = np.copy(v_opt)
        if delta_inf[i] < epsilon:
            stop = True
            delta_inf = delta_inf[: i + 1]
        i += 1
    return v_opt, delta_inf


def question_7():
    print("\n######################")
    print("##### Question 7 #####")
    print("######################\n")
    print("OPTIMAL BELLMAN OPERATOR\n")

    starting_time = perf_counter()
    V_opt, Delta_inf = value_function_optimal(1e-4, 10000)
    print(convert_time(starting_time, perf_counter()))
    print(f"Optimal value function:\n")
    print(V_opt)
    print_values(V_opt)

    plt.figure()
    plt.title(
        "Semi-log graph of $n \mapsto || V_{n+1} - V_n ||_\infty $ \n\
    The Linearity of this graph proves exponential convergence"
    )
    plt.semilogy(Delta_inf)
    plt.xlabel("Iterate")
    plt.ylabel(r"$|| V_{n+1} - V_n ||_\infty$")
    plt.savefig("question7.png")
    print(f"\nNumber of iterations: {Delta_inf.size}")
    print(f"Last residual {Delta_inf[-1]:.6f}")


if True:
    question_7()

pdb.set_trace()
# Question 8


def value_iteration(epsilon, max_iter):
    """
    epsilon : float
    Used as a threshold for the stopping rule

    max_iter : int
    Hard threshold on the number of loops

    returns:
    -------
    pi : int array, size (env.nS,)
    An optimal policy
    """
    r_pi_a = np.zeros((env.nA, env.nS))
    P_pi_a = np.zeros((env.nA, env.nS, env.nS))
    for action in range(env.nA):
        for state in range(env.nS):
            for proba, next_state, reward, _ in env.P[state][action]:
                P_pi_a[action, state, next_state] += proba
                r_pi_a[action, state] += reward * proba
    # Compute the value function V_pi of the policy pi
    v_opt = np.zeros(env.nS)
    v_opt_old = np.ones(env.nS)
    i = 0
    while np.linalg.norm(v_opt - v_opt_old) > epsilon and (i < max_iter):
        v_opt_a = []
        v_opt_old = np.copy(v_opt)
        for action in range(env.nA):
            x = r_pi_a[action] + GAMMA * (P_pi_a[action] @ v_opt_old)
            v_opt_a.append(np.expand_dims(x, 0))
        v_opt = np.max(np.concatenate(v_opt_a, axis=0), axis=0)

        i += 1

    # We know the value function, we can compute the policy:
    pi = np.zeros((env.nS))
    for state in range(env.nS):
        pi[state] = np.argmax(
            [
                sum(
                    [
                        P_pi_a[action, state, next_state]
                        * (r_pi_a[action, state] + GAMMA * v_opt[next_state])
                        for next_state in range(env.nS)
                    ]
                )
                for action in range(env.nA)
            ]
        )
    return pi


ARROWS = {fl.RIGHT: "→", fl.LEFT: "←", fl.UP: "↑", fl.DOWN: "↓"}


def print_policy(policy):
    for row in range(env.nrow):
        for col in range(env.ncol):
            print(ARROWS[policy[to_s(row, col)]], end="")
        print("")


def question_8():
    """Question 8."""
    print("\n######################")
    print("##### Question 8 #####")
    print("######################\n")
    print("VALUE ITERATION\n")
    starting_time = perf_counter()
    Pi_opt = value_iteration(1e-4, 1000)
    print(Pi_opt.shape)
    print(convert_time(starting_time, perf_counter()))
    print("An optimal policy is:\n")
    print_policy(Pi_opt)


if True:
    question_8()

pdb.set_trace()
# Question 9


# The danger of Policy Iteration lies in the stopping criterion
# If not careful, one might end up with an algorithm that does not
# terminate and oscillates between optimal policies
# Even if it is computationally more expensive, we sometimes rather
# compare value functions of the policies than policies from one iterate
# to another.

# An easy improvement on the following code would be to use
# a warm start for policy evaluation steps (if iteration methods is used)
# That is to say, using the previously computed value function
# as the first step for the next policy evaluation


def policy_improvement(*, v, r_pi_a, P_pi_a):
    """
    V : float array, size (env.nS,)
    Value function of a policy

    returns:
    -------
    pi : int array, size (env.nS,)
    A policy that is greedy with respect to V
    """
    pi = np.zeros((env.nS))
    for state in range(env.nS):
        pi[state] = np.argmax(
            [
                sum(
                    [
                        P_pi_a[action, state, next_state]
                        * (r_pi_a[action, state] + GAMMA * v[next_state])
                        for next_state in range(env.nS)
                    ]
                )
                for action in range(env.nA)
            ]
        )
    return pi


def policy_iteration(max_iter, epsilon: float = 1e-4):
    """
    epsilon : float
    Used as a threshold for the stopping rule

    max_iter : int
    Hard threshold on the number of loops

    returns:
    -------
    pi : int array, size (env.nS,)
    An optimal policy
    """
    r_pi_a = np.zeros((env.nA, env.nS))
    P_pi_a = np.zeros((env.nA, env.nS, env.nS))
    for action in range(env.nA):
        for state in range(env.nS):
            for proba, next_state, reward, _ in env.P[state][action]:
                P_pi_a[action, state, next_state] += proba
                r_pi_a[action, state] += reward * proba
    # Compute the value function V_pi of the policy pi
    v = np.zeros(env.nS, dtype=np.int16)
    i = 0
    policy = np.zeros(env.nS)
    old_policy = np.ones(env.nS)

    # While the policy is changing
    while np.linalg.norm(policy - old_policy) > 0 and (i < max_iter):
        # Find the value of V_{\pi}
        v_pi, _ = value_function_2(policy, 1e-4, max_iter)

        # Recompute the policy
        old_policy = np.copy(policy)
        policy = policy_improvement(v=v_pi, r_pi_a=r_pi_a, P_pi_a=P_pi_a)

        i += 1
    return policy


def question_9():
    print("\n######################")
    print("##### Question 9 #####")
    print("######################\n")
    print("POLICY ITERATION\n")

    starting_time = perf_counter()
    Pi_opt = policy_iteration(1000)
    print(convert_time(starting_time, perf_counter()))
    print("An optimal policy is:\n")
    print_policy(Pi_opt)


if True:
    question_9()
"""
VALUE ITERATION

(16,)
Running time: 0.0031 sec

An optimal policy is:

←↑←↑
←←←←
↑↓←←
←→↓←


POLICY ITERATION

Running time: 0.0057 sec

An optimal policy is:

←↑←↑
←←←←
↑↓←←
←→↓←

Bon, déjà on obtient les mêmes résultats, ce qui est rassurant.
En terme de temps, il semblerait que la policy iteration mette un
peu plus de temps à converger vers la solution.

Pour vérifier cela, après quelques recherches internet, cela me semble
 plus nuancé, car on peut retrouver de ferveurs défenseurs des 2 méthodes.

"""
pdb.set_trace()
# Question 11


def state_value_function_optimal(epsilon, max_iter):
    """
    epsilon : float
    Used as a threshold for the stopping rule

    max_iter : int
    Hard threshold on the number of loops

    returns:
    -------
    q_opt : float array, (env.nS, env.nA) size
    Optimal state-action value function on the FrozenLake MDP
    given a discount GAMMA
    q_opt[state index][action index] = state-action value of that state
    """
    r_pi_a = np.zeros((env.nA, env.nS))
    P_pi_a = np.zeros((env.nA, env.nS, env.nS))
    for action in range(env.nA):
        for state in range(env.nS):
            for proba, next_state, reward, _ in env.P[state][action]:
                P_pi_a[action, state, next_state] += proba
                r_pi_a[action, state] += reward * proba

    # Iterate until the Q is good
    Q = np.zeros((env.nS, env.nA))
    Q_old = np.ones_like(Q)
    delta_inf = np.zeros(max_iter)
    i = 0
    stop = False
    while (not stop) and i < max_iter:

        for state in range(env.nS):
            for action in range(env.nA):
                Q[state, action] = sum(
                    [
                        P_pi_a[action, state, next_state]
                        * (r_pi_a[action, state] + GAMMA * np.max(Q[next_state]))
                        for next_state in range(env.nS)
                    ]
                )

        delta_inf[i] = np.max(np.abs(Q - Q_old))
        Q_old = np.copy(Q)
        if delta_inf[i] < epsilon:
            stop = True
            delta_inf = delta_inf[: i + 1]

        i += 1
    return Q, delta_inf


def question_11_and_12():
    print("\n#######################")
    print("##### Question 11 #####")
    print("#######################\n")
    print("OPTIMAL Q-BELLMAN OPERATOR METHOD\n")

    starting_time = perf_counter()
    Q_opt, Delta_inf = state_value_function_optimal(1e-4, 100)
    print(convert_time(starting_time, perf_counter()))
    # print(Q_opt)
    V_opt = Q_opt.max(axis=1)
    print(f"Optimal value function:\n")
    print_values(V_opt)

    print(f"Optimal policy")
    Pi_opt = np.argmax(Q_opt, axis=1)
    print_policy(Pi_opt)

    plt.figure()
    plt.title(
        "Semi-log graph of $n \mapsto || Q_{n+1} - Q_n ||_\infty $ \n\
    The Linearity of this graph proves exponential convergence"
    )
    plt.semilogy(Delta_inf)
    plt.xlabel("Iterate")
    plt.ylabel(r"$|| Q_{n+1} - Q_n ||_\infty$")
    plt.savefig("question11.png")
    print(f"\nNumber of iterations: {Delta_inf.size}")
    print(f"Last residual {Delta_inf[-1]:.6f}")
    # plt.show()
    return Pi_opt


if True:
    Pi_opt = question_11_and_12()


# render policy
def trajectory(pi, max_moves=100, render=False):
    done = False
    i = 0
    env.reset()
    cumulative_reward = 0
    discount = 1
    while not done and i < max_moves:
        i += 1
        _, r, done, _ = env.step(pi[env.s])
        cumulative_reward += discount * r
        discount *= GAMMA
        if render:
            env.render()
            print("")
    return cumulative_reward


def question_13():
    print("\n#######################")
    print("##### Question 13 #####")
    print("#######################\n")
    print("RENDER A TRAJECTORY\n")

    cr = trajectory(Pi_opt)
    print("\nThe GOAL has been reached! Congrats! :-)")
    print(f"The cumulative discounted reward along the above trajectory is: {cr:.3f}\n")


if True:
    question_13()


def question_13_bis():
    """Multiple trajectories."""

    crs = []
    for _ in range(100):
        crs.append(trajectory(Pi_opt))
    plt.figure()
    plt.hist(crs, bins=10)
    plt.show()


if True:
    print("DISPLAY THE RESULTS FOR MULTIPLE TRAJECTORIES")
    print('Because not all of them succeed, so the seed might be a "wrong" one')
    question_13_bis()
