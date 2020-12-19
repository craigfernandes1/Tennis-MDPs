import numpy as np
import pandas as pd
from pathlib import Path
import math
import DataManipulation as dm
import matplotlib.pyplot as plt
import random
from PlottingFunction import createBins

# Action set dictionary
# {0: 'serve_corner', 1: 'serve_middle', 2: 'rally_short_ad', 3: 'rally_short_middle',
#  4: 'rally_short_deuce', 5: 'rally_deep_ad', 6: 'rally_deep_middle', 7: 'rally_deep_deuce'}

# Import all the T's for each action
T_0_ad_corner = pd.read_pickle(Path.cwd() / 'pickle' / 'SDP' / 'T_serve_ad_corner_fault.plk')
T_1_ad_middle = pd.read_pickle(Path.cwd() / 'pickle' / 'SDP' / 'T_serve_ad_middle_fault.plk')

T_0_deuce_corner = pd.read_pickle(Path.cwd() / 'pickle' / 'SDP' / 'T_serve_deuce_corner_fault.plk')
T_1_deuce_middle = pd.read_pickle(Path.cwd() / 'pickle' / 'SDP' / 'T_serve_deuce_middle_fault.plk')

T_2_short_ad = pd.read_pickle(Path.cwd() / 'pickle' / 'SDP' / 'T_rally_short_ad_fault.plk')
T_3_short_middle = pd.read_pickle(Path.cwd() / 'pickle' / 'SDP' / 'T_rally_short_middle_fault.plk')
T_4_short_deuce = pd.read_pickle(Path.cwd() / 'pickle' / 'SDP' / 'T_rally_short_deuce_fault.plk')
T_5_deep_ad = pd.read_pickle(Path.cwd() / 'pickle' / 'SDP' / 'T_rally_deep_ad_fault.plk')
T_6_deep_middle = pd.read_pickle(Path.cwd() / 'pickle' / 'SDP' / 'T_rally_deep_middle_fault.plk')
T_7_deep_deuce = pd.read_pickle(Path.cwd() / 'pickle' / 'SDP' / 'T_rally_deep_deuce_fault.plk')

T_ad_serve_set = {0: T_0_ad_corner, 1: T_1_ad_middle}

T_deuce_serve_set = {0: T_0_deuce_corner, 1: T_1_deuce_middle}

T_rally_set = {2: T_2_short_ad, 3: T_3_short_middle, 4: T_4_short_deuce,
               5: T_5_deep_ad, 6: T_6_deep_middle, 7: T_7_deep_deuce}

# Import all the Counts's for each action
C_0_ad_corner = pd.read_pickle(Path.cwd() / 'pickle' / 'SDP' / 'counts_serve_ad_corner_fault.plk')
C_1_ad_middle = pd.read_pickle(Path.cwd() / 'pickle' / 'SDP' / 'counts_serve_ad_middle_fault.plk')

C_0_deuce_corner = pd.read_pickle(Path.cwd() / 'pickle' / 'SDP' / 'counts_serve_deuce_corner_fault.plk')
C_1_deuce_middle = pd.read_pickle(Path.cwd() / 'pickle' / 'SDP' / 'counts_serve_deuce_middle_fault.plk')

C_2_short_ad = pd.read_pickle(Path.cwd() / 'pickle' / 'SDP' / 'counts_rally_short_ad_fault.plk')
C_3_short_middle = pd.read_pickle(Path.cwd() / 'pickle' / 'SDP' / 'counts_rally_short_middle_fault.plk')
C_4_short_deuce = pd.read_pickle(Path.cwd() / 'pickle' / 'SDP' / 'counts_rally_short_deuce_fault.plk')
C_5_deep_ad = pd.read_pickle(Path.cwd() / 'pickle' / 'SDP' / 'counts_rally_deep_ad_fault.plk')
C_6_deep_middle = pd.read_pickle(Path.cwd() / 'pickle' / 'SDP' / 'counts_rally_deep_middle_fault.plk')
C_7_deep_deuce = pd.read_pickle(Path.cwd() / 'pickle' / 'SDP' / 'counts_rally_deep_deuce_fault.plk')

C_ad_serve_set = {0: C_0_ad_corner, 1: C_1_ad_middle}

C_deuce_serve_set = {0: C_0_deuce_corner, 1: C_1_deuce_middle}

C_rally_set = {2: C_2_short_ad, 3: C_3_short_middle, 4: C_4_short_deuce,
               5: C_5_deep_ad, 6: C_6_deep_middle, 7: C_7_deep_deuce}


def get_dist_count(set, offset=0):
    C_np = np.zeros((5297, len(set)))
    for i in range(len(set)):
        C_np[:, i] = set[i + offset][0]

    return C_np.transpose()


def get_particular_q_dist(set, offset=0):
    q = get_dist_count(set,offset)
    q = pd.DataFrame(q.transpose())
    q['total'] = q.sum(axis=1)

    for i in range(q.shape[1] - 1):
        q.iloc[:, i] = q.iloc[:, i] / q['total']

    q.drop(columns=['total'], inplace=True)
    q = q.replace(np.nan, 0)

    return q

def get_q_dist():
    q_ad_serve = get_particular_q_dist(C_ad_serve_set)
    q_deuce_serve = get_particular_q_dist(C_deuce_serve_set)
    q_rally = get_particular_q_dist(C_rally_set, 2)

    return q_ad_serve, q_deuce_serve, q_rally

def get_most_frequent_policy():
    C_ad_serve_np = get_dist_count(C_ad_serve_set)
    C_deuce_serve_np = get_dist_count(C_deuce_serve_set)
    C_rally_np = get_dist_count(C_rally_set, 2)

    freq_policy = np.concatenate((np.ones((1764, 1)) * 0, np.ones((1764 * 2, 1)) * 2)).astype(int)

    for state in range(5292):

        if state < 1764:  # serve states
            serve_type = check_serve_ad_vs_deuce(state)

            if serve_type == 'ad':
                freq_policy[state] = np.argmax(C_ad_serve_np[:, state])
            elif serve_type == 'deuce':
                freq_policy[state] = np.argmax(C_deuce_serve_np[:, state])

        elif state >= 1764:  # rally states
            freq_policy[state] = np.argmax(C_rally_np[:, state])+2

    return freq_policy

def add_generic_error_to_policy(policy, error_frac):

    policy_err = np.copy(policy)
    n_states = 5292
    perturb = np.random.choice(np.arange(n_states), int(n_states * error_frac), replace=False)

    for state in perturb:
        action_current = policy_err[state]

        serve_actions = [0, 1]
        rally_actions = [2, 3, 4, 5, 6, 7]

        if action_current <= 1:
            serve_actions.remove(action_current)
            policy_err[state] = serve_actions
        else:
            rally_actions.remove(action_current)
            policy_err[state] = random.choice(rally_actions)

    return policy_err

def plot_error_values(V_stats,title,col=3):
    plt.figure()
    V_stats.iloc[:, col].plot(marker='^')
    plt.xlabel('Fraction of Execution Error')
    if col == 0:
        plt.ylabel('Sum of Values')
    elif col == 1:
        plt.ylabel('Average Value of States (n>0)')
    elif col == 2:
        plt.ylabel('Average Value of States (n>30)')
    elif col == 3:
        plt.ylabel('Average Value of Serve States')
    elif col == 4:
        plt.ylabel('Number of States with Less Value than Base')
    plt.title(title)


def get_policy_specific_transition_matrix(opt_policy):
    T = pd.DataFrame(np.zeros((5297, 5297)))

    for state in range(5292):
        action = opt_policy[state][0]

        T_row = get_particular_state_transition(state, action)

        T.iloc[state] = T_row

    T = fix_T_for_unseen_states(T)

    return T.to_numpy()


def get_particular_state_transition(state, action):
    # add loop for action to return a matrix of T_row
    if state < 1764:  # serve states
        serve_type = check_serve_ad_vs_deuce(state)

        if serve_type == 'ad':
            T_row = T_ad_serve_set[action].iloc[state]
        elif serve_type == 'deuce':
            T_row = T_deuce_serve_set[action].iloc[state]

    elif state >= 1764:  # rally states
        T_row = T_rally_set[action].iloc[state]

    return T_row.to_numpy()


def get_probabilistic_transition_matrix(q_ad_serve, q_deuce_serve, q_rally):

    T = pd.DataFrame(np.zeros((5297, 5297)))

    for state in range(5292):
        T.iloc[state] = get_particular_probabilistic_transition(state, q_ad_serve, q_deuce_serve, q_rally)

    T = fix_T_for_unseen_states(T)

    return T.to_numpy()


def get_particular_probabilistic_transition(state, q_ad_serve, q_deuce_serve, q_rally):
    T_row = pd.Series(np.zeros(5297))

    if state < 1764:  # serve states
        serve_type = check_serve_ad_vs_deuce(state)
        if serve_type == 'ad':

            for i in range(q_ad_serve.shape[1]):
                T_row += q_ad_serve.iloc[state, i] * T_ad_serve_set[i].iloc[state]

        elif serve_type == 'deuce':

            for i in range(q_deuce_serve.shape[1]):
                T_row += q_deuce_serve.iloc[state, i] * T_deuce_serve_set[i].iloc[state]


    elif state >= 1764:  # rally states

        for i in range(q_rally.shape[1]):
            T_row += q_rally.iloc[state, i] * T_rally_set[i+2].iloc[state]

    return T_row.to_numpy()


def fix_T_for_unseen_states(T):
    # for states we enter, but have no observations, create a uniform dist (include scoring states)
    states_leave = T.sum(axis=1)
    states_enter = T.sum(axis=0)

    ind_no_leave = states_leave == 0
    ind_enter = states_enter > 0.000001

    ind_enter_but_no_leave = ind_enter & ind_no_leave
    T.loc[ind_enter_but_no_leave, 3528:5296] = 1 / 1769

    return T


def check_serve_ad_vs_deuce(state):
    check = math.floor(state / 126)
    if check % 2 == 0:  # even
        serve_type = 'deuce'
    else:
        serve_type = 'ad'
    return serve_type

def check_forehand_vs_backhand(state):
    check = math.floor((state - 1764) / 42)
    if (check % 6) < 4:  # even
        shot_type = 'forehand'
    else:
        shot_type = 'backhand'
    return shot_type

def calculate_V(T):
    Vp = T[0:5292, 5293] * 1 + T[0:5292, 5294] * 1
    Vp = np.reshape(Vp, (-1))

    # Value Iteration
    # V_old = np.zeros((5292))
    # for i in range(200):
    #     V_new = T[0:5292, 0:5292].dot(V_old) + Vp
    #     if all((abs(V_new - V_old)) < 1e-6):
    #         print('converged')
    #         break
    #     V_old = V_new

    # Solve SOE with A_pinv
    # eye = pd.DataFrame(np.identity(5292))
    # A = eye - T[0:5292, 0:5292]
    # A = A.to_numpy()
    # A_pinv = np.linalg.pinv(A)  # Solve V using lease squares with the pseudo-inverse
    # V = A_pinv.dot(Vp)

    # Solve SOE Directly
    eye = pd.DataFrame(np.identity(5292))
    A = eye - T[0:5292, 0:5292]
    A = A.to_numpy()
    V = np.linalg.solve(A, Vp)

    return V

def get_V_measures(V,count, V_base):
    idx_avg = count > 0
    idx_top_avg = count > 30
    # serve states with most observations
    arr = [364, 370, 376, 616, 622, 628, 403, 409, 415, 655, 661, 667]

    V_sum = V.sum()
    V_avg = V[idx_avg.iloc[0:5292, 0]].mean()
    V_top_avg = V[idx_top_avg.iloc[0:5292, 0]].mean()
    V_serve = V[arr].mean()
    V_count_worse = np.sum(V < V_base)

    return V_sum, V_avg, V_top_avg, V_serve, V_count_worse

def change_action(action_current):
    serve_actions = [0, 1]
    rally_actions = [2, 3, 4, 5, 6, 7]

    if action_current <= 1:
        serve_actions.remove(action_current)
        action_new = serve_actions
    else:
        rally_actions.remove(action_current)
        action_new = rally_actions

    return np.array(action_new)


def get_best_V_and_action(state, action_new, V):
    V_new_other_actions = pd.DataFrame(np.zeros((action_new.size, 1)), columns=['V_new']).set_index(action_new)

    for action in action_new:
        T_row_new = get_particular_state_transition(state, action)
        V_new_other_actions.loc[action] = T_row_new[0:5292].dot(V) + T_row_new[5293] * 1 + T_row_new[5294] * 1

    V_new_best = V_new_other_actions.max()[0]
    action_new_best = V_new_other_actions.idxmax()[0]

    return V_new_best, action_new_best


def get_opt_policy_counts(opt_policy):
    counts = np.zeros((5292, 1))

    for state in range(5292):

        action = opt_policy[state][0]

        if state < 1764:  # serve states
            serve_type = check_serve_ad_vs_deuce(state)

            if serve_type == 'ad':
                counts[state] = C_ad_serve_set[action].iloc[state]
            elif serve_type == 'deuce':
                counts[state] = C_deuce_serve_set[action].iloc[state]

        elif state >= 1764:  # rally states
            counts[state] = C_rally_set[action].iloc[state]

    return counts


def solve_policy_iteration(opt_policy, max_iterations=50, eps=1e-6, n_states=5292):
    for i in range(max_iterations):
        change = 0

        # get transition matrix for this current policy
        T = get_policy_specific_transition_matrix(opt_policy)

        # calculate V's (aka win absorption probs)
        V = calculate_V(T)

        for state in range(n_states):
            # current V (win prob) and action
            V_current = V[state]
            action_current = opt_policy[state]

            # get alternate actions
            action_new = change_action(action_current)

            # get best V_new and action from all other alternate actions
            V_new_best, action_new_best = get_best_V_and_action(state, action_new, V)

            # # update opt_policy if new action is better
            if (V_new_best > (V_current + eps)):
                opt_policy[state] = action_new_best
                change = 1

            # update opt_policy if new action is better
            # if action_current != action_new_best:
            #     opt_policy[state] = action_new_best
            #     change = 1

        if change == 0:
            break  # change equals 0 => optimal policy found

    # warning if the solution did not converge
    if i > max_iterations & change == 1:
        print('Warning: Reached max iterations: {}.'.format(max_iterations))
    else:
        print('Optimal policy found in {} iterations.'.format(i))

    # Format the final results
    counts = get_opt_policy_counts(opt_policy)
    counts_full = dm.reformatVector(counts)
    V_full = dm.reformatVector(np.reshape(V, (-1, 1)))
    opt_policy_full = dm.reformatVector(opt_policy)

    return V_full, opt_policy_full, counts_full, i
