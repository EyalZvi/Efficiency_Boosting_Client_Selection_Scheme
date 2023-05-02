import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import random
import seaborn as sns
from operator import add
np.random.seed(1110112)


# Write client selection choices into a logfile
f = open("logfile.txt", "a")


def FedCS3(est_exchange_times, expected_num_of_arms, arm_indicators):
    """
    This function implements the FedCS(3) algorithm from the previous paper:
    Client Selection for Federated Learning with Heterogeneous Resources in Mobile Edge.
    A client selection algorithm for federated learning that will be used as a test case.
    This algorithm is motivated by minimizing exchange times.

    :param est_exchange_times: List of estimated exchange times of clients in a federated round.
    :param expected_num_of_arms: Optimal number of client for a federated round.
    :param arm_indicators: List of indicator values showing client's availability for a federated round.

    :return: List of indicators representing chosen clients for a federated round
    """

    # ---------------------------------------
    # Initializing parameters:
    S = []
    t = 0
    K = []
    est = est_exchange_times.copy()
    round_est_times = []

    for k in range(len(arm_indicators)):
        if arm_indicators[k] == 1:
            K.append(k+1)
            round_est_times.append(est[k])

    # ---------------------------------------

    # Algorithm implementation:
    # Note: We first check a selection is needed
    if len(K) <= expected_num_of_arms:
        return arm_indicators
    else:
        while len(K) > 0:
            x = int(np.argmax(np.multiply(-1, round_est_times)))
            val = K[x]
            K.pop(x)
            t_tag = t + round_est_times[x]
            round_est_times.pop(x)

            if len(S) < expected_num_of_arms:
                t = t_tag
                S.append(val)

    return [1 if i+1 in S else 0 for i in range(len(arm_indicators))]

    # ---------------------------------------


def random_choice(expected_num_of_arms, arm_indicators):
    """
    A  random client selection algorithm for federated learning that will be used as a test case.
    This algorithm is chooses clients at random from a list of available clients.

    :param expected_num_of_arms: Optimal number of client for a federated round.
    :param arm_indicators: List of indicator values showing client's availability for a federated round.

    :return: List of indicators representing chosen clients for a federated round.
    """
    # ---------------------------------------
    if sum(arm_indicators) <= expected_num_of_arms:
        return arm_indicators
    else:
        indices = random.sample([index for index in range(len(arm_indicators))], expected_num_of_arms)
        return [1 if i in indices else 0 for i in range(len(arm_indicators))]

    # ---------------------------------------


def calcP4(S_n_max, queue_lens, length, V, Tau_t):
    """
    This is a sub-function of Algorithm 1: Divide-and-Conquer.
    The purpose of this function is to calculate the value of the objective function in the.
    P4 integer linear programming problem, given a set S_n_max.
    Note: This function only calculates values, not the solution for P4.

    :param S_n_max: One of the possible sets of clients to chose.
    :param queue_lens: Length of virtual queues in round t.
    :param length: Number of available clients (in the set).
    :param V: Fairness factor of the RBCS-F algorithm.
    :param Tau_t: Estimated exchange times of all clients.

    :return: A possible solution for the P4 ILP problem, client selection.
    """
    # ---------------------------------------
    x_t = [1 if i+1 in S_n_max else 0 for i in range(0, length)]
    factor = max(np.multiply(x_t, Tau_t).tolist())
    return V * factor - np.dot(x_t, queue_lens)

    # ---------------------------------------


def divide_and_conquer(est_exchange_times, expected_num_of_arms, arm_indicators, queue_lens, V):
    """
    Algorithm 1: Divide-and-Conquer.
    The purpose of this function is to solve the P4 integer linear programming problem for round t.
    Each possible set of available clients is tested in order to find the set that achieves the minimal
    value of the P4 objective function while maintaining the needed constraints.

    :param est_exchange_times: List of estimated exchange times of clients in a federated round.
           Will be refered to as Tau_t in this function
    :param expected_num_of_arms: Optimal number of client for a federated round.
    :param arm_indicators: List of indicator values showing client's availability for a federated round.
    :param queue_lens: Length of virtual queues in round t.
    :param V: Fairness factor of the RBCS-F algorithm.

    :return: The solution for P4 in round t.
    """

    # ---------------------------------------
    # Initializing parameters:

    # N_t: List of available clients
    # Z_t: List of queue length for available clients
    # A_t: List of available clients ordered with decending order of queue lengths
    # k: Number of clients to be chosen

    set_dict = {}
    N_t = [i+1 for i in range(len(arm_indicators)) if arm_indicators[i] == 1]
    Z_t = [queue_lens[i] for i in range(len(arm_indicators)) if arm_indicators[i] == 1]
    A_t = [n for _, n in sorted(zip(Z_t, N_t), reverse=True)]
    k = min(expected_num_of_arms, sum(arm_indicators))
    Tau_t = est_exchange_times

    # ---------------------------------------
    # Algorithm implementation:

    # Build all possible sets and collect their P4 values
    for n_max in N_t:
        S_n_max = []
        for n in A_t:
            if Tau_t[n-1] <= Tau_t[n_max - 1]:
                S_n_max.append(n)
            if len(S_n_max) == k:
                F_n_max = calcP4(S_n_max, queue_lens, len(arm_indicators), V, Tau_t)
                set_dict[n_max] = (S_n_max, F_n_max)
                break
    Fmin = np.infty

    # Obtain the minimal value of the objective function and return the optimal set
    S_chosen = 0
    for key in set_dict.keys():
        tmpS, tmpF = set_dict[key]
        if tmpF < Fmin:
            Fmin = tmpF
            S_chosen = tmpS

    if type(S_chosen) == list:
        return [1 if i+1 in S_chosen else 0 for i in range(0, len(arm_indicators))]
    else:
        return [0] * len(arm_indicators)

    # ---------------------------------------


def RBCS_F(expected_num_of_arms, exploration_vector, client_set,
           ridge_lambda, beta, balance_param, total_time, algo):
    """
    Algorithm 2: Reputation Based Client Selection With Fairness.
    The purpose of this function is to optimaly select clients for a federated round in real-time manner.
    RBCS-F strives to minimize exchange time while ensuring no violation in the long-term fairness constraint.
    Different balance parameters (V) are tested and compaired in this simulation

    :param expected_num_of_arms: Optimal number of client for a federated round.
    :param exploration_vector: List of exploration factor for each t.
    :param client_set: Set of clients who are willing to participate in the federated learning.
    :param ridge_lambda: Ridge regression constatnt that aids estimating exchange times based on historical performance.
    :param beta: Lower bound of expected guaranteed selection rate, estimate queue length in real-time (Lyapunov).
    :param balance_param: (V) Fairness factor.
    :param total_time: Number of training (federated) rounds
    :param algo: Choose algorithm to simulate (for teseting purposes) RBCS_F/Random/FedCS3

    :return: The control policy (list of client selections) for all training rounds
    """

    # ---------------------------------------
    # Initializing parameters:

    # m: Expected number of arms
    # alpha: List (Vector) of exploration parameters
    # N: Set of clients
    # Lambda: Ridge regression constant
    # Beta: Lower bound of expected guaranteed selection rate (beta)
    # V: Fairness factor

    m = expected_num_of_arms
    alpha = exploration_vector
    N = client_set
    Lambda = ridge_lambda
    Beta = beta
    V = balance_param

    # ---------------------------------------
    # Initializing matrices and vectors:

    # H: List of matrices for the UCB algorithm
    # b: List of vectors for the UCB algorithm
    # z: List of queue lengths (vectors)
    # I: List of client availability lists (vectors)

    H = []
    b = []
    z = []
    I = [[0] * len(N)]

    # ---------------------------------------
    # Initializing contextual vector, model size and RVs for (real) exchange time simulation after trainning:

    # c: Contextual features vector
    # Mu: Available CPU ratio of the client
    # s: Indicator for participation in last round
    # B: Allocated bandwidth
    # M: Size of model's parameters

    c = [[[0] * 3] * len(N)]
    Mu = [[0] * len(N)]
    s = [[0] * len(N)]
    B = [[0] * len(N)]
    M = 2 * 2**20

    # ---------------------------------------
    # Initializing key parameters:

    # x: Client selection list
    # Tau_estimated: Estimated exchange time
    # Tau_avg: Estimated average exchange time
    # Tau_real: Estimated (real) exchange time after trainning
    # Theta: Static coefficient factors unknown to the scheduler

    x = [[0] * len(N)]
    Tau_estimated = [[0] * len(N)]
    Tau_avg = [[0] * len(N)]
    Tau_real = [[0] * len(N)]
    Theta = [[0] * len(N)]

    for _ in range(len(N)):
        H.append([[Lambda, 0, 0], [0, Lambda, 0], [0, 0, Lambda]])
        b.append([[0.0], [0.0], [0.0]])
        z.append(0)

    # ---------------------------------------
    # Write to logfile

    if algo == 'RBCS_F':
        L = 'Algorithm: ' + "RBCS-F" + "(" + str(V) + ")"
    elif algo == 'FedCS3':
        L = 'Algorithm: FedCS(3)'
    else:
        L = 'Algorithm: random'
    print(f'- Testing {L}\n...')
    f.write(L)
    f.write("\n\n")

    # ---------------------------------------
    # For loop: Iterate through all rounds (should be real-time)

    for t in range(1, total_time+1):
        # ---------------------------------------
        # Generate random variables:

        Mu.append(np.random.uniform(0.5, 2, len(N)))
        s.append(x[-1])
        B.append(np.random.uniform(2000000, 4000000, len(N)).tolist())

        c.append([np.transpose([1/Mu[t][i], s[t][i], M/B[t][i]]).tolist() for i in range(len(N))])
        I.append([1 if prob <= 0.8 else 0 for prob in np.random.random(size=len(N)).tolist()])

        # ---------------------------------------
        # Intialize temporary lists

        Theta_tmp = []
        Tau_est_tmp = []
        Tau_avg_tmp = []

        # ---------------------------------------
        # Calculate current estimated average exchange time (and others) and add it to the list

        for n in N:
            if t == 1:
                Theta_tmp.append(np.matmul(inv(H[n-1]), np.matrix(b[n-1])).tolist())
                Tau_est_tmp.append((np.matmul(np.transpose(c[t][n-1]), Theta_tmp[n-1]))[0])
                tmp = np.sqrt(np.matmul(np.matmul(c[t][n-1], inv(H[n-1])), np.transpose(c[t][n-1])))
                Tau_avg_tmp.append(max(Tau_est_tmp[n-1] - alpha[t] * tmp, 0))

            else:
                Theta_tmp.append(np.matmul(inv(H[t-1][n-1]), np.matrix(b[t-1][n-1])).tolist())
                Tau_est_tmp.append((np.matmul(np.transpose(c[t][n-1]), Theta_tmp[n-1]))[0])
                tmp = np.sqrt(np.matmul(np.matmul(c[t][n-1], inv(H[t-1][n-1])), np.transpose(c[t][n-1])))
                Tau_avg_tmp.append(max(Tau_est_tmp[n-1] - alpha[t] * tmp, 0))

        if t == 1:
            Theta = [Theta, Theta_tmp]
            Tau_estimated = [Tau_estimated, Tau_est_tmp]
            Tau_avg = [Tau_avg, Tau_avg_tmp]

        else:
            Theta.append(Theta_tmp)
            Tau_estimated.append(Tau_est_tmp)
            Tau_avg.append(Tau_avg_tmp)

        # ---------------------------------------
        # Obtain client selction list for round t according to one of three algorithms and write to logfile
        # RBCS_F/Random/FedCS(3)

        if algo == 'RBCS_F':
            if t == 1:
                x.append(divide_and_conquer(Tau_avg[t], m, I[t], z, V))
            else:
                x.append(divide_and_conquer(Tau_avg[t], m, I[t], z[t - 1], V))
        if algo == 'Random':
            x.append(random_choice(m, I[t]))
        if algo == 'FedCS3':
            x.append(FedCS3(Tau_avg[t], m, I[t]))

        L = "Client selection for training round: " + str(t) + "\n" + str(x[-1]) + "\n"
        f.write(L)
        f.write("\n")

        # ---------------------------------------
        # Calculate current estimated (real) exchange time and add it to the list

        # Theta star is chosen according to inherent setting of clients
        # Clients are divided to four groups of 10 by their estimated effiencies

        Tau_tmp = []

        for n in N:
            if n <= 10:
                Theta_star = [1, 1, 1/np.log(1 + 1000)]
            elif n <= 20:
                Theta_star = [2, 1, 1/np.log(1 + 100)]
            elif n <= 30:
                Theta_star = [3, 1, 1/np.log(1 + 10)]
            else:
                Theta_star = [4, 1, 1/np.log(1 + 1)]

            time_dist = np.matmul(np.transpose(c[t][n-1]), Theta_star)
            Tau_tmp.append(time_dist + np.random.uniform(-time_dist, time_dist))

        Tau_real.append(Tau_tmp)
        # ---------------------------------------
        # Update key parameters for next round

        z_tmp = []
        H_tmp = []
        b_tmp = []

        for n in N:
            if t == 1:
                z_tmp.append(max(0, z[n-1] + Beta - x[t][n-1]))
                H_tmp.append((np.array(H[n-1]) +
                              x[t][n-1] * np.matmul([c[t][n - 1]], np.transpose([c[t][n - 1]]))).tolist())
                b_tmp.append((np.array(b[n-1]) +
                              x[t][n-1] * Tau_real[t - 1][n - 1] * np.transpose([c[t][n - 1]])).tolist())

            else:
                z_tmp.append(max(0, z[t-1][n-1] + Beta - x[t][n-1]))
                H_tmp.append((np.array(H[t-1][n-1]) +
                              x[t][n-1] * np.matmul([c[t][n - 1]], np.transpose([c[t][n - 1]]))).tolist())
                b_tmp.append((np.array(b[t-1][n-1]) +
                              x[t][n-1] * Tau_real[t - 1][n - 1] * np.transpose([c[t][n - 1]])).tolist())

        if t == 1:
            z = [z, z_tmp]
            H = [H, H_tmp]
            b = [b, b_tmp]
        else:
            z.append(z_tmp)
            H.append(H_tmp)
            b.append(b_tmp)
        # ---------------------------------------
    # Return control policy (and other values)

    return x, z, np.multiply(Tau_real, x).tolist()


def generate_group(g1, g2, num_groups=40, group_size=10, range_max=400):
    """
    This function generates random sub-groups of size 10 from the integer field : [1,400].
    Each client recieves 10 different numbers representing "new data" that is feeded to the FL proccess.
    The accuracy of an FL epoch is later tested by the amount of unique numbers obtained divided by 400 (max. possible).

    :param g1: Concentration parameter controlling the extent of identicalness among clients
    :param g2: Another concentration parameter controlling the extent of identicalness among clients (given g1)
    :param num_groups: Number of clients (a constant of 40)
    :param group_size: Number of numbers to assign per sub-group (a constant of 10)
    :param range_max: Range of numbers to pick from (a constant representing [1-400])

    :return: A unique sub-list of 10 numbers
    """
    grps = []
    all_numbers = set()
    for i in range(num_groups):
        q = np.random.dirichlet(float(g1) * np.ones(range_max)) * float(g2)
        q = q / sum(q)
        group = np.random.choice(range(range_max), size=group_size, p=q)
        grps.append(group.tolist())
        all_numbers.update(group)
    return grps, list(all_numbers)


if __name__ == '__main__':
    """
    Main function: Collecting statistics and plotting simulation graphs
    
    :param T: Number of trainning rounds
    :param test: List of V values
    """
    T = 500
    test = [5, 10, 20, 50]

    # ---------------------------------------
    # Initializing key parameters:
    # Note: there are 40 potential clients in the simulation

    queuq_len_per_test = []
    queuqe_len_list = []
    Tau_max_per_test = []
    Tau_max_list = []
    num_clients_chosen = []
    clients_chosen_per_test = [0] * 40
    X_most_algo_sample = []

    # ---------------------------------------
    # Run simulation :

    X, Z_vec, Tau = RBCS_F(8, [0.1] * (T + 1), [i for i in range(1, 41)], 1, 0.15, 0, T, 'Random')
    X_most_algo_sample.append(X)
    for time in range(1, T + 1):
        clients_chosen_per_test = list(map(add, clients_chosen_per_test, X[time]))
        if time == 1:
            Tau_max_per_test.append(max(Tau[time]))
        else:
            Tau_max_per_test.append(Tau_max_per_test[-1] + max(Tau[time]))
    num_clients_chosen.append(sorted(clients_chosen_per_test))
    Tau_max_list.append(Tau_max_per_test)

    for V in test:
        X, Z_vec, Tau = RBCS_F(8, [0.1] * (T+1), [i for i in range(1, 41)], 1, 0.15, V, T, 'RBCS_F')
        if V != 10:
            X_most_algo_sample.append(X)
        queuq_len_per_test = []
        Tau_max_per_test = []
        clients_chosen_per_test = [0] * 40
        for time in range(1, T+1):
            queuq_len_per_test.append(sum(Z_vec[time]))
            if V != 10:
                if time == 1:
                    Tau_max_per_test.append(max(Tau[time]))
                else:
                    Tau_max_per_test.append(Tau_max_per_test[-1] + max(Tau[time]))
                clients_chosen_per_test = list(map(add, clients_chosen_per_test, X[time]))

        queuqe_len_list.append(queuq_len_per_test)
        if V != 10:
            num_clients_chosen.append(sorted(clients_chosen_per_test))
            Tau_max_list.append(Tau_max_per_test)

    X, Z_vec, Tau = RBCS_F(8, [0.1] * (T + 1), [i for i in range(1, 41)], 1, 0.15, 0, T, 'FedCS3')
    X_most_algo_sample.append(X)
    queuq_len_per_test = []
    Tau_max_per_test = []
    for time in range(1, T + 1):
        clients_chosen_per_test = list(map(add, clients_chosen_per_test, X[time]))
        if time == 1:
            Tau_max_per_test.append(max(Tau[time]))
        else:
            Tau_max_per_test.append(Tau_max_per_test[-1] + max(Tau[time]))
    num_clients_chosen.append(sorted(clients_chosen_per_test))
    Tau_max_list.append(Tau_max_per_test)
    print("Done!")

    # ---------------------------------------
    # Plot graphs:
    # 1) Pull record of arms under different client-selection strategies
    # 2) The impact of V on the convergence of the queues
    # 3) Training time of different client-selection strategies

    fig_num = 1
    plt.figure(fig_num)
    fig_num += 1
    plt.title("Pull record of arms under different client-selection strategies")

    ylabels = ['random', 'RBCS-F(5)', 'RBCS-F(20)', 'RBCS-F(50)', 'FedCS(3)']
    xlabels = [i*2 for i in range(1, 21)]
    xticks = [i for i in range(20)]
    plt.xticks(xticks, xlabels, rotation=0)
    heatmap = sns.heatmap(num_clients_chosen, yticklabels=ylabels, vmin=0, vmax=400, cmap='coolwarm')
    plt.ylabel('Selection Methods')
    plt.xlabel('Clients')
    plt.tight_layout()
    plt.show(block=False)

    plt.figure(fig_num)
    fig_num += 1
    for plotGraph in queuqe_len_list:
        plt.plot(plotGraph)

    plt.xticks(range(0, T+1, 100))
    plt.ylabel('Total Queue Length')
    plt.xlabel('Federated Rounds')
    plt.title("The impact of V on the convergence of the queues")
    plt.legend(['RBCS-F(5)', 'RBCS-F(10)', 'RBCS-F(20)', 'RBCS-F(50)'])
    plt.show(block=False)

    plt.figure(fig_num)
    fig_num += 1
    for plotGraph in Tau_max_list:
        plt.plot(plotGraph)

    plt.xticks(range(0, T+1, 100))
    plt.ylabel('Time Consumption(s)')
    plt.xlabel('Federated Rounds')
    plt.title("Training time of different client-selection strategies")
    plt.legend(['random', 'RBCS-F(5)', 'RBCS-F(20)', 'RBCS-F(50)', 'FedCS(3)'])
    plt.show()

    # ---------------------------------------
    # Plot graphs:
    # 1) Fairness Impact With γ1 = 1
    # 2) Fairness Impact With γ1 = 10
    # 3) Fairness Impact With γ1 → ∞

    x_for_V_eq_5 = X_most_algo_sample[1]
    for g1_values in [1, 10, 10 ** 100]:
        plt.figure(fig_num)
        fig_num += 1
        for g2_values in [0.1, 0.5, 1, 10]:
            accuracy = []
            groups, numbers_chosen = generate_group(g1_values, g2_values)
            tmp_set = set()
            for i in range(1, len(x_for_V_eq_5)):
                for j in range(len(x_for_V_eq_5[i])):
                    if x_for_V_eq_5[i][j] == 1:
                        tmp_set.add(np.random.choice(groups[j]))

                # Accuracy is tested by amount of unique numbers obtained
                accuracy.append((len(tmp_set) / 400) * 100)
            plt.plot(accuracy)
        plt.ylabel('Test accuracy')
        plt.xlabel('Federated Rounds')
        if g1_values == 1 or g1_values == 10:
            string = "Fairness Impact With γ1 = " + str(g1_values)
        else:
            string = "Fairness Impact With γ1 → ∞"

        plt.title(string)
        plt.yticks(range(0, 80, 10))
        plt.legend(["γ2 = 0.1", "γ2 = 0.5", "γ2 = 1", "γ2 = 10"])
        if g1_values != 10 ** 100:
            plt.show(block=False)
        else:
            plt.show()

    # ---------------------------------------
    # Plot graphs:
    # 1) Accuracy Vs. Federated Rounds With γ1 = 1
    # 2) Accuracy Vs. Federated Rounds With γ1 = 10
    # 3) Accuracy Vs. Federated Rounds With γ1 → ∞

    for g1_values in [1, 10, 10 ** 100]:
        plt.figure(fig_num)
        fig_num += 1
        for x_value in X_most_algo_sample:
            accuracy = []
            groups, numbers_chosen = generate_group(g1_values, 1)
            tmp_set = set()
            for i in range(1, len(x_value)):
                for j in range(len(x_value[i])):
                    if x_value[i][j] == 1:
                        tmp_set.add(np.random.choice(groups[j]))

                # Accuracy is tested by amount of unique numbers obtained
                accuracy.append((len(tmp_set) / 400) * 100)
            plt.plot(accuracy)
        plt.ylabel('Test accuracy')
        plt.xlabel('Federated Rounds')
        if g1_values == 1 or g1_values == 10:
            string = "Accuracy Vs. Federated Rounds With γ1 = " + str(g1_values)
        else:
            string = "Accuracy Vs. Federated Rounds With γ1 → ∞"

        plt.title(string)
        plt.yticks(range(0, 80, 10))
        plt.legend(['random', 'RBCS-F(5)', 'RBCS-F(20)', 'RBCS-F(50)', 'FedCS(3)'])
        if g1_values != 10 ** 100:
            plt.show(block=False)
        else:
            plt.show()

# Close logfile
f.close()
