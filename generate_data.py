import numpy as np, numpy.random
import random


def generate_bernoulli(rounds, actions):
    # we generate a table that is k by n
    data = np.zeros((actions, rounds))
    # this gives us our probabilities of the k actions to be used in Bernoulli; sums to 1
    prob_actions = np.random.random_sample((actions,))
    # fills in data using Bernoulli dist on generated probabilities
    for n in range(rounds):
        for k in range(actions):
            # returns 1 with probability = prob_actions[k]
            data[k][n] = int(random.random() < prob_actions[k])
    return data


def generate_uniform(rounds, actions):
    # we generate a table that is k by n
    data = np.zeros((actions, rounds))

    for n in range(rounds):
        # this gives us the values produced by a uniform random variable in [0,1]
        uniform_sample = np.random.random_sample((actions,))
        for k in range(actions):
            # fill in data from generated uniform dist
            data[k][n] = uniform_sample[k]
    return data


uniform84 = [[0.52523143, 0.06816666, 0.01862035, 0.07049225, 0.98007096, 0.83426685,
  0.08228063, 0.15409858],
 [0.99749652, 0.20665794, 0.5313938,  0.50427355, 0.06180704, 0.62328076,
  0.61891933, 0.3203466],
 [0.69969211, 0.63227044, 0.53635962, 0.55687499, 0.39685893, 0.92291347,
  0.68853226, 0.77038372],
 [0.70942483, 0.47256435, 0.2381702,  0.28094063, 0.99246134, 0.31944938,
  0.84078344, 0.16766838]]

bernoulli84 = [[1, 0, 1, 1, 1, 0,
  0 , 1],
 [0, 1, 0,  1, 0, 0,
  1, 0],
 [0, 0, 0, 0, 1, 1,
  0, 0],
 [1, 1, 1,  0, 0, 0,
  1, 0]]


def ew_empirical_opt_rate_regret(table):
    table = np.array(table)
    k = table.shape[0]
    n = table.shape[1]
    learning_rates = np.linspace(start=0, stop=1, num=100)
    regrets = []
    # first we find E[OPT]
    opt_total_payoff = 0
    for i in range(k):
        sum = 0
        for j in range(n):
            sum += table[i][j]
        if (sum > opt_total_payoff):
            opt_total_payoff = sum

    # now we find E[EW] using a range of learning rates and save them in a regrets vector
    for rate in range(len(learning_rates)):
        ew_total_payoff = 0
        for i in range(n):
            # first we find the probability of each action in round i
            prob_action = []
            for j in range(k):
                prob_action.append((1+learning_rates[rate]) ** np.sum(table[j][0:i-1]))
            total = np.sum(np.array(prob_action))
            prob_action = np.true_divide(prob_action, total)
            # we choose the action based on these probabilities
            action = numpy.random.choice(numpy.arange(0, k), p=prob_action)
            ew_total_payoff += table[action][i]
        # we add regret for each rate to list (regret = 1/n(E[OPT] - E[EW]))
        regrets.append((1/n)*(opt_total_payoff - ew_total_payoff))

    # now find the lowest regret in our vector and find the associated learning rate.
    min_regret = min(regrets)
    opt_learn_rate = regrets.index(min(regrets))*0.01
    return min_regret, opt_learn_rate, regrets[42]


def ftpl_empirical_opt_rate_regret(table):
    table = np.array(table)
    k = table.shape[0]
    n = table.shape[1]
    learning_rates = np.linspace(start=0, stop=.99, num=100)
    regrets = []
    # first we find E[OPT]
    opt_total_payoff = 0
    for i in range(k):
        sum = 0
        for j in range(n):
            sum += table[i][j]
        if (sum > opt_total_payoff):
            opt_total_payoff = sum
    # now we find E[FTPL] using a range of learning rates and save them in a regrets vector
    for rate in range(len(learning_rates)):
        ftpl_total_payoff = 0

        hallucinations = []
        for i in range(k):
            hallucinations.append(int(np.random.geometric(p=learning_rates[k], size=1)))
        # now we run the FTPL algorithm
        actions_sum = np.zeros((1, k))
        for action in range(len(actions_sum)):
            actions_sum[action] = hallucinations[action]
        for j in range(n):
            best_action = numpy.max(actions_sum)
            ftpl_total_payoff += best_action
            for i in range(k):
                actions_sum[0][i] += table[i][j]
        # now we calculate regrets
        regrets.append((1/n)*(opt_total_payoff - ftpl_total_payoff))
    min_regret = min(regrets)
    opt_learn_rate = regrets.index(min(regrets)) * 0.01
    return min_regret, opt_learn_rate


print(ftpl_empirical_opt_rate_regret(uniform84))










