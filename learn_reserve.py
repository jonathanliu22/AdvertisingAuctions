import numpy as np


class Reserve(object):

    def uniform_dist(self):
        return np.random.uniform(low=0.0, high=1.0)

    # will add 1-2 more distributions...

    def generate_bidder_values(self, m, n, dist):
        """
        :param m: number of bidders
        :param n: number of rounds (best to use many to ensure reserve price is found)
        :param dist: function of F distribution to use to generate values

        :return bidder_values: array of bidder values [i][j]; ith bidder, jth round
        """
        bidder_values = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                bidder_values[i][j] = dist()
        return bidder_values


    def revenue_from_reserve(self, bidder_values, num_items):
        """
        :param bidder_values: array of bidder values [i][j]; ith bidder, jth round
        :param num_items: number of items for sale each round (e.g. selling 1 item with truthful 2nd price auction)

        :return revenue: array of revenue outcomes given various reserve price options in a truthful auction
                reserve prices are values from 0 to 1, [i][j]; ith reserve price, jth round
        """
        num_rounds = bidder_values.shape[1]
        reserve_prices = np.linspace(0.0, 1.0, num=101)
        revenue = np.zeros((101, num_rounds))

        for r in range(num_rounds):
            bids = np.sort(bidder_values[:, r])
            for p in range(101):
                if bids[-1*(num_items)] < reserve_prices[p]:
                    revenue[p][r] = 0
                else:
                    rev = 0
                    for i in bids[-1*(num_items+1):-1]:
                        if i > reserve_prices[p]:
                            rev += i
                        else:
                            rev += reserve_prices[p]
                    revenue[p][r] = rev
        return revenue

    def reserve_exponential_weights(self, action_payoff_table, eps):
        """
        :param action_payoff_table: table of payoffs for each action for each round
                (in this case, payoff is revenue and actions are reserve prices)
        :param eps: learning rate

        :return reserve_history: history of reserve prices chosen at each round
        :return revenue_history: history of total revenue of learning algorithm after each round
        """
        k = action_payoff_table.shape[0]
        n = action_payoff_table.shape[1]

        prob_weights = np.zeros((k, n))
        vals = np.zeros((k, ))
        reserve_history = []
        revenue = 0
        revenue_history = []


        for round in range(n):
            num = np.zeros((k, ))
            for action in range(k):
                exp = vals[action]  # note that upper-bound is 1 by construction
                num[action] = (1 + eps) ** exp
            denom = np.sum(num)
            prob = np.true_divide(num, denom)

            for action in range(k):
                prob_weights[action, round] = prob[action]

            draw = np.random.choice(prob_weights[:, round], prob_weights[:, round].size, p=prob_weights[:, round])
            draw_prob = draw[0]
            index_locations = np.where(prob_weights[:, round] == draw_prob)[0]
            choose_action = np.random.choice(index_locations)
            reserve_price = 0.01 * choose_action

            reserve_history.append(reserve_price)
            revenue += action_payoff_table[choose_action][round]
            revenue_history.append(revenue)

            vals = np.add(vals, action_payoff_table[:, round])

        return reserve_history, revenue_history







res = Reserve()

bids = res.generate_bidder_values(2, 1000, res.uniform_dist)
rev = res.revenue_from_reserve(bids, 1)

reserve, revenue = res.reserve_exponential_weights(rev, 0.2)

print(reserve)

