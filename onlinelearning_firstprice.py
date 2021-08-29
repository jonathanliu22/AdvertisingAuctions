
#(We show that the perturbed mechanism provably
#converges in the case of first-price auctions)

# buyer's values are fixed distribution U[0,1]
#



def new_calc_ftpl_pt_two(n, k, epsilon1, epsilon2 payoffs, h):
    hallucinations = h*np.around(np.random.geometric(epsilon,k))

    action_nxt = np.argmax(hallucinations)
    running_tot = hallucinations
    ult_payoffs = np.zeros(k)
    payoff = 0
    for rnd in range(n):
        #already picked action_nxt
        #print("round ", rnd+1, " and we picked action ", action_nxt)
        payoff += payoffs[action_nxt,rnd]
        best_action = -1
        best_payoff = -1
        for action in range(k):
            ult_payoffs[action] += payoffs[action,rnd]
            running_tot[action] += payoffs[action,rnd]
            if best_payoff < running_tot[action]:
                best_action = action
            best_payoff = running_tot[action]
        action_nxt = best_action
    actionz = np.arange(0, value, value/(k+1))[1:]
    #print("OPT: ", max(ult_payoffs), " with action ", np.argmax(ult_payoffs), " which was optimal bid ", actionz[np.argmax(ult_payoffs)])
    #print("ALG: ", payoff)
    return (max(ult_payoffs) - payoff) / n