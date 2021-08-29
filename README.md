# AdvertisingAuctions

I simulated a truthful second-price auction using online learning algorithms to optimize the reserve price and revenue. 

### What are Online Learning Algorithms?
Online Learning Algorithms are a method of machine learning in which data becomes available in a sequential order and is used to update the best predictor for future data at each step. There are various algorithms such as Exponential Weights and Follow the Leader as well as variations of each.

### Exponential Weights Algorithm
This algorithm keeps track of a probability distribution over actions that is updated in each round of the prediction task by multiplying the probability of each action by a factor that is exponentially decreasing in the action’s error or loss in that round, and renormalizing.

### Second-Price Auction
This is a game theory model where bidders submit written bids without knowing the bid of the other people in the auction. The highest bidder wins but the price paid is the second-highest bid. This auction is truthful if bidders only bid their percieved value of the item.

### So What?
Exploring this topic is important since auctions are commonly seen in online advertising such as those run by Google or Facebook. These method informs which advertiser's ad shows up and the cost of it. Online learning algorithms can help bidders understand what to bid as well as inform the company on how to set up the auction to maximize revenue.

## The Simulation
In our experiment, we have m bidders and k items, where m > k, and generate the bidder values for each round using a given distribution (e.g. F ∼ U[0, 1]). We then create a data set to learn from, using the action-space consists of reserve prices R<sub>i</sub>; R∈ [0, h], where h is the highest payoff. For each round, we calculate the revenue based on the bidder’s values and the reserve price and enter it into the data set. We then use the exponential weights online learning algorithm on this data set to find the converged action (average reserve price) and the corresponding converged revenue (average payoff for each round). We can compare these values to the theoretical optimal reserve price given by φ<sub>0</sub>(0) and the theoretical optimal revenue given by E[φ], where (v) = v =1−F(v)f(v). We also looked at what would happen tothese converged values as we varied the number of bidders or items by using the same set upas before but running the experiment with different revenue or value generations and then lotting the values we get from our online learning algorithm.

## Findings
As we increase the number of bidders, the reserve price remains constant, while the revenue increases, eventually converging to 1. This makes sense since having more bidders will not affect the reserve price, while increasing the chance of having high bids and thus increasing the potential revenue. As we increase the number of items, the reserve price also remains constant, while the revenue increase at a constant rate. Even in different scenarios such as selling introductions, we found out that we were able to use exponential weights algorithm to quickly determine the optimal mechanism, even when the players had different distributions. However, in the real world, there’s not only one employer and one employee. Something that could be explored further is whether our algorithm will still perform optimally when we add more players to each side.

