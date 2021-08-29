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
In our experiment, we have m bidders and k items, where m > k, and generate the bidder values for each round using a given distribution (e.g. F ∼ U[0, 1]). We then create a data set to learn from, using the action-space consists of reserve prices R<sup>i</sup>; R∈ [0, h], where h is the highest payoff. For each round, we calculate the revenue based on the bidder’s values and the reserve price and enter it into the data set. We then use the exponential weights online learning algorithm on this data set to find the converged action (average reserve price) and the corresponding converged revenue (average payoff for each round). We can compare these values to the theoretical optimal reserve price given by <sup>0</sup>(0) and the theoretical optimal revenue given by E[φ], where φ(v) = v =1−F(v)f(v). We also looked at what would happen tothese converged values as we varied the number of bidders or items by using the same set upas before but running the experiment with different revenue or value generations and then lotting the values we get from our online learning algorithm.

