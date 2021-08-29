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



