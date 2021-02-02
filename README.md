# rtb-price-predictor-app
A web app that predicts a reserve CPM price for Publishers.

### Background
This project is aimed to design a prediction system that will predict the cost per impression (CPM) or reserve price an advertiser should pay for an ad slot on a publisher's website. An algorithm and prediction model will be used in the project. The system will be used by the Publisher to identify a `pred_CPM` to better forecast how much their ad inventory is worth.

Real-time bidding (RTB) refers to programmatic advertising or the online auction process where buying and selling of online ad impressions are done in real-time and often facilitated through an ad exchange. This happens when a user lands on a website, the bidders (advertisers) bid for different ad slots on the page and the one with the highest winning bid displays their ad in the ad space and pays the amount he/she bid to the exchange.

### Problem Statement

The process of RTB encourages bid shading - the practice of bidding lesser than the perceived value of the ad space to maximize utilization for the advertiser while maintaining a particular win rate at lowest prices. 

As a result, Publishers want to value their inventory correctly so that they can set a reserve price for their ad space (or minimum price can be set up in the auctions.) and Advertisers want to pay the least amount possible to have their ads shown. Setting a reserve price causes bidders to lose at lower bids and encourages higher bidding which translates to higher revenue for the publisher.

### Hypothesis
Using regression, we can predict the reserve price an advertiser would have to bid to win an auction for their ad to be served.

### Risk & Assumptions
- There is no datetime, so we only have data aggregated to the day. There are multiple records per day and there is no way we can tell what time of day each record represents. This could potentially skew the data and effect the results. For instance, there could be higher volume of traffic volumes during certain times of the day.
- The data is only of the actual revenue generation and not at bid level. 
- There are many records with impressions but $0 revenue. 

### Goals & Success Metrics
The goal of this dataset is to predict the base price an advertiser would need to pay to win an auction for their ad to be served on a publisher's website. The dataset provided has data for several website owned by the same company and they are asking what the reserve price should be for June and what the range for reserve prices should be setting for July.

### Data Sources
The advertising data can be found on kaggle: https://www.kaggle.com/saurav9786/real-time-advertisers-auction

### Tools
- ML Enviornment: Google Collab : https://colab.research.google.com/
- Models: Scikit-learn: https://scikit-learn.org/stable/
- Web App: Streamlit: https://www.streamlit.io/
- Cloud Application Platform: https://www.heroku.com/

### Notes
- Publisher – person who owns and publishes content on the website
- Inventory – all the users that visit the website * all the ad slots present in the website for the observation period
- Impressions - showing an ad to a user constitutes one impression. If the ad slot is present but an ad is not shown, it falls as “unfilled impression”. Inventory is the sum of impressions + unfilled impressions.
- CPM – cost per Mille. This is one of the most important ways to measure performance. It is. Calculated as revenue/impressions * 1000. 'bids' and 'price' are measured in terms of CPM
