# Fashion Booster

## What is Fashion Booster?
Fashion Booster is a consulting project that I did for a fashion E-commerce
startup.
 My client company provides an online marketplace,
 enabling women to lend or rent fashion items 
 including vacation bags, accessories, shoes, and clothes.
It is an *Airbnb of women's closets*. 

 This project provides a series of data-driven solutions for my client
 to improve their strategy.
 In particular, I performed extensive market analysis to estimate important metrics
 that investors care most about [Task 1: Market Analysis](#t1). 
 To help my client understand the profile of ideal customers, 
 I also built a highly interpretable model to predict whether a customer will convert 
 [Task 2: Predicting user conversion](#t2).
 Finally, I built a recommendation system for customized 
 newsletter delivery to increase their returning users
 [Task 3: Customized recommendation for users](#3).
 


## Task 1: Market Analysis

One of the most important metric related to financial modeling for startups
 is [payback period](http://www.portocapital.com/startup-financial-advice-metrics-payback-irr-gross-margin-porto-capital/),
 a metric evaluating the time span to recoup investments. This number can be estimated by 
 two other equally import metrics:
 
 - CAC, as you probably know, is the cost of convincing a potential customer to buy a product or service. 
 Here I defined it as the cost to acquiring an actual shopper (in other words, renter).
 - LTV is the projected revenue that a customer will generate during their lifetime. 
 Because for this newly born startup it's hard to estimate lifespan a renter. 
 Here I used a modified version *LTV per year*.
 
 ![market_analysis](img/task1/market_analysis.png)
 

The payback period (in months) is then defined as CAC, divided by LTV, and multiply by 12.

In this article, we will explain the CAC metric in more detail, how you can measure it, and what steps you can take to improve it.
https://blog.kissmetrics.com/customer-acquisition-cost/


First, I performed a series of market analysis using my client's database. 




## Task 2: Predicting user conversion

![feature_engineer](img/task2/feature_engineer.png)
![imbalanced_data](img/task2/imbalanced_data.png)
![confusion_matrix](img/task2/confusion_matrix.png)
![feature_importance](img/task2/feature_importance.png)




## Task 3: Customized recommendation for users

![brand](img/task3/brand_similarity.png)

![tokenizer](img/task3/example_tokenizer.png)

![description](img/task3/word2vec_description_similarity.png)






## What's Next
Recommendation systems make product recommendations to customers and 
achieve a lot of success in E-commerce.  In this project, I incorporated
both collaborative recommendation on brands and customized aspects
based on users' renting history. For a startup E-commerce at very
 early state, I expect this recommendation to help them increase the percentage
 of returning users.
 
 
In principle, I should validate 
my recommendation by separating the data set into training and testing, 
making the top-*N* recommendation, and checking the [F1 score](http://aimotion.blogspot.com/2011/05/evaluating-recommender-systems.html)
against different hyper-parameter settings. Although this might not
be feasible on my current data set due to the scarcity of 
shopping history, it would be very interesting to do this validation
when more data has veen accumulated.
