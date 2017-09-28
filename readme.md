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
 
 - **CAC**, as you probably know, is the cost of convincing a potential customer to buy a product or service. 
 Here I defined it as the cost to acquiring an actual shopper (in other words, renter).
 - **LTV** is the projected revenue that a customer will generate during their lifetime. 
 Because for this newly born startup it's hard to estimate lifespan a renter. 
 Here I used a modified version **LTV per year**.
 
 ![market_analysis](img/task1/market_analysis.png)
 

The payback period (in months) is then defined as **CAC, divided by LTV, and multiplied by 12**.

*To protect the information of this company, detailed analysis results can only be found in the notebooks.
The actual numbers are hided for all graphs, and the true database link is not provided.*


## Task 2: Predicting user conversion
Will a registered user convert to an actual renter? 
Understanding user profile can help to increase conversion rate, a key metric
for E-commerce. To help my client understand what are strong indicators
 determining conversion rate, I built a Logistic Regression model 
 to predict whether a register user can convert. 
 
 First, I extracted some categorical features and numerical features
 from available user data and shopping history.


![feature_engineer](img/task2/feature_engineer.png)

### Ensemble classifier

Before moving on to train the model, we notice that our data is 
highly imbalanced. Only 3% of the users can convert to renters, with 97% of them not. 
Here I used a combination of undersampling and ensemble method. In particular,
I trained 20 models with each using 80% randomly selected renters and equal number
of non-renters. These 20 models are made as an ensemble classifier, in which
the classification is determined by the decision of majority of models.

<p align="center">
<img src="img/task2/imbalanced_data.png" width="300">
</p>

### Validation

The test of the ensemble classifier is performed on 20% of the orginal imbalanced 
data. 

In the following graph, I intentionally delete the feature *sign_in_counts* b/c this
feature is too strong that the classification accuracy could be as high as 0.99 when adding
it back in. This feature dims the importance of all other features.

<p align="center">
<img src="img/task2/confusion_matrix.png" width="300">
</p>


### Feature Importance

The average coefficient weights tell us on how predictive the features are.
Several interesting findings here.

- Although the age determines how much one spend on the website, it is **not**
a strong indicator on conversion.
- Domestic cities did a better job than international cities, which is later
 confirmed by my clients b/c they do have representatives in big domestic cities.
- Email account is a strong negative indicator, registering by FaceBook or hotmail 
might not provide their most frequently used emails.


<p align="center">
<img src="img/task2/feature_importance.png" width="200">
</p>


## Task 3: Customized recommendation for users
The recommendation is based on the renters'shopping history and the similarities between all available items using the description text of items stored in my client's database. The model also takes advantages of renters' size records, brand preferences, and cloth type preferences.

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
