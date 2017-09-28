# Fashion Booster

## What is Fashion Booster?



## Task 1: Market Analysis
First, I performed a series of market analysis using my client's database. 
![market_analysis](img/task2/market_analysis.png)



## Task 2: Predict conversion on imbalanced dataset

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
