# Detection of Fraudulent Event Postings
Team:
- [Katie Lazell-Fairman](https://github.com/lazell)
- [Greg Rafferty](https://github.com/raffg)
- [Kristie Sarkar](https://github.com/ks2282)
- [Rohit Unni](https://github.com/rohitunni)

## Overview
The goal of this project is to build a web application that identifies fraudulent event listings.

## Feature Engineering
We performed exploratory data analysis on a historical data set, containing event information and indicators of fraudulent activity, to determine which features to include or modify for modeling.

- Features related to ticket sales were excluded, as our model is for detecting fraud earlier in the process, before transactions are made.
- Specific organizations highly associated with fraudulent events were blacklisted,
and a flag for whether an event's organization is blacklisted was added as a binary feature.
- The hour of day was extracted from datetimes to use as a feature.
- Ticket pricing information was extracted from the 'ticket_types' field.

The most influential features to our model were
- Whether or not there has been a previous payout to that user
- Whether or not the payout type is provided
- The domain of the email address
- The age of the user

## Model Specifications
Grid search was performed, with 10-fold cross validation, to assess various models and hyperparameters:
- Logistic regression with regularization
- Random forest
- Gradient boosting
- K nearest neighbors

The optimal model was gradient boosting with the following hyperparameters:
- loss='deviance'
- learning_rate=.5
- n_estimators=100,
- max_depth=3,
- min_samples_split=2,
- min_samples_leaf=2,
- max_features='auto'

## Model Results
We separated our data in train and test sets, and then trained each model while optimizing the hyperparameters for maximum recall. After testing on an unseen test set, our models produced the following results:

|   |Gradient Boosting|Random Forest|Logistic Regression|KNN|Ensemble|
|---:|:--------------:|:-----------:|:-----------------:|:--:|:-----:|
|Accuracy|98%|98%|97%|98%|98%|
|Precision|91%|95%|92%|88%|95%|
|Recall|85%|83%|78%|84%|83%|
|F1 score|88%|89%|84%|86%|89%|

## About the App

## Next Steps

## Code Files
