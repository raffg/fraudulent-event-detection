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

## Model Specifications
Grid search was performed, with 10-fold cross validation, to assess various models and hyperparameters:
- Logistic regression with regularization
- Random forest
- Gradient boosting

The optimal model was gradient boosting with the following hyperparameters:
-

## Model Results
Our logistic regression, with 10-fold cross validation, performed with the following metrics on the training data:
- Accuracy: 0.979
- Precision: 0.908
- Recall: 0.855
- F1: 0.881

## About the App

## Next Steps

## Code Files
