# Detection of Fraudulent Event Postings
Team:
- [Katie Lazell-Fairman](https://github.com/lazell)
- [Greg Rafferty](https://github.com/raffg)
- [Kristie Sarkar](https://github.com/ks2282)
- [Rohit Unni](https://github.com/rohitunni)

## Feature Engineering
We performed exploratory data analysis to determine which features to include in modeling.

- Features related to ticket sales were excluded, as our model is for detecting fraud earlier in the process, before transactions are made.
- Specific organizations highly associated with fraudulent events were blacklisted,
and a flag for whether an event's organization is blacklisted was added as a binary feature.
- The hour of day was extracted from datetimes to use as a feature.
- Ticket pricing information was extracted from the 'ticket_types' field.

## Model Specifications
Grid search was performed, with 10-fold cross validation, to identify the optimal hyperparameters for regularizing the logistic regression, optimizing for recall. We also tried optimizing for accuracy, F1, and ROC_AUC.

The optimal parameters, identified while optimizing for recall are:
- Type of regularization:
- Lambda parameter for regularization:  

## Model Results

## About the App

## Next Steps

## Code Files
