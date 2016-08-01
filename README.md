# DataMining_CreditScoring
In this project, I implement different machine learning algorithms to improve credit scoring.
The dataset used for training the model contains more than 150,000 observations and 7 features.
Features engineering helps to expand the features.
Several models have been applied:
- Logistic Regression with Regularization, Decision Tree, SVM with different kernels, Naive Bayes and alpha-Tree;
- Emsemble methods such as Random Forest, Gradient Boosting;

Since the classification problem is unbalanced, resampling approaches is applied to balance the dataset before training.

Cross validation is applied to avoid over-fitting.

Use the model to predict for the testing dataset, improve the Area under Curve from 84% to 91%.
