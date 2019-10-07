################################ SHAP CHEATSHEET ##################################

# Cheatsheet for deploying explainable ML in python - shap. 
# Examples taken from the SHAP github page
# https://github.com/slundberg/shap


# TreeExplainer -------------------------------------------------------------

# Import packages
import xgboost
import shap

# load JS visualization code to notebook
shap.initjs()

# train XGBoost model
X,y = shap.datasets.boston()
model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)

# explain the model's predictions using SHAP values
# (same syntax works for LightGBM, CatBoost, and scikit-learn models)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# visualize the first prediction's explanation 
# (use matplotlib=True to avoid Javascript)
shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])

# visualize the training set predictions
shap.force_plot(explainer.expected_value, shap_values, X)

# create a SHAP dependence plot to show the effect of a single feature 
# across the whole dataset
shap.dependence_plot("RM", shap_values, X)

# summarize the effects of all the features - generally summary/bar plot
shap.summary_plot(shap_values, X)
shap.summary_plot(shap_values, X, plot_type="bar")

# KernelExplainer ------------------------------------------------------------

import sklearn
import shap
from sklearn.model_selection import train_test_split

# print the JS visualization code to the notebook
shap.initjs()

# train a SVM classifier
X_train,X_test,Y_train,Y_test = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)
svm = sklearn.svm.SVC(kernel='rbf', probability=True)
svm.fit(X_train, Y_train)

# use Kernel SHAP to explain test set predictions
explainer = shap.KernelExplainer(svm.predict_proba, X_train, link="logit")
shap_values = explainer.shap_values(X_test, nsamples=100)

# plot the SHAP values for the Setosa output of the first instance
shap.force_plot(explainer.expected_value[0], shap_values[0][0,:], X_test.iloc[0,:], link="logit")



