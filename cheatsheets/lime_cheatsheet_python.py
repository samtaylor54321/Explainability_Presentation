################################### LIME CHEATSHEET ##################################

# Cheatsheet for deploying LIME on a model in python
# Example taken from LIME Github package
# https://marcotcr.github.io/lime/tutorials/Using%2Blime%2Bfor%2Bregression.html

# Lime -------------------------------------------------------------------------------

# Load packages
from sklearn.datasets import load_boston
import sklearn.ensemble
import numpy as np
import lime
import lime.lime_tabular

# Load dataset
boston = load_boston()

# Set up model
rf = sklearn.ensemble.RandomForestRegressor(n_estimators=1000)
# Get labels
train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(boston.data, boston.target, train_size=0.80)
# Fit model
rf.fit(train, labels_train)
# Tidy categorical features
categorical_features = np.argwhere(np.array([len(set(boston.data[:,x])) for x in range(boston.data.shape[1])]) <= 10).flatten()

# Explainer
explainer = lime.lime_tabular.LimeTabularExplainer(train, feature_names=boston.feature_names, class_names=['price'], categorical_features=categorical_features, verbose=True, mode='regression')
i = 25
exp = explainer.explain_instance(test[i], rf.predict, num_features=5)

# Plots
exp.show_in_notebook(show_table=True)
exp.show_in_notebook(show_table=True)
