############################### SHAP CHEATSHEET ###################################

# Cheatsheet for R LIME  
# Taken from IML github page
# https://cran.r-project.org/web/packages/iml/vignettes/intro.html

# SHAP ------------------------------------------------------------------------

# Load packages
library("iml")
library("randomForest")

# Load data
data("Boston", package  = "MASS")
head(Boston)

# Train model
set.seed(42)
data("Boston", package  = "MASS")
rf = randomForest(medv ~ ., data = Boston, ntree = 50)

# Create predictor object
X = Boston[which(names(Boston) != "medv")]
predictor = Predictor$new(rf, data = X, y = Boston$medv)

# Generate SHAP and plot
shapley = Shapley$new(predictor, x.interest = X[1,])
shapley$plot()

shapley$explain(x.interest = X[2,])
shapley$plot()
