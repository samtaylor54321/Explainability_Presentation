############################### LIME CHEATSHEET ###################################

# Cheatsheet for R LIME  
# Taken from LIME github page

# LIME ------------------------------------------------------------------------

library(caret)
library(lime)

# Split up the data set
iris_test <- iris[1:5, 1:4]
iris_train <- iris[-(1:5), 1:4]
iris_lab <- iris[[5]][-(1:5)]

# Create Random Forest model on iris data
model <- train(iris_train, iris_lab, method = 'rf')

# Create an explainer object
explainer <- lime(iris_train, model)

# Explain new observation
explanation <- explain(iris_test, explainer, n_labels = 1, n_features = 2)

# The output is provided in a consistent tabular format and includes the
# output from the model.
explanation
#> # A tibble: 10 x 13
#>    model_type case  label label_prob model_r2 model_intercept
#>    <chr>      <chr> <chr>      <dbl>    <dbl>           <dbl>
#>  1 classific… 1     seto…          1    0.680           0.120
#>  2 classific… 1     seto…          1    0.680           0.120
#>  3 classific… 2     seto…          1    0.675           0.125
#>  4 classific… 2     seto…          1    0.675           0.125
#>  5 classific… 3     seto…          1    0.682           0.122
#>  6 classific… 3     seto…          1    0.682           0.122
#>  7 classific… 4     seto…          1    0.667           0.128
#>  8 classific… 4     seto…          1    0.667           0.128
#>  9 classific… 5     seto…          1    0.678           0.121
#> 10 classific… 5     seto…          1    0.678           0.121
#> # … with 7 more variables: model_prediction <dbl>, feature <chr>,
#> #   feature_value <dbl>, feature_weight <dbl>, feature_desc <chr>,
#> #   data <list>, prediction <list>

# And can be visualised directly
plot_features(explanation)