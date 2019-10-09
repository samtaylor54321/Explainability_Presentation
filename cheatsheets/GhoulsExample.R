# https://www.kaggle.com/jhuno137/classification-tree-using-rpart-100-accuracy


library(dplyr)
library(tidyverse)
library(data.table)


library(nnet) 
library(rpart)
library(rpart.plot)
library(rattle)
library(caret)
library(e1071)
library(randomForest)


library(plotly)
library(ggplot2)


#----------------------------------------------------------------------------------------#

# list.files(path = "../input")
# 
# # read train file: ####
# train.data = read.csv("../input/ghouls-goblins-and-ghosts-boo/train.csv")
# test.data = read.csv("../input/ghouls-goblins-and-ghosts-boo/test.csv")

#----------------------------------------------------------------------------------------#

train.data = read.csv("train.csv")
test.data = read.csv("test.csv")

# identify fields within the data:
colnames(train.data)

# just to confirm: the same fields would appear withn the test data:
colnames(test.data)

# how are the fields saved: what are the classes of the fields:
sapply(train.data, class)

str(train.data)

#----------------------------------------------------------------------------------------#

# use multinomial regression to fit model:

train.data.multnml = train.data

color.tbl = as.data.frame( t(table(train.data$color))) %>% 
  mutate(
    col.names = Var2,
    clr = c(1, 2, 3, 0, 4, 5)
  ) %>%
  select(-Var1, -Var2, -Freq)

train.data.multnml = inner_join(
  
  x = train.data.multnml,
  y = color.tbl,
  by = c("color" = "col.names")
  
  
)



mltnml.model = multinom( formula = type ~ bone_length + rotting_flesh + 
                           hair_length + has_soul + clr,
                         data = train.data.multnml)

# predicted probabilities:
mltnml.prdct.prblts = as.data.frame(fitted(mltnml.model))

# get the maximum values (rowwise)
max.probs = apply(X = mltnml.prdct.prblts, MARGIN = 1, FUN = max )

# pre-allocate id:
mltnml.prdct.prblts = mltnml.prdct.prblts %>% mutate( predicted.label = 0)

# assign data label:
for (i in 1:length(max.probs)){
  
  mltnml.prdct.prblts$predicted.label[i] = 
    colnames(mltnml.prdct.prblts)[
      which(max.probs[i] == mltnml.prdct.prblts[i, 1:3])
      ]
  
}

# attach actual label:
mltnml.prdct.prblts = mltnml.prdct.prblts %>% mutate( actual.label = train.data$type)

# view of assigned labelled:
with(mltnml.prdct.prblts, 
     table(predicted.label)
)

# train data confusion matrix:

mtlnml.tbl = with(mltnml.prdct.prblts, 
                  table(predicted.label, actual.label)
)


confusionMatrix(mtlnml.tbl)
#-------------------------------------------------------------------------------------------------------------------------------#

# nueral networks:

train.data.nnet = train.data
train.data.nnet = train.data.nnet %>% mutate(color = as.character(color))

# one hidden layer:
nnet.model = nnet(formula = type ~ bone_length +  rotting_flesh + hair_length + has_soul + color, data =  train.data.nnet, size = 1)

nnet.predicted = predict(nnet.model, train.data.nnet[, c("bone_length", "rotting_flesh", "hair_length", "has_soul", "color")], type = "class")

train.data.nnet = train.data.nnet %>% mutate(
  predicted = nnet.predicted,
  actual = type
)

nnet.tbl = with(train.data.nnet,
                table(predicted, actual)
)

confusionMatrix(nnet.tbl)

#-------------------------------------------------------------------------------------------------------------------------------#

# two hidden layers:
nnet2.model = nnet(formula = type ~ bone_length +  rotting_flesh + hair_length + has_soul + color, data =  train.data.nnet, size = 2)

nnet2.predicted = predict(nnet2.model, train.data.nnet[, c("bone_length", "rotting_flesh", "hair_length", "has_soul", "color")], type = "class")

train.data.nnet = train.data.nnet %>% mutate(
  predicted2 = nnet2.predicted
)

nnet2.tbl = with(train.data.nnet,
                 table(predicted2, actual)
)

confusionMatrix(nnet2.tbl)

#-------------------------------------------------------------------------------------------------------------------------------#

# three hidden layers:
nnet3.model = nnet(formula = type ~ bone_length +  rotting_flesh + hair_length + has_soul + color, data =  train.data.nnet, size = 3)

nnet3.predicted = predict(nnet3.model, train.data.nnet[, c("bone_length", "rotting_flesh", "hair_length", "has_soul", "color")], type = "class")

train.data.nnet = train.data.nnet %>% mutate(
  predicted3 = nnet3.predicted
)

nnet3.tbl = with(train.data.nnet,
                 table(predicted3, actual)
)
confusionMatrix(nnet3.tbl)
#-------------------------------------------------------------------------------------------------------------------------------#

# tree
train.data.tree = train.data

train.data.tree = train.data.tree %>% mutate(
  
  color = as.character(color),
  type = as.character(type)
  
)

# fit model
tree.model = rpart( type ~ bone_length +  rotting_flesh + hair_length + has_soul + color, data = train.data.tree)

# summary:
summary(tree.model)

fancyRpartPlot(tree.model)

pred = predict(object=tree.model, train.data.tree[, c("bone_length", "rotting_flesh", "hair_length", "has_soul", "color")], type = "class")

train.data.tree = train.data.tree %>% mutate(
  predicted = pred,
  actual = type
)
tree.tbl = with(train.data.tree,
                table(predicted, actual)
)

confusionMatrix(tree.tbl)

#-------------------------------------------------------------------------------------------------------------------------------#
train.data.rf = train.data

train.data.rf = train.data.rf  %>% select(-id)

# characters should be converted to factors:
model.rf = randomForest(type ~ ., data = train.data.rf,
                        ntree=100)


summary(model.rf)

train.data.rf = train.data.rf %>% mutate(
  predicted = predict(model.rf,train.data.rf[, c("bone_length", "rotting_flesh", "hair_length", "has_soul", "color")] , type="response"),
  actual = type
)



rf.tbl = with(train.data.rf,
              table(predicted, actual)
)

confusionMatrix(rf.tbl)

#-------------------------------------------------------------------------------------------------------------------------------#