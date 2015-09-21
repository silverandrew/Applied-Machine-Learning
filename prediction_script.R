#Grow Decision Tree from Example Data. Analyze Accuracy, Precision, and Recall with Cross Validation
#Credit to http://eric.univ-lyon2.fr/~ricco/tanagra/fichiers/en_Tanagra_Validation_Croisee_Suite.pdf

#import dataset

#you need the dataset to run this of course...
b <- read.csv("~/Downloads/data.csv")
summary(b)

#you could do stuff such as print(b) or summary(b) here for background info

# split dataset into k folds
n <- nrow (b)
K <- 10
taille <- n%/%K
set.seed(5)
alea <- runif(n)
rang <- rank(alea)
bloc <- (rang-1) %/%taille + 1
bloc <- as.factor(bloc)
#print(b$att1)

library("rpart", lib.loc="/builds/R-packages")

# after cross validation on dataset, run and predict recall, accuracy, and precision
#all.accuracy <- numeric(0)
#all.precision <- numeric(0)
all.recall <- numeric(0)
for (k in 1:K) {
arbre <- rpart(att1 ~., data = b[bloc!=k,], method="class") 
  pred <- predict(arbre, newdata=b[bloc==k,], type="class")
  mc <- table(b$att1[bloc==k],pred)
  accuracy <- (mc[1,1] + mc[2,2])/sum(mc)
  all.accuracy <- rbind(all.accuracy,accuracy)
  precision <- (mc[1,1])/(mc[1,1] + mc[2,1])
  all.precision <- rbind(all.precision, precision)
  recall <- (mc[1,1])/(mc[1,1]+mc[1,2])
  all.recall <- rbind(all.recall, recall)
}

#now statistics are in vectors, so average them!

accuracy.cv = mean(all.accuracy)
precision.cv = mean(all.precision)
recall.cv = mean(all.recall)

print("accuracy")
print(accuracy.cv)
print("precision")
print(precision.cv)
print("recall")
print(recall.cv)

