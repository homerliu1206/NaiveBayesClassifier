import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Read the File
df = pd.read_csv("/Users/homerliu/Desktop/coding/midterm.csv")
# print(df)

# Split data into training and testing dataset
X = df.drop("output", axis = 1)
Y = df["output"]
# print(X)
Xtrain, Xtest, Ytrain, Ytest = train_test_split (X, Y, test_size = 0.3, random_state = 1)
# print(Xtrain)

# Define Naice Bayes classification model
model = GaussianNB()
model.fit(Xtrain, Ytrain)
Ypred = model.predict(Xtest)
# print(Ypred)
a = model.score(Xtrain, Ytrain)
print(f"Model score for training data: {a}")
b = model.score(Xtest, Ytest)
print(f"Model score for testing data: {b}")

# Calcualte confusion matrix
cm = confusion_matrix(Ypred, Ytest)
# Plot confusion matrix
c = sns.heatmap(cm, annot = True, cmap = "Blues")
print(c)

# Classification report metrics
d = classification_report(Ypred, Ytest)
print(d)

# Predict the label
new_data = {"age": [48], "sex": [1], "cp": [3], 
			"trtbps": [130], "chol": [218], "fbs": [1], 
			"restecg": [1], "thalachh": [152], "exng": [0], 
			"oldpeak": [1.8], "slp": [1], "caa": [0], "thall": [2]}
new_df = pd.DataFrame(new_data)
# print(new_df)
e = model.predict(new_df)
print(f"output: {e}")





# a) What is model score or accuracy for training data?
# b) What is model score or accuracy for testing data?
# c) Show confusion matrix usisng seaborn 
# d) Print classification report metrics.  
# e) Predict the label (class) of heart attack for a person with features below:
