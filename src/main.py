# design an mnist machine learning model using sklearn.
# the model will be trained on the training data and tested on the test data.
# the model will be saved to disk and loaded from disk.
# the model will be used to predict the class of a new image.
# the model will be evaluated using the confusion matrix.
# the model will be evaluated using the classification report.
# the model will be evaluated using the accuracy score.

# import the required libraries
from sklearn import datasets
from sklearn import model_selection
from sklearn import metrics

# load the mnist dataset
mnist = datasets.fetch_openml("mnist_784")

# get the features and labels
X = mnist.data
y = mnist.target

# split the data into training and testing sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# create a machine learning model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

# train the model
model.fit(X_train, y_train)

# test the model
y_pred = model.predict(X_test)

# evaluate the model
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
classification_report = metrics.classification_report(y_test, y_pred)
accuracy_score = metrics.accuracy_score(y_test, y_pred)

# print the evaluation results
print("Confusion Matrix:")
print(confusion_matrix)
print("Classification Report:")
print(classification_report)
print("Accuracy Score:")
print(accuracy_score)

# save the model to disk
import joblib

joblib.dump(model, "model/model.pkl")

# load the model from disk
model = joblib.load("model/model.pkl")
