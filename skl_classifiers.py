#!/usr/bin/env python3

# skl_classifiers.py

# Import necessary libraries required to perform your supervised Machine Learning 
# task including preprocessing, visualizations, and evaluation
import pandas as pd
import sklearn as skl
import matplotlib.pyplot as plt
import mglearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# metrics
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score as f1
from sklearn.metrics import accuracy_score
# The confusion matrix gives false positive rate, FPR, also called 'fallout'
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from collections import Counter

#from sklearn.linear_model import LogisticRegression
#from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN


# Load the card_transdata dataset. Dataset contains imbalance in Real and Fraud classes
data_file = "skl_classifiers_data.csv"
df = pd.read_csv(data_file, sep=",")
print(df.shape)

##########################################################
# Resample the original dataset with 87,403 data points for both Real and Fraud classes
##########################################################

#print(df.head(2))
#print('\n')

fraud_df = df.loc[df['fraud'] == 0.0]
real_df = df.loc[df['fraud'] == 1.0]

fraud_resampled = skl.utils.resample(fraud_df, replace=True, n_samples=87403, random_state=0, stratify=None)
real_resampled = skl.utils.resample(real_df, replace=True, n_samples=87403, random_state=0, stratify=None)
combined_df = pd.concat([fraud_resampled, real_resampled])

print('Count of Fraud values:')
print(fraud_resampled.fraud.value_counts())
print('\n')

print('Count of Real values:')
print(real_resampled.fraud.value_counts())
print('\n')

print('Total Count of "fraud" values:')
print(combined_df.fraud.value_counts())
print('\n')


# get X and y
X = combined_df.iloc[:,[0,1,2,3,4,5,6,7]]        #what data we use
y = combined_df.iloc[:,7]                        #what we predict

# train_test_split() 70/30
X_train_70, X_test_30, y_train_70, y_test_30 = train_test_split(X, y, random_state=0, test_size=0.3)

# get y_true, an array (a Pandas Series) of true classifications. 
# We supply this to our metrics and matrices.
# `y_true` must be the same length as the `y_test_predictions_30` 
# which will be the output of the `predict` methods below. 
# The series assigned to `y_test_predictions_30` below has the shape of (52442,) 
# which is 30% of 174,806. For simplicity, we just slice `y_true` down to 52,442: 
#y_true = combined_df['fraud']
#y_true = y_true[y_true != False]
#y_true = y_true[:52442]

y_true = y_test_30

print("Shape of y_true: ", y_true.shape)
print("Shape of combined_df: ", combined_df.shape)
print(y_true.head(2))


############################################################
# Perform KNN classification to classify Real/Fraud(use k=5)
############################################################

# instantiate and fit This is the "Classifier" mentioned in the confusion_matix example at:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
knn = KNeighborsClassifier(n_neighbors=5).fit(X_train_70, y_train_70)

# we use this later
knn_classifier = knn

# Make preditions, both labels and probabilities!
y_test_predictions_knn = knn.predict(X_test_30)
y_test_probabilities_knn = knn.predict_proba(X_test_30)[:,1]

print("Shape of output array: ", y_test_predictions_knn.shape)

# Generate metrics for comparison
# precision_score, 
# recall_score, 
# f1_score 
# take 
# y_test_predictions_...
precision = precision_score(y_true, y_test_predictions_knn)
recall = recall_score(y_true, y_test_predictions_knn)
# note there is name conflict between sciKit-learn f1_score and numpy.f1_score. To resolve this,
# we just imported f1_score as f1
f_one = f1(y_true, y_test_predictions_knn)


# average_precision_score, 
# PrecisionRecallDisplay 
# take 
# y_test_probabilities_...
avg_precision = average_precision_score(y_true, y_test_probabilities_knn)
display = PrecisionRecallDisplay.from_predictions(y_true, y_test_probabilities_knn, name="KNN", plot_chance_level=True)

print("Precision Score: ", precision)
print("Average Precision Score: ", avg_precision)
print("Recall Score: ", recall)
print("F1 Score: ", f_one)

# visualizations
ConfusionMatrixDisplay.from_predictions(y_true, y_test_predictions_knn)
plt.show()


_ = display.ax_.set_title("2-class Precision-Recall curve")


############################################################
# Perform Naïve bayes to classify Real/Fraud
############################################################

# instantiate and fit
gnb = GaussianNB().fit(X_train_70, y_train_70)

# we use this later
gnb_classifier = gnb


# Make preditions, both labels and probabilities!
y_test_predictions_gnb = gnb.predict(X_test_30)
y_test_probabilities_gnb = gnb.predict_proba(X_test_30)[:,1]

# Generate metrics for comparison
# precision_score, 
# recall_score, 
# f1_score 
# take 
# y_test_predictions_...
precision = precision_score(y_true, y_test_predictions_gnb)
recall = recall_score(y_true, y_test_predictions_gnb)
f_one = f1(y_true, y_test_predictions_gnb)

# average_precision_score, 
# PrecisionRecallDisplay 
# take 
# y_test_probabilities_...
avg_precision = average_precision_score(y_true, y_test_probabilities_gnb)
display = PrecisionRecallDisplay.from_predictions(y_true, y_test_probabilities_gnb, name="GaussianNB", plot_chance_level=True)

print("Precision Score: ", precision)
print("Average Precision Score: ", avg_precision)
print("Recall Score: ", recall)
print("F1 Score: ", f_one)

# visualizations
ConfusionMatrixDisplay.from_predictions(y_true, y_test_predictions_gnb)
plt.show()

_ = display.ax_.set_title("2-class Precision-Recall curve")


############################################################
# Perform Decision Trees to classify Real/Fraud
############################################################

# instantiate and fit
dtc = DecisionTreeClassifier().fit(X_train_70, y_train_70)

# we use this later in question 7
dtc_classifier = dtc

# Make preditions, both labels and probabilities!
y_test_predictions_dtc = dtc.predict(X_test_30)
y_test_probabilities_dtc = dtc.predict_proba(X_test_30)[:,1]

# Generate metrics for comparison
# Precision_score, 1 is best, 0 is worst

# precision_score, 
# recall_score, 
# f1_score 
# take 
# y_test_predictions_...
precision = precision_score(y_true, y_test_predictions_dtc)
recall = recall_score(y_true, y_test_predictions_dtc)
f_one = f1(y_true, y_test_predictions_dtc)

# average_precision_score, 
# PrecisionRecallDisplay 
# take 
#y_test_probabilities_...
avg_precision = average_precision_score(y_true, y_test_probabilities_dtc)
display = PrecisionRecallDisplay.from_predictions(y_true, y_test_probabilities_dtc, name="DecisionTreeClassification", plot_chance_level=True)


print("Precision Score: ", precision)
print("Average Precision Score: ", avg_precision)
print("Recall Score: ", recall)
print("F1 Score: ", f_one)

# visualizations
ConfusionMatrixDisplay.from_predictions(y_true, y_test_predictions_dtc)
plt.show()

_ = display.ax_.set_title("2-class Precision-Recall curve")


############################################################
# 8. Compare the accuracies of three classifiers 
############################################################

acc_knn = accuracy_score(y_test_30, y_test_predictions_knn)
acc_gnb = accuracy_score(y_test_30, y_test_predictions_gnb)
acc_dtc = accuracy_score(y_test_30, y_test_predictions_dtc)


print('Accuracy of KNN: ', acc_knn)
print('Accuracy of Gaussian Naive Beyes: ', acc_gnb)
print('Accuracy of Decision Tree Classifier: ', acc_dtc)

