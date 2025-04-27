#!/usr/bin/env python3


# Be aware, this script usually takes >40 seconds
# to complete.


# This is the start of our timer
import time
start = time.time()

# Imports
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import sys
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Our original code ran perfectly in the Jupyter Notebook 
# in which it was originally written but did not terminate 
# when run from this script. 
# The solution was to wrap the code in a function, then 
# call it at the  bottom of the script:
# 	if __name__ == "__main__":
# 	    main()
# at the bottom of the file. See the included article 
# for full (and rather obscure) details. 

def main():

	print("*********************************")
	print("Multi-layer Perceptron Classifier")
	print("*********************************")

	# This little hack hides a later warning
	warnings.filterwarnings("ignore", category=RuntimeWarning)

	# Load dataset from CSV file using pandas.
	try: 
		df = pd.read_csv(r'mlp_classifier_data.csv')
	except FileNotFoundError:
	    print("Dataset not found. Make sure it's in the working directory.")
	    print('\n')
	    sys.exit(1)

	if df.empty == False:
		print("Dataset loaded.")
		print("Please be patient, expected run-time: 45 to 300 seconds! \n")

	# Extract candidate features
	X = df[["SigPktLenIn","ConTcpFinCntIn","ConTcpSynCntIn","InPktLen32s10i[0]","InPktLen1s10i[2]", "InPktLen8s10i[7]","OutPktLen1s10i[0]","FourGonAngleIn[9]","InPktLen8s10i[1]", "PolyInd8ordOut[5]","PolyInd8ordIn[5]","SumTTLIn","MedTTLOut","MeanTTLIn","SumPktOut", "MedTCPHdrLen","SigTTLOut","SumPktLenIn","SigTdiff2PktsOut","BytesPerSessOut"]]

	# Extract non-numeric label_2
	y = df["label_2"]

	# Encode label using LabelEncoder
	le = LabelEncoder() 
	y = le.fit_transform(y) 

	# Normalize the features using StandardScaler from sklearn.preprocessing
	scaler = StandardScaler()
	scaled = scaler.fit_transform(X) # makes numpy array

	# Turn scaled from numpy array back to a pandas dataframe
	# otherwise we loose column names.
	scaled = pd.DataFrame(scaled, columns=X.columns)

	# Make train/test split, be sure to use the scaled data! but X_train becomes a numpy array
	X_train, X_test, y_train, y_test = train_test_split(scaled, y, test_size=0.2, random_state=1)

	# Initialize a MLP Classifier
	mlp_full = MLPClassifier(activation='relu', 
								alpha=0.0001,
								hidden_layer_sizes=(100,),
								learning_rate_init=0.01,
								solver='adam',
								random_state=1,
								verbose=False,
								early_stopping=True,
								max_iter=1000
							)

	# Train MLP on training data, fit() does the training.
	mlp_full.fit(X_train, y_train)

	# Tune hyperparameters with grid search 
	# Amazing... but it takes a while.
	parameter_grid = {
	   'activation': ['identity', 'logistic', 'tanh', 'relu'],
	   'alpha': [0.0001, 0.001],
	   'hidden_layer_sizes': [(128,32), (100,), (50, 50)],
	   'learning_rate_init': [0.001, 0.01],
	   'solver': ['sgd', 'adam'],
	}
	
	grid = GridSearchCV(mlp_full, parameter_grid, cv=5, scoring='accuracy',  n_jobs=-1)
	grid.fit(X_train, y_train)

	print("Best parameters: ")
	for i in grid.best_params_:
		print('    ', i)

	print("\n Best cross-validation score:")
	print('    ', grid.best_score_, '\n')

	# The crossgrid search for optimal parameters found:
	# Best parameters: {'activation': 'relu', 
	#                   'alpha': 0.0001, 
	#                   'hidden_layer_sizes': (100,), 
	#                   'learning_rate_init': 0.01, 
	#                   'solver': 'adam'}
	# Best cross-validation score: 0.8634920634920634
	# Note that hidden_layer_sizes difference from 
	# the suggestion in the assignment spec.


	# Make predictions using the test data.
	# V1:
	# y_pred = mlp_full.predict(X_test)
	# Alternatively, we can use the results of the grid search.
	# You can't use this if the grid object is commented out above, obviously
	# V2:
	y_pred = grid.best_estimator_.predict(X_test)


	# First accuracy output:
	############################################
	accuracy = accuracy_score(y_test, y_pred)
	print(f"Accuracy for mlp_full classifier: {accuracy}")

	# Output run time of first phase
	end1 = time.time()
	print(f"Mid-point runtime: {end1 - start:.4f} seconds")
	############################################

	# Visualize detection performance using a confusion matrix
	# Ground truth, or 'y_true' is the labelled test data.
	cm = confusion_matrix(y_test, y_pred, labels=mlp_full.classes_) 
	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mlp_full.classes_)
	fig, ax = plt.subplots()

	# This creates the plot
	disp.plot(ax=ax)

	# Saves image to current directory
	# Must save before showing!
	plt.savefig('output_figure_from_script.png', bbox_inches='tight')
	print("Confusion matrix image saved.")

	# This makes figure pop-up on the screen
	plt.show()


	###################################
	# Use Forward Feature Selection. 
	###################################
	# Using SequentialFeatureSelector from scikit-learn

	# To make reduce runtime, we add two optimizations:
	# 	1. reduce the value of cv from 5 to 3
	#	2. add n_jobs=-1 to use all CPU cores
	ffs = SequentialFeatureSelector(mlp_full, n_features_to_select=9, tol=None, direction='forward', scoring=None, cv=3, n_jobs=-1)

	# fit returns self, the FFS object itself
	ffs.fit(X_train, y_train)

	# print selected features using get_support()
	selected_mask = ffs.get_support()
	selected_features = X_train.columns[selected_mask]
	print("Selected features: ")
	for i in selected_features:
		print('    ', i)

	###################################
	# Model training using the selected subset 
	###################################
	# make the transformation
	X_train_selected = ffs.transform(X_train)
	X_test_selected = ffs.transform(X_test)

	# train model on selected features
	mlp_reduced = MLPClassifier(random_state=42, max_iter=1000)
	mlp_reduced.fit(X_train_selected, y_train)

	y_pred = mlp_reduced.predict(X_test_selected)

	# Second accuracy output:
	############################################
	accuracy = accuracy_score(y_test, y_pred)
	print(f"Accuracy for mlp_reduced classifier: {accuracy}")

	# Output run time of second phase
	end2 = time.time()
	print(f"Total runtime: {end2 - start:.4f} seconds", "\n")
	############################################

if __name__ == "__main__":
	main()

