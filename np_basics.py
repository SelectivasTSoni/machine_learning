#!/usr/bin/env python3

# np_basics.py

import numpy as np

# this uses randint():
# uint16_array = np.random.randint(1, 999, size=(4, 2), dtype=np.uint16)

#The docs indicate:
#``New code should use the integers method of a Generator instance instead.'' So:

# create a Generator instance, seed for reproducible results
rng = np.random.default_rng(seed=42)  

# use the integers method of the instance
uint16_array = rng.integers(1, 999, size=(4, 2), dtype=np.uint16) 


print(uint16_array)
print(uint16_array.shape)
print(uint16_array.ndim)
print(uint16_array.itemsize)
print(uint16_array.dtype)

# here we use arange method, which excludes the 200.
int_array1 = np.arange(100, 200, 10)
print(int_array1)

# if we want the 200, we just hack a bit:
int_array1 = np.arange(100, 210, 10)
print(int_array1)

# here we use linspace which includes the 200 and works with non-integer step sizes.
# This gives an unexpected result which I did not investigate.
int_array2 = np.linspace(100, 200, num=10)
print(int_array2)

# similar to above.
int_array1 = np.arange(200, 400, 20)
print(int_array1)

SampleArray = np.array([[11 ,22, 33], [44, 55, 66], [77, 88, 99]]) 
print("Printing Input Array")
print(SampleArray)

print(SampleArray[0,1])
print(SampleArray[1,1])
print(SampleArray[2,1])

SampleArray = np.array([[3 ,6, 9, 12], [15 ,18, 21, 24], 
[27 ,30, 33, 36], [39 ,42, 45, 48], [51 ,54, 57, 60]]) 
print("Printing Input Array")
print(SampleArray)
print('\n')

# idea from https://stackoverflow.com/questions/10198747/how-can-i-simultaneously-select-all-odd-rows-and-all-even-columns-of-an-array
# Get odd rows
odd_rows = SampleArray[::2] 

#get even columns
even_columns = SampleArray[:, 1::2]

# create new combined array object
new_array = SampleArray[::2, 1::2]

print(new_array)

arrayOne = np.array([[5, 6, 9], [21 ,18, 27]])
arrayTwo = np.array([[15 ,33, 24], [4 ,7, 1]])

array_sum = np.add(arrayOne, arrayTwo)
print(array_sum)

# alternatively
print(arrayOne + arrayTwo)


print("\n")
print("Theory questions & answers")
print("\n")
print("1. Explain the difference between 'filter', 'wrapper', and 'embedded' methods of feature selection. Provide an example for each. ")
print("\n")
print("These answers come directly from the slides. The Filter approach ranks features or feature subsets independently of the predictor. This can include univariate and multivariate methods. Examples of Filter methods include correlation, Chi-square test and informaion gain. The Wrapper approach uses a predictor assess features or feature subsets. Example techniques include forward selection, backward elimination, and step-wise selection. The Embedding approach uses a predictor to build a model with a subset of features that are internally selected. Examples include Regularization methods such as Lasso, Ridge Regression, and Elastic Net. ")
print("\n")
print("2. You are given a dataset with 100 features to predict a binary outcome. After applying a correlation filter method for feature selection, you find that 30 features have a correlation coefficient less than 0.05 with the target variable. What should you do with these features and why?")
print("\n")
print("Assuming use of the Pearson correlation coefficient, we know that a value of 1 indicates perfect correlation with the target variable and -1 is perfectly un-correlated. In the case presented, what became of the features with <0.05 correlation coefficient (cc) would depend on the cc of the the remaining 70 features. It depends on the data: for all we know here, none of the features have cc's above 0.05 which would mean these are the most informative features available to us. If we assume the remaining 70 features have cc's above 0.05, we could write our program is such a way as to ignore the less correlated features. We could also rank the features and determine some sensisble cut-off point to improve the time-performance of our model. ")
print("\n")



