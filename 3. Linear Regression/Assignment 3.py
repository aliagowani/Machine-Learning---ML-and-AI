# -*- coding: utf-8 -*-


# Boston Housing Study (Python)
# using data from the Boston Housing Study case
# as described in "Marketing Data Science: Modeling Techniques
# for Predictive Analytics with R and Python" (Miller 2015)

# Here we use data from the Boston Housing Study to evaluate
# regression modeling methods within a cross-validation design.

# program revised by Thomas W. Milller (2017/09/29)

# Scikit Learn documentation for this assignment:
# http://scikit-learn.org/stable/modules/model_evaluation.html 
# http://scikit-learn.org/stable/modules/generated/
#   sklearn.model_selection.KFold.html
# http://scikit-learn.org/stable/modules/generated/
#   sklearn.linear_model.LinearRegression.html
# http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
# http://scikit-learn.org/stable/modules/generated/
#   sklearn.linear_model.Ridge.html
# http://scikit-learn.org/stable/modules/generated/
#   sklearn.linear_model.Lasso.html
# http://scikit-learn.org/stable/modules/generated/
#   sklearn.linear_model.ElasticNet.html
# http://scikit-learn.org/stable/modules/generated/
#   sklearn.metrics.r2_score.html

# Textbook reference materials:
# Geron, A. 2017. Hands-On Machine Learning with Scikit-Learn
# and TensorFlow. Sebastopal, Calif.: O'Reilly. Chapter 3 Training Models
# has sections covering linear regression, polynomial regression,
# and regularized linear models. Sample code from the book is 
# available on GitHub at https://github.com/ageron/handson-ml

# prepare for Python version 3x features and functions
# comment out for Python 3.x execution
# from __future__ import division, print_function
# from future_builtins import ascii, filter, hex, map, oct, zip

import os
os.getcwd()

os.chdir('/Users/aliagowani/Documents')


# seed value for random number generators to obtain reproducible results
RANDOM_SEED = 1

# although we standardize X and y variables on input,
# we will fit the intercept term in the models
# Expect fitted values to be close to zero
SET_FIT_INTERCEPT = True

# import base packages into the namespace for this program
import numpy as np
import pandas as pd

# modeling routines from Scikit Learn packages
import sklearn.linear_model 
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score  
from math import sqrt  # for root mean-squared error calculation

# read data for the Boston Housing Study
# creating data frame 
url=''
boston_input = pd.read_csv(url)

# check the pandas DataFrame object boston_input
#print('\nboston DataFrame (first and last five rows):')
boston_input.head()

boston_input.tail()

print('\nGeneral description of the boston_input DataFrame:')
print(boston_input.info())

# drop neighborhood from the data being considered
boston = boston_input.drop('neighborhood', 1)
print('\nGeneral description of the boston DataFrame:')
print(boston.info())

print('\nDescriptive statistics of the boston DataFrame:')
print(boston.describe())

# set up preliminary data for data for fitting the models 
# the first column is the median housing value response
# the remaining columns are the explanatory variables
prelim_model_data = np.array([boston.mv,\
    boston.crim,\
    boston.zn,\
    boston.indus,\
    boston.chas,\
    boston.nox,\
    boston.rooms,\
    boston.age,\
    boston.dis,\
    boston.rad,\
    boston.tax,\
    boston.ptratio,\
    boston.lstat]).T

# dimensions of the polynomial model X input and y response
# preliminary data before standardization
print('\nData dimensions:', prelim_model_data.shape)

# standard scores for the columns... along axis 0
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
print(scaler.fit(prelim_model_data))

# show standardization constants being employed
print(scaler.mean_)

print(scaler.scale_)

# the model data will be standardized form of preliminary model data
model_data = scaler.fit_transform(prelim_model_data)

# dimensions of the polynomial model X input and y response
# all in standardized units of measure
print('\nDimensions for model_data:', model_data.shape)

"""## Our code starts here"""

# import additional packages
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.model_selection import ShuffleSplit

# make target for log mv
boston['logmv'] = np.log(boston.mv)
boston.head()

unscaled_x = np.array([boston.crim,\
    boston.zn,\
    boston.indus,\
    boston.chas,\
    boston.nox,\
    boston.rooms,\
    boston.age,\
    boston.dis,\
    boston.rad,\
    boston.tax,\
    boston.ptratio,\
    boston.lstat]).T

unscaled_y = np.array(boston.mv).T
unscaled_log_y = np.array(boston.logmv).T

prelim_log_model_data = np.array([boston.logmv,\
    boston.crim,\
    boston.zn,\
    boston.indus,\
    boston.chas,\
    boston.nox,\
    boston.rooms,\
    boston.age,\
    boston.dis,\
    boston.rad,\
    boston.tax,\
    boston.ptratio,\
    boston.lstat]).T

log_model_data = scaler.fit_transform(prelim_log_model_data)

"""## Exploratory Data analysis

### Example of why we need to scale
"""
plt.style.use(['seaborn'])

scatter_x = np.array(boston.nox).T
scatter_y = np.array(boston.mv).T

fig, ax = plt.subplots()
plt.scatter(x=scatter_x, y=scatter_y)
# hard code scales to be the same
plt.xlim(0, 50)
plt.ylim(0, 50)
plt.xlabel('Air Pollution')
plt.ylabel('Median value in thousands of dollars')
plt.title('Unscaled Air Pol vs Median Value')
z = np.polyfit(scatter_x, scatter_y, 1)
p = np.poly1d(z)
plt.plot(scatter_x,p(scatter_x),"r--")
plt.show()
fig.savefig("plot-Unscaled-AirPol-vs-MedianValue.pdf",
        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', orientation='portrait', papertype=None, format=None, pad_inches=0.25, frameon=None)
print("y=%.6fx+(%.6f)"%(z[0],z[1]))

fig, ax = plt.subplots()
scatter_x = np.array(boston.ptratio).T
plt.scatter(x=scatter_x, y=scatter_y)
plt.xlim(0,50)
plt.ylim(0,50)
plt.xlabel('Pupil/Teacher Ratio')
plt.ylabel('Median value in thousands of dollars')
plt.title('Unscaled Pupil/Teacher Ratio vs Median Value')
z = np.polyfit(scatter_x, scatter_y, 1)
p = np.poly1d(z)
plt.plot(scatter_x,p(scatter_x),"r--")
plt.show()
fig.savefig("plot-Unscaled-Pupil-Teacher-vs-Median-Value.pdf",
        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', orientation='portrait', papertype=None, format=None, pad_inches=0.25, frameon=None)
print("y=%.6fx+(%.6f)"%(z[0],z[1]))

"""The two plots above are comparing two plots of unscaled data on the same axes values from 0-50. Make note of the trendline slope values. Those would be the associated with the weights assigned to linear regressions. So you can already see before scaling, categories with huge differences will pull the line one way. 

Another way to think about it is if you plotted these on a 3D chart and tried to draw a best fit line for the resulting scatter plot.
"""

# scaled version
scatter_x = scaler.fit_transform(np.array(boston.nox).T.reshape(-1,1)).flatten()
scatter_y = scaler.fit_transform(np.array(boston.mv).T.reshape(-1,1)).flatten()

fig, ax = plt.subplots()
plt.scatter(x=scatter_x, y=scatter_y)
# hard code scales to be the same
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel('Air Pollution')
plt.ylabel('Median value in thousands of dollars')
plt.title('Scaled Air Pol vs Median Value')
z = np.polyfit(scatter_x, scatter_y, 1)
p = np.poly1d(z)
plt.plot(scatter_x, p(scatter_x),"r--")
plt.show()
fig.savefig("plot-Scaled-AirPol-vs-Median-Value.pdf",
        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', orientation='portrait', papertype=None, format=None, pad_inches=0.25, frameon=None)
print("y=%.6fx+(%.6f)"%(z[0],z[1]))

scatter_x = scaler.fit_transform(np.array(boston.ptratio).T.reshape(-1,1)).flatten()

fig, ax = plt.subplots()
plt.scatter(x=scatter_x, y=scatter_y)
# hard code scales to be the same
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel('Pupil/Teacher Ratio')
plt.ylabel('Median value in thousands of dollars')
plt.title('Scaled Pupil/Teacher Ratio vs Median Value')
z = np.polyfit(scatter_x, scatter_y, 1)
p = np.poly1d(z)
plt.plot(scatter_x, p(scatter_x),"r--")
plt.show()
fig.savefig("plot-Scaled-Pupil-Teacher-vs-Median-Value.pdf",
        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', orientation='portrait', papertype=None, format=None, pad_inches=0.25, frameon=None)
print("y=%.6fx+(%.6f)"%(z[0],z[1]))

"""After scaling with StandardScaler which converts the data point into a z-score. Now, the scales are both from -1 to 1. You can see the weights are now much more similar and can be compared by the algorithm more neatly now."""

boston.head()

boston.columns

# box plots of variables
fig, ax = plt.subplots(2, 7, figsize=(13,10))

for i, column in enumerate(boston.columns):
  if i <= 6:
    sns.boxplot(y=boston[column], ax=ax[0, i]).set_title(column)
  else:
    sns.boxplot(y=boston[column], ax=ax[1, i-7]).set_title(column)
plt.savefig("plot-boxplot-t-boston-housing.pdf",
         bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', orientation='portrait', papertype=None, format=None, pad_inches=0.25)
fig.show()

"""Charles River status can be ignored because that's a binary variable. The values are only 1 or 0.

The bunches of outliers in mv and logmv may be a precursor for how sensitive we should be when adjusting our models.  

Crime rate definitely looks suspect, and rooms may be too. The rest are well within their quantiles.
"""

plt.subplots(figsize=(12,5))
sns.distplot(boston.crim)
plt.savefig("plot-distplot-boston-housing.pdf",
         bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', orientation='portrait', papertype=None, format=None, pad_inches=0.25)
plt.show()

"""crime rate is heavily skewed towards 0 and has low values otherwise."""

boston.crim.value_counts(bins=[0, 1, 5, 80])

"""Crime rate is very imbalanced, and it should be considered being removed as a feature in our model.

Create a pairplot of all the features to see which features have a (positive or negative) correlation to the Median Value.
"""

# Create Data Frame
model_data_df = pd.DataFrame(model_data)
plt.subplots(figsize=(17,7))
# Label Data Frame
model_data_df.head()
model_data_df.columns = ['Median_Value', 'Crime_Rate', 'Zone_Lots', 'Indus', 'Chas_River', 'Air_Pollution', 'Rooms', 'Age', 'Dist_Employment', 'Radial_Hwy', 'Tax_Rate', 'Pupil_Teacher_Ratio', 'Lower_Socio']

# Plot to see whether any relationships stand out.
sns.pairplot(model_data_df)
plt.savefig("plot-pairplot-boston-housing.pdf",
         bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', orientation='portrait', papertype=None, format=None, pad_inches=0.25)
plt.show()

"""Create a histogram of the features to determine the distribution. We want to see whether any of the features are normally distributed for our model.

Show a correlation matrix to detemine which features have strong correlation to Median Value. It seems at the onset, Rooms and Lower Socio features are strongly correlated.
"""

corrmat = model_data_df.corr()
corrmat
# Rooms and Lower Socio features seem to be highly correlated to Median Value

# Heatmap of the features to quantify and visualize the correlations between the features.
fig, ax = plt.subplots(figsize = (18,10), facecolor='w')
sns.heatmap(corrmat, annot=True, annot_kws={'size': 12})
ax.set_ylim(13.0, 0)
fig.savefig("plot-heatmap-corrmat-boston-housing.pdf",
         bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', orientation='portrait', papertype=None, format=None, pad_inches=0.25)

"""The code below can be adjusted based on the threshold we seek. By entering 0.50 for threshold, we see that the absolute correlations of greater than 0.60 for Median Value belongs to Rooms and Lower_Socio features."""

def getCorrelatedFeature(corrdata, threshold):
        feature = []
        value = []

        for i, index in enumerate(corrdata.index):
                if abs(corrdata[index]) > threshold:
                        feature.append(index)
                        value.append(corrdata[index])

        model_data_df_getCorrFea = pd.DataFrame(data = value, index = feature, columns=['Corr Value'])
        return model_data_df_getCorrFea

# Defining the Threshold for correlation. We want correlation (positive and negative) that is less than or greater than .50
threshold = 0.50
corr_value = getCorrelatedFeature(corrmat['Median_Value'], threshold)
corr_value

"""This suggests rooms, pupil/teacher ratio, and lower socio economic status are the strongest features."""

corr_value.index.values

correlated_data = model_data_df[corr_value.index]
correlated_data.head()

#Pairplots of selected features that meet our defined threshold
sns.pairplot(correlated_data)
plt.tight_layout
plt.savefig("plot-pairplot-corrdata-boston-housing.pdf",
         bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', orientation='portrait', papertype=None, format=None, pad_inches=0.25)

#Heatmap of selected features that meet our defined threshold
fig, ax = plt.subplots()
sns.heatmap(correlated_data.corr(), annot=True, annot_kws={'size': 12})
ax.set_ylim(4.0, 0)
fig.savefig("plot-heatmap-correlated_data-threshold.pdf",
         bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', orientation='portrait', papertype=None, format=None, pad_inches=0.25)

"""Shows a linear relationship between the test and predicted results for Median Value."""

#Histograms
fig, ax = plt.subplots(2, 7, figsize=(17, 7), sharex=True)
sns.distplot( model_data_df["Median_Value"], ax=ax[0, 0])
sns.distplot( model_data_df["Crime_Rate"], ax=ax[0, 1])
sns.distplot( model_data_df["Zone_Lots"], ax=ax[0, 2])
sns.distplot( model_data_df["Indus"], ax=ax[0, 3])
sns.distplot( model_data_df["Chas_River"], ax=ax[0, 4])
sns.distplot( model_data_df["Air_Pollution"], ax=ax[0, 5])
sns.distplot( model_data_df["Rooms"], ax=ax[0, 6])
sns.distplot( model_data_df["Age"], ax=ax[1, 0])
sns.distplot( model_data_df["Dist_Employment"], ax=ax[1, 1])
sns.distplot( model_data_df["Radial_Hwy"], ax=ax[1, 2])
sns.distplot( model_data_df["Tax_Rate"], ax=ax[1, 3])
sns.distplot( model_data_df["Pupil_Teacher_Ratio"], ax=ax[1, 4])
sns.distplot( model_data_df["Lower_Socio"], ax=ax[1, 5])
fig.savefig("plot-distplot-boston-housing-mv.pdf",
        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', orientation='portrait', papertype=None, format=None, pad_inches=0.25)
plt.show()

"""## Feature selection"""

# all features
x = [row[1:12] for row in model_data]
y = [row[0] for row in model_data] # median values with StandardScaler
log_y = [row[0] for row in log_model_data] # log median values with Standard Scaler
# unscaled_y is raw median values
# unscaled_log_y is log transformed median values

"""So far, it looks like we want to focus on rooms, pupil/teacher ratio, and lower socio economic status."""

# preprcoessing transformers
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import PowerTransformer

sscaler = StandardScaler()
pscaler = PowerTransformer()
nscaler = Normalizer()

sscaler.fit_transform(x)
scaled_crim = sscaler.fit_transform(np.array(boston.crim).reshape(-1,1))

sns.distplot(scaled_crim)
plt.savefig("plot-scaled_crime-rate-boston-housing.pdf",
         bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', orientation='portrait', papertype=None, format=None, pad_inches=0.25)

features_df = boston.copy()

features_df.head()

sscaled_features_df = pd.DataFrame(sscaler.fit_transform(features_df), columns = features_df.columns)

pscaled_features_df = pd.DataFrame(pscaler.fit_transform(features_df), columns = features_df.columns)

nscaled_features_df = pd.DataFrame(nscaler.fit_transform(features_df), columns = features_df.columns)

# set sizes and grids for histograms
fig, ax = plt.subplots(2, 7, figsize=(13,10))

def plot_histograms(df):
    fig, ax = plt.subplots(2, 7, figsize=(13,10))
    for i, column in enumerate(df.columns):
        if i <= 6:
            sns.distplot(df[column], ax=ax[0, i]).set_title(column)
        else:
            sns.distplot(df[column], ax=ax[1, i-7]).set_title(column)
    fig.show()

# StandardScaler
plot_histograms(sscaled_features_df)
plt.savefig("plot-scaled-histograms-t-boston-housing.pdf",
         bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', orientation='portrait', papertype=None, format=None, pad_inches=0.25)

"""Features to keep: nox, rooms, ptratio, lstat"""

# Normalizer scaled
# plot_histograms(nscaled_features_df) # this crashes my colab script but it runs locally
# plt.savefig("plot-nscaled-hisotgrams-boston-housing.pdf",
#         bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', orientation='portrait', papertype=None, format=None, pad_inches=0.25)

"""Features to keep: nox, rooms, age, ptratio, lstat."""

# Power Transformer scaled
# plot_histograms(pscaled_features_df)
# plt.savefig("pscaled-histograms-t-boston-housing.pdf",
#         bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', orientation='portrait', papertype=None, format=None, pad_inches=0.25)

"""Features to keep: nox(4), rooms(5), dis(7), ptratio(10), lstat(11)

Expanding on this, it looks like power transformer does the best job of normalizing our dataset. We may be able to use more features if modeling requires we use more. As of right now, it looksl ike nox, rooms, dis, ptratio, and lstat are going to be the ones to keep.
"""

pscaled_features_df.iloc[:, [4,5,7,10,11]].head()

"""## Modeling"""

# import necessary packages from sklearn
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Use selected features from powertransformer
pscaled_x = np.array(pscaled_features_df.iloc[:, [4,5,7,10,11]])

#cv = ShuffleSplit(n_splits=5, test_size=0.2, train_size=0.8, random_state=42)
# cross-validation to be done on training set
cv = ShuffleSplit(n_splits=5, test_size=0.25, train_size=0.75, random_state=RANDOM_SEED) #slightly better results with 70-30 test size split but wanted to prevent overfitting

# Construct a list of regression models to iterate over
reg_models = [LinearRegression(), Ridge(), Lasso(), ElasticNet()]
models = ['Linear Regression', 'Ridge', 'Lasso', 'ElasticNet']

"""### Models using median values

Using the entire dataset is just a prototyping test to see the general RMSE values generated by each linear regressor.
"""

#run  the linear regression model with cross validation computing mean squared error as output
# used abs to compute to positive values
lr_mse = np.abs(cross_val_score(estimator = reg_models[0], cv=cv, X=x, y=y, scoring='neg_mean_squared_error'))
#convert to RMSE
lr_rmse =np.sqrt(lr_mse)
lr_rmse

lr_rmse.mean()

def get_rmse(model, x, y):
    # This function returns the RMSE of a model after using cross validation
    return np.abs(cross_val_score(estimator = model, X=x, y=y, scoring = 'neg_mean_squared_error', cv=cv))

#print output and convert to rmse
for i,model in enumerate(reg_models):
    print('{} RMSE: {}'.format(models[i], np.mean(np.sqrt(get_rmse(model, x, y)))))

"""The lower the RMSE value the better.  In this case the linear regression and ridge regression models standard out as the best two models followed by elastic net and lasso as the worst of the 4 models.  the best model was the linear regression by a slim margin against ridge regression in both the mean and standard deviation.  It's interesting to see that as RMSE value increases the deviation goes down.

### Models using log median values
"""

#print output and convert to rmse
for i,log_model in enumerate(reg_models):
    print('{} RMSE: {}'.format(models[i], np.mean(np.sqrt(get_rmse(log_model, x, log_y)))))

"""Running the models again using the log MV homes produced slightly improved results on the linear regression and ridge regression models which were the 2 best models but a slight decline in the lasso and elastic regression (utilizing average RMSE values).   We can see the variation between runs declined on all scenarios with the standard deviation decreasing, where the transformed log MV homes would be reccomended over just the MV homes

### Models with selected features
"""

for i,model in enumerate(reg_models):
    print('{} RMSE: {}'.format(models[i], np.mean(np.sqrt(get_rmse(model, pscaled_x, y)))))

for i,model in enumerate(reg_models):
    print('{} RMSE: {}'.format(models[i], np.mean(np.sqrt(get_rmse(model, pscaled_x, log_y)))))

"""With this, it looks like Power Transformer scaled, selected features produce the best results with either Linear Regression or Ridge RMSE. Since LinearReggressor does not have many tuning features, we can further optimize Ridge with GridSearchCV."""

# from sklearn.model_selection import learning_curve, ShuffleSplit

# def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
#                         n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10)):
    
#     plt.figure()
#     plt.title(title)
#     plt.xlabel("Training examples")
#     plt.ylabel("Score")
    
#     train_sizes, train_scores, test_scores = learning_curve(
#         estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)
    
#     plt.grid()

#     plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                      train_scores_mean + train_scores_std, alpha=0.1,
#                      color="r")
#     plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                      test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
#     plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
#              label="Training score")
#     plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
#              label="Cross-validation score")

#     plt.legend(loc="best")
#     return plt

# X = correlated_data.drop(labels = ['Median_Value'], axis = 1)
# y = correlated_data['Median_Value']

# title = "Learning Curves (Linear Regression) " + str(X.columns.values)

# cv = ShuffleSplit(n_splits=100, test_size=0.3, random_state=0)

# estimator = LinearRegression()
# plot_learning_curve(estimator, title, X, y, ylim=(0.8, 1.01), cv=cv, n_jobs=-1)

# plt.show()

# # work in code for gradient descent after best paramters are found?
# # I think this this will end a very exhaustive search for an optimized model on this dataset


# from sklearn.linear_model import LinearRegression
# clf_lr = LinearRegression()
# clf_lr.fit(X_train, y_train)
# lr_predicted = clf_lr.predict(X_test)
# lr_expected = y_test

# plt.figure(figsize=(4, 3))
# plt.scatter(lr_expected, lr_predicted, color='#666666')
# plt.plot([-3, 3], [-3, 3], 'C1', '--k')
# plt.axis('tight')
# plt.xlabel('True price')
# plt.ylabel('LinearReg. Predicted price')
# plt.tight_layout()


# from sklearn.ensemble import GradientBoostingRegressor
# clf_gb = GradientBoostingRegressor(n_estimators=1500, max_depth=4, min_samples_split=2, learning_rate=.05, loss='ls')
# clf_gb.fit(X_train, y_train)
# gb_predicted = clf_gb.predict(X_test)
# gb_expected = y_test

# plt.figure(figsize=(4, 3))
# plt.scatter(gb_expected, gb_predicted, color='#666666')
# plt.plot([-3, 3], [-3, 3], 'C2', '--k')
# plt.axis('tight')
# plt.xlabel('True price')
# plt.ylabel('GradientBoost Predicted price')
# plt.tight_layout()

# from sklearn import linear_model
# clf_sgd = linear_model.SGDRegressor(loss="squared_loss", penalty='l2', 
#                       alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, 
#                       tol=0.001, shuffle=True, verbose=0, epsilon=0.1, random_state=None, 
#                       learning_rate='invscaling', eta0=0.1, power_t=0.25, 
#                       early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, 
#                       warm_start=False, average=False)
# clf_sgd.fit(X_train, y_train)
# sgd_predicted = clf_sgd.predict(X_test)
# sgd_expected = y_test

# plt.figure(figsize=(4, 3))
# plt.scatter(sgd_expected, sgd_predicted, color='#666666')
# plt.plot([-3, 3], [-3, 3], 'C3', '--k')
# plt.axis('tight')
# plt.xlabel('True price')
# plt.ylabel('SGD Predicted price')
# plt.tight_layout()

# print("LinearReg. RMS: %r " % np.sqrt(np.mean((lr_predicted - lr_expected) ** 2)))
# print("GradientBoost RMS: %r " % np.sqrt(np.mean((gb_predicted - gb_expected) ** 2)))
# print("Stochastic Gradient Descent RMS: %r " % np.sqrt(np.mean((sgd_predicted - sgd_expected) ** 2)))

## Model Hyperparameter Optimization

"""Use GridSearchCV and a dictionary of parmeters to tune"""

from sklearn.model_selection import GridSearchCV
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

# set final training and testing set
X_train, X_test, y_train, y_test = train_test_split(pscaled_x, y, test_size = 0.25, random_state = RANDOM_SEED)

param_grid = {'alpha': [1000, 100, 10, 1, 0.1, 0.001, 0.0001],
              'max_iter': [10, 100, 1000, 10000],
             'solver': ['lsqr', 'sparse_cg', 'sag', 'saga'],
             'tol': [10000, 1000, 100, 10, 1, 0.1, 0.001, 0.0001, 0.00001]}

clf = Ridge(random_state=RANDOM_SEED)
search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=cv, scoring='neg_mean_squared_error')
search.fit(X_train, y_train)
@ignore_warnings(category=ConvergenceWarning)

"""### Final Ridge Regression Test"""

score = search.best_score_
best_rmse = np.sqrt(np.abs(score))
print('Best score: {}'.format(best_rmse))
print('Best parameters: {}'.format(search.best_params_))

final_ridge = Ridge(alpha=10, max_iter=10, solver='saga', 
                  tol=0.1, random_state=RANDOM_SEED)
final_ridge.fit(X_train, y_train)
ridge_predictions = final_ridge.predict(X_test)

# scatter plot of predictions vs actual values

sns.scatterplot(x=ridge_predictions, y=y_test)
plt.savefig("plot-ridge_pred_vs_test.pdf",
         bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', orientation='portrait', papertype=None, format=None, pad_inches=0.25)

print('Test score: {}'.format(np.sqrt(mean_squared_error(y_test, ridge_predictions))))

"""### Final Linear Regression Test"""

final_lr = LinearRegression()
final_lr.fit(X_train, y_train)
lr_predictions = final_lr.predict(X_test)

sns.scatterplot(x=lr_predictions, y=y_test)
plt.savefig("plot-linear_regression_test_vs_pred.pdf",
         bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', orientation='portrait', papertype=None, format=None, pad_inches=0.25)

print('Test score: {}'.format(np.sqrt(mean_squared_error(y_test, lr_predictions))))

"""So even compared to a tuned Ridge Regression model, Linear Regression performs slightly better using selected features scaled with Power Transformer and raw, unscaled targets."""