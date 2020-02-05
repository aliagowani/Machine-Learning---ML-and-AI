# -*- coding: utf-8 -*-


# seed value for random number generators to obtain reproducible results
RANDOM_SEED = 1

"""## Import Packages"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.model_selection import ShuffleSplit
import time

# sklearn imports

"""## Load Data"""

total_start = time.time()

# read data for the Boston Housing Study
# creating data frame restdata
url=''
boston_input = pd.read_csv(url)

# check the pandas DataFrame object boston_input
#print('\nboston DataFrame (first and last five rows):')
boston_input.head()



"""# Data

## Exploration
"""

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

# standard scores for the columns... along axis 0
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
print(scaler.fit(prelim_model_data))

# make target for log mv
boston['logmv'] = np.log(boston.mv)
boston.head()

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

# dimensions of the polynomial model X input and y response
# preliminary data before standardization
print('\nData dimensions:', prelim_model_data.shape)

# show standardization constants being employed
print(scaler.mean_)

print(scaler.scale_)

# the model data will be standardized form of preliminary model data
model_data = scaler.fit_transform(prelim_model_data)

# dimensions of the polynomial model X input and y response
# all in standardized units of measure
print('\nDimensions for model_data:', model_data.shape)

# box plots of variables
fig, ax = plt.subplots(2, 7, figsize=(13,10))
for i, column in enumerate(boston.columns):
  if i <= 6:
    sns.boxplot(y=boston[column], ax=ax[0, i]).set_title(column)
  else:
    sns.boxplot(y=boston[column], ax=ax[1, i-7]).set_title(column)
plt.savefig("plot-box-plots-t-boston-housing.pdf",
         bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', orientation='portrait', papertype=None, format=None, pad_inches=0.25)
fig.show()

"""Charles River status can be ignored because that's a binary variable. The values are only 1 or 0.

The bunches of outliers in mv and logmv may be a precursor for how sensitive we should be when adjusting our models.

Crime rate definitely looks suspect, and rooms may be too. The rest are well within their quantiles.
"""

plt.subplots(figsize=(12,5))
sns.distplot(boston.crim)
plt.savefig("plot-crimerate-histogram-boston-housing.pdf",
         bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', orientation='portrait', papertype=None, format=None, pad_inches=0.25)
plt.show()

"""crime rate is heavily skewed towards 0 and has low values otherwise."""

boston.crim.value_counts(bins=[0, 1, 5, 80])

"""Crime rate is very imbalanced, and it should be considered being removed as a feature in our model.

Create a pairplot of all the features to see which features have a (positive or negative) correlation to the Median Value.
"""

# Create Data Frame
model_data_df = pd.DataFrame(model_data)
#plt.subplots(figsize=(17,7))
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
fig.savefig("plot-heatmap-corrmat-boston-housing.pdf", bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', orientation='portrait', papertype=None, format=None, pad_inches=0.25)
plt.show()

"""The code below can be adjusted based on the threshold we seek. By entering 0.60 for threshold, we see that the absolute correlations of greater than 0.60 for Median Value belongs to Rooms and Lower_Socio features."""

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
plt.savefig("plot-pairplot-correlated_data-threshold.pdf",
         bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', orientation='portrait', papertype=None, format=None, pad_inches=0.25)

#Heatmap of selected features that meet our defined threshold
fig, ax = plt.subplots(figsize = (9,5), facecolor='w')
sns.heatmap(correlated_data.corr(), annot=True, annot_kws={'size': 12})
plt.savefig("plot-heatmap-correlated_data-threshold.pdf",
         bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', orientation='portrait', papertype=None, format=None, pad_inches=0.25)

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
plt.savefig("plot-sns-hist-boston-housing.pdf",
         bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', orientation='portrait', papertype=None, format=None, pad_inches=0.25)
plt.show()

"""## Feature Engineering"""

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
plt.savefig("plot-sscaled_crime-rate-boston-housing.pdf",
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
plt.savefig("plot-sscaled-histograms-t-boston-housing.pdf",
         bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', orientation='portrait', papertype=None, format=None, pad_inches=0.25)

"""Features to keep: nox, rooms, ptratio, lstat"""

# Normalizer scaled
nscaled_features_df.hist(figsize=(18,10))
plt.savefig("plot-nscaled-hisotgrams-boston-housing.pdf",
         bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
            orientation='portrait', papertype=None, format=None, pad_inches=0.25)

"""Features to keep: nox, rooms, age, ptratio, lstat."""

# Power Transformer scaled
plot_histograms(pscaled_features_df)
plt.savefig("plot-pscaled-histograms-t-boston-housing.pdf",
         bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', orientation='portrait', papertype=None, format=None, pad_inches=0.25)

"""Features to keep: nox(4), rooms(5), dis(7), ptratio(10), lstat(11)

Expanding on this, it looks like power transformer does the best job of normalizing our dataset. We may be able to use more features if modeling requires we use more. As of right now, it looksl ike nox, rooms, dis, ptratio, and lstat are going to be the ones to keep.
"""

pscaled_features_df.iloc[:, [4,5,7,10,11]].head()

"""## Prepare Data"""

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.model_selection import ShuffleSplit
import sklearn.linear_model 
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score  
from math import sqrt  # for root mean-squared error calculation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

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

# cross-validation to be done on training set
cv = ShuffleSplit(n_splits=5, test_size=0.25, train_size=0.75, 
                  random_state=RANDOM_SEED) #slightly better results with 70-30 test size split but wanted to prevent overfitting

# Use selected features from powertransformer
pscaled_x = np.array(pscaled_features_df.iloc[:, [4,5,7,10,11]])

# Construct a list of regression models to iterate over
reg_models = [LinearRegression(), Ridge(), Lasso(), ElasticNet()]
models = ['Linear Regression', 'Ridge', 'Lasso', 'ElasticNet']

"""Using the entire dataset is just a prototyping test to see the general RMSE values generated by each linear regressor.

# Linear Modeling Experiment

## Baseline Modeling

### Using Median Values
"""

def get_rmse(model, x, y):
    # This function returns the RMSE of a model after using cross validation
    return np.abs(cross_val_score(estimator = model, X=x, y=y, scoring = 'neg_mean_squared_error', cv=cv))

#print output and convert to rmse
for i,model in enumerate(reg_models):
    print('{} RMSE: {}'.format(models[i], np.mean(np.sqrt(get_rmse(model, x, y)))))

"""The lower the RMSE value the better. In this case the linear regression and ridge regression models standard out as the best two models followed by elastic net and lasso as the worst of the 4 models. the best model was the linear regression by a slim margin against ridge regression in both the mean and standard deviation. It's interesting to see that as RMSE value increases the deviation goes down.

### Using Log Median Values
"""

#print output and convert to rmse
for i,log_model in enumerate(reg_models):
    print('{} RMSE: {}'.format(models[i], np.mean(np.sqrt(get_rmse(log_model, x, log_y)))))

"""Running the models again using the log MV homes produced slightly improved results on the linear regression and ridge regression models which were the 2 best models but a slight decline in the lasso and elastic regression (utilizing average RMSE values). We can see the variation between runs declined on all scenarios with the standard deviation decreasing, where the transformed log MV homes would be reccomended over just the MV homes

### Using Median Values and Selected Features
"""

for i,model in enumerate(reg_models):
    print('{} RMSE: {}'.format(models[i], np.mean(np.sqrt(get_rmse(model, pscaled_x, y)))))

"""### Using Log Median Values and Selected Features"""

for i,model in enumerate(reg_models):
    print('{} RMSE: {}'.format(models[i], np.mean(np.sqrt(get_rmse(model, pscaled_x, log_y)))))

"""With this, it looks like Power Transformer scaled, selected features produce the best results with either Linear Regression or Ridge RMSE. Since LinearReggressor does not have many tuning features, we can further optimize Ridge with GridSearchCV.

## Model Tuning
"""

from sklearn.model_selection import GridSearchCV
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

def gridsearch_alphas(estimator, param_grid, X, y):
    search = GridSearchCV(estimator=estimator, param_grid=param_grid, # search parameters
                          cv=cv, scoring='neg_mean_squared_error',
                          refit=True) # refits model to best parameters
    search.fit(X, y)
    best_score = search.best_score_
    rmse = np.sqrt(np.abs(best_score))
    best_params = search.best_params_
    print("Best Parameters: {}".format(best_params))
    print("RMSE: {}".format(rmse))
    return search # returns the fitted model with best parameters

# set final training and set aside testing set
lim_X_train, lim_X_test, lim_y_train, lim_y_test = train_test_split(pscaled_x, y, test_size = 0.25, 
                                                    random_state = RANDOM_SEED, shuffle=True)

param_grid = {'alpha': [1000, 100, 10, 1, 0.1, 0.001, 0.0001],
              'max_iter': [10, 100, 1000, 10000],
             'solver': ['lsqr', 'sparse_cg', 'sag', 'saga'],
             'tol': [10000, 1000, 100, 10, 1, 0.1, 0.001, 0.0001, 0.00001]} 

# alpha parameters for 0.0001, to 10000 by factors of 10
alpha_params = {'alpha': [10**i for i in range(-4,4)]}

# List of regressors
reg_regressors = [Ridge(random_state=RANDOM_SEED), Lasso(random_state=RANDOM_SEED), 
                  ElasticNet(random_state=RANDOM_SEED)]

"""### Ridge"""

ridge = Ridge(random_state=RANDOM_SEED)
tuned_ridge = GridSearchCV(estimator=ridge, param_grid=param_grid, cv=cv,
                     scoring='neg_mean_squared_error', refit=True)
tuned_ridge.fit(lim_X_train, lim_y_train)
#@ignore_warnings(category=ConvergenceWarning)

score = tuned_ridge.best_score_
best_rmse = np.sqrt(np.abs(score))
print('Best score: {}'.format(best_rmse))
print('Best parameters: {}'.format(tuned_ridge.best_params_))

"""### Lasso"""

lasso = Lasso(random_state=RANDOM_SEED)
tuned_lasso = gridsearch_alphas(lasso, alpha_params, lim_X_train, lim_y_train)

"""### ElasticNet"""

elasticnet = ElasticNet(random_state=RANDOM_SEED)
tuned_elasticnet = gridsearch_alphas(elasticnet, alpha_params, lim_X_train, lim_y_train)

"""# Linear Model Performance"""

def print_test_score(estimator, x_test, y_test):
    predictions = estimator.predict(x_test)
    score = np.sqrt(mean_squared_error(y_test, predictions))
    #print("Test score: {}".format(score))
    return score

"""## Ridge Regression"""

#tuned_ridge.fit(X_train, y_train)
ridge_predictions = tuned_ridge.predict(lim_X_test)

# scatter plot of predictions vs actual values

sns.scatterplot(x=ridge_predictions, y=lim_y_test)
plt.savefig("plot-ridge_pred_vs_test.pdf",
         bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
           orientation='portrait', papertype=None, format=None, pad_inches=0.25)

print('Test score: {}'.format(np.sqrt(mean_squared_error(lim_y_test, ridge_predictions))))

print_test_score(tuned_ridge, lim_X_test, lim_y_test)

"""## Linear Regression"""

final_lr = LinearRegression()
final_lr.fit(lim_X_train, lim_y_train)
lr_predictions = final_lr.predict(lim_X_test)
sns.scatterplot(x=lr_predictions, y=lim_y_test)
plt.savefig("plot-linear_regression_test_vs_pred.pdf",
         bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
           orientation='portrait', papertype=None, format=None, pad_inches=0.25)

print('Test score: {}'.format(np.sqrt(mean_squared_error(lim_y_test, lr_predictions))))

final_lr = LinearRegression()
final_lr.fit(lim_X_train, lim_y_train)
print_test_score(final_lr, lim_X_test, lim_y_test)

"""## Lasso"""

print_test_score(tuned_lasso, lim_X_test, lim_y_test)

"""## Elasticnet"""

print_test_score(tuned_elasticnet, lim_X_test, lim_y_test)

"""# Tree Modeling Experiment"""

from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, \
GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor, VotingRegressor
from sklearn.tree import DecisionTreeRegressor

# all features
x = [row[1:12] for row in model_data]
y = [row[0] for row in model_data] # median values with StandardScaler
log_y = [row[0] for row in log_model_data] # log median values with Standard Scaler

# Feature importances by Random Forests
# Would it help to plot this?
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=RANDOM_SEED)
rf.fit(x, y)
rf.feature_importances_

"""## Baseline Modeling"""

# can continue to implement additional models perform side by side 
seed = 1
reg_models = [LinearRegression(), Ridge(), Lasso(), ElasticNet(),
              RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED),
              RandomForestRegressor(max_features=1, n_estimators=100, random_state=seed),
              RandomForestRegressor(max_features="log2", n_estimators=100, random_state=seed),
              RandomForestRegressor(n_estimators=100, max_depth=2, random_state=seed),
              GradientBoostingRegressor(n_estimators=100, random_state=seed, learning_rate=0.1),
              GradientBoostingRegressor(n_estimators=100, random_state=seed, learning_rate=0.1, max_features = 1),
              GradientBoostingRegressor(n_estimators=100, random_state=seed, learning_rate=0.1, max_features ='log2'),
              GradientBoostingRegressor(n_estimators=100, random_state=seed, learning_rate=0.1, max_depth=2),
              BaggingRegressor(n_estimators = 10, random_state =seed)]
              
models = ['Linear Regression', 'Ridge', 'Lasso', 'ElasticNet',
          'Random Forest - No Max' ,'Random Forest - Max 1 features',
          'Random Forest - log2 Max Features', 'Random Forest -  Max 2 Depth',
          'Gradient Boost No Max', 'Gradient Boost Max 1 features', 
          'Gradient Boost Log2 max features', 'Gradient Boost Max 2 depth', 'Bagging Regressor']

def get_rmse(model, x, y):
    # This function returns the RMSE of a model after using cross validation
    return np.abs(cross_val_score(estimator = model, X=x, y=y, scoring = 'neg_mean_squared_error', cv=cv))

#print output and convert to rmse
for i,model in enumerate(reg_models):
    print('{} RMSE: {}'.format(models[i], np.mean(np.sqrt(get_rmse(model, x, y)))))

def get_rmse(model, x, log_y):
    # This function returns the RMSE of a model after using cross validation
    return np.abs(cross_val_score(estimator = model, X=x, y=log_y, scoring = 'neg_mean_squared_error', cv=cv))

#print output and convert to rmse
for i,model in enumerate(reg_models):
    print('{} RMSE: {}'.format(models[i], np.mean(np.sqrt(get_rmse(model, x, log_y)))))

# list of regressors
tree_regressors = [RandomForestRegressor(random_state=RANDOM_SEED), 
                   BaggingRegressor(random_state=RANDOM_SEED), 
                   GradientBoostingRegressor(random_state=RANDOM_SEED),
                   ExtraTreesRegressor(random_state=RANDOM_SEED), 
                   AdaBoostRegressor(random_state=RANDOM_SEED)]
# list of tree names
tree_names = ['Random Forests', 'Bagging', 'Gradient Boosting', 'Extra Trees',
              'Adaboost']

def baseline_tree_score(regressor, X, y):
    cv_scores = np.abs(cross_val_score(estimator = regressor, X=x, y=y, cv=cv,
                               scoring='neg_mean_squared_error'))
    mean_score = np.mean(np.sqrt(cv_scores))
    return mean_score
    #print(mean_score)

scores_list = list()
for i, regressor in enumerate(tree_regressors):
    mean_score = baseline_tree_score(regressor, x, y)
    scores_list.append(mean_score)

log_scores_list = list()
for i, regressor in enumerate(tree_regressors):
    mean_score = baseline_tree_score(regressor, x, log_y)
    log_scores_list.append(mean_score)

for i, score in enumerate(scores_list):
    print("{} : {}".format(tree_names[i], score))

"""Now for log median value with trees:"""

for i, score in enumerate(log_scores_list):
    print("{} : {}".format(tree_names[i], score))

"""## Tree Model Tuning on Full Dataset"""

# Train test split with all data
all_X_train, all_X_test, all_y_train, all_y_test = train_test_split(x, y, test_size = 0.25, 
                                                    random_state = RANDOM_SEED,
                                                    shuffle=True)

from sklearn.model_selection import RandomizedSearchCV
# This is the same function using RandomizedSearchCV 
def randomsearch_rmse(estimator, param_grid, X, y):
    search = RandomizedSearchCV(estimator=estimator, param_distributions=param_grid, # search parameters
                                n_iter=30,
                                cv=cv, scoring='neg_mean_squared_error',
                                refit=True,# refits model to best parameters
                                random_state = RANDOM_SEED) 
    search.fit(X, y)
    best_score = search.best_score_
    rmse = np.sqrt(np.abs(best_score))
    best_params = search.best_params_
    #print("Best Parameters: {}".format(best_params))
    #print("RMSE: {}".format(rmse))
    return search # returns the fitted model with best parameters

def print_search_details(search):
    best_params = search.best_params_
    best_score = search.best_score_
    rmse = np.sqrt(np.abs(best_score))
    print("Best Parameters: {}".format(best_params))
    print("RMSE: {}".format(rmse))

"""### Bagging Regressor"""

tune_start = time.time()
bagging_grid = {'n_estimators':[10**i for i in range(1, 4)],
             'max_samples':[i for i in range(1,6)],
             'max_features':[i for i in range(1,10)]
             }
bag_regressor = BaggingRegressor(random_state=RANDOM_SEED, n_jobs=-1)
optimized_bag = randomsearch_rmse(bag_regressor, bagging_grid, all_X_train, all_y_train)
print_search_details(optimized_bag)

"""### Random Forest"""

rf_grid = {'n_estimators': [10**i for i in range(1, 4)],
           'max_depth': [i for i in range(1,10)],
           'max_features': [i for i in range(1,10)]
           }
rf_regressor = RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1)
optimized_rf = randomsearch_rmse(rf_regressor, rf_grid, all_X_train, all_y_train)
print_search_details(optimized_rf)

"""### Gradient Boosting Regressor"""

gradient_boost_grid = {'n_estimators': [10**i for i in range(1, 4)],
                       'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.8],
                       'max_depth': [i for i in range(1,10)],
                       'max_features': [i for i in range(1,10)]
                       }
gb_regressor = GradientBoostingRegressor(random_state=RANDOM_SEED)
optimized_gb = randomsearch_rmse(gb_regressor, gradient_boost_grid, all_X_train, all_y_train)
print_search_details(optimized_gb)

"""### Extra Trees Regressor"""

extratrees_grid = {'n_estimators': [10**i for i in range(1, 4)],
                   'max_depth': [i for i in range(1,10)],
                   'max_features': [i for i in range(1,10)]
                   }
et_regressor = ExtraTreesRegressor(random_state=RANDOM_SEED, n_jobs=-1)
optimized_et = randomsearch_rmse(et_regressor, extratrees_grid, all_X_train, all_y_train)
print_search_details(optimized_et)

"""### Adaboost"""

adaboost_grid = {'n_estimators': [10**i for i in range(1, 4)],
                 'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.8],
                 'loss': ['linear', 'square', 'exponential']
                 }
ada_regressor = AdaBoostRegressor(random_state=RANDOM_SEED)
optimized_ada = randomsearch_rmse(ada_regressor, adaboost_grid, all_X_train, all_y_train)
print_search_details(optimized_ada)

tune_end = time.time()

tuning_minutes_elapsed = (tune_end - tune_start)/60
print(tuning_minutes_elapsed)

"""# Tree Model Performance on Full Dataset"""

def print_test_score(estimator, x_test, y_test):
    predictions = estimator.predict(x_test)
    score = np.sqrt(mean_squared_error(y_test, predictions))
    #print("Test score: {}".format(score))
    return score

"""## Bagging Regressor"""

print_test_score(optimized_bag, all_X_test, all_y_test)

"""## Random Forests"""

print_test_score(optimized_rf, all_X_test, all_y_test)

"""## Gradient Boosting"""

print_test_score(optimized_gb, all_X_test, all_y_test)

"""## Extra Trees"""

print_test_score(optimized_et, all_X_test, all_y_test)

"""## Adaboost"""

print_test_score(optimized_ada, all_X_test, all_y_test)

"""## Testing conclusion: 

The random search optimized extra trees produces the best root mean squared error at 0.35159057123119847 which is far better than any linear classifier did.

# Tree Tuning and Testing on Selected Features

## All Trees using Ben's Loop
"""

# list of regressors
tree_regressors = [RandomForestRegressor(random_state=RANDOM_SEED), 
                   BaggingRegressor(random_state=RANDOM_SEED), 
                   GradientBoostingRegressor(random_state=RANDOM_SEED),
                   ExtraTreesRegressor(random_state=RANDOM_SEED), 
                   AdaBoostRegressor(random_state=RANDOM_SEED)]
# list of tree names
tree_names = ['Random Forests', 'Bagging', 'Gradient Boosting', 'Extra Trees',
              'Adaboost']

# Gridsearch parameter grids for limited features
bagging_grid = {'n_estimators':[10**i for i in range(1, 4)],
             'max_samples':[i for i in range(1,6)],
             'max_features':[i for i in range(1,5)] # changed max features here to fit in new feature set
             }

rf_grid = {'n_estimators': [10**i for i in range(1, 4)],
           'max_depth': [i for i in range(1,10)],
           'max_features': [i for i in range(1,5)]
           }

gradient_boost_grid = {'n_estimators': [10**i for i in range(1, 4)],
                       'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.8],
                       'max_depth': [i for i in range(1,10)],
                       'max_features': [i for i in range(1,5)]
                       }
                 
adaboost_grid = {'n_estimators': [10**i for i in range(1, 4)],
                 'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.8],
                 'loss': ['linear', 'square', 'exponential']
                 }
        
extratrees_grid = {'n_estimators': [10**i for i in range(1, 4)],
                   'max_depth': [i for i in range(1,10)],
                   'max_features': [i for i in range(1,5)]
                   }

grids = [rf_grid, bagging_grid, gradient_boost_grid, extratrees_grid, adaboost_grid]

def tune_and_score_tree(tree, tree_name, params, X_train, y_train, X_test, y_test):
    search = randomsearch_rmse(tree, params, X_train, y_train)
    #print(tree_name)
    rmse = print_test_score(search, X_test, y_test)
    return rmse

start_time = time.time()
tree_scores = list() # store name and according score
for i in range(len(tree_names)):
    tree_scores.append(tune_and_score_tree(tree_regressors[i], tree_names[i], 
                                           grids[i], lim_X_train, lim_y_train, 
                                           lim_X_test, lim_y_test))
    
end_time = time.time()

tree_scores

time_elapsed = (end_time - start_time) / 60
print("Minutes elapsed: {}".format(time_elapsed))

"""## Randomized Search and Tests by Noah and Vani

### Random Forest
"""

# # Vani
# from sklearn.model_selection import GridSearchCV
# param_grid=[{'bootstrap':[True,False], 'n_estimators': [3,10,15,20], 'max_features':[2,3,4,6,8]}]
# forest_reg = RandomForestRegressor ()
# grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
# grid_search.fit(x,y)
# print ('the best_params : ', grid_search.best_params_)
# print ('the best_estimator : ', grid_search.best_estimator_)
# print ('the best_score : ', grid_search.best_score_)

param_grid = {'n_estimators': [1000, 100, 10, 1], #the default is 100
              'max_depth': [None, 1, 2, 3],
             'min_samples_split': [2,3,10,10],
             'min_samples_leaf': [1,2,3,10],
             'max_features': ['auto', 1,2, 'log2','sqrt'],
             'max_leaf_nodes': [None, 2,10,100]} 
random_forest = RandomForestRegressor(random_state=RANDOM_SEED)

random_forest_search = RandomizedSearchCV(estimator=random_forest, # input estimator from above
                      param_distributions = param_grid, # set parameter distributions to values we want to test
                      cv=cv, # state CV to above shufflesplit one
                      scoring='neg_mean_squared_error', # set scoring mode
                      n_iter=50,
                      random_state=RANDOM_SEED,
                      refit=True) # don't refit to the best parameters
random_forest_search.fit(lim_X_train, lim_y_train)
random_forest_predictions = random_forest_search.predict(lim_X_test)
random_forest_score = mean_squared_error(lim_y_test, random_forest_predictions)
print("Random Forest Score: {}".format(np.sqrt(random_forest_score)))

"""### Gradient Boosting"""

gradientboost = GradientBoostingRegressor(random_state=RANDOM_SEED)
param_grid = {'n_estimators': [1000, 100, 10, 1], #the default is 100
              'max_depth': [None, 1, 2, 3,10],
             'min_samples_split': [2,3,10,10],
             'min_samples_leaf': [1,2,3,10],
             'max_features': ['auto', 1,2, 'log2','sqrt'],
             'max_leaf_nodes': [None, 2,10,100],
             'learning_rate': [0.1,1]} 

gradientboost_search = RandomizedSearchCV(estimator=gradientboost, # input estimator from above
                      param_distributions = param_grid, # set parameter distributions to values we want to test
                      cv=cv, # state CV to above shufflesplit one
                      scoring='neg_mean_squared_error', # set scoring mode
                      n_iter=50,
                      random_state=RANDOM_SEED,
                      refit=True) # don't refit to the best parameters
gradientboost_search.fit(lim_X_train, lim_y_train)
print("Best Parameters: {}".format(gradientboost_search.best_params_))
gradientboost_predictions = gradientboost_search.predict(lim_X_test)
gradientboost_score = mean_squared_error(lim_y_test, gradientboost_predictions)
print("Gradient Boosting Score: {}".format(np.sqrt(gradientboost_score)))

"""### Extra Trees"""

# extra_tree = ExtraTreesRegressor(random_state=RANDOM_SEED)

# param_grid = {'n_estimators': [1000, 100, 10, 1], #the default is 100
#               'max_depth': [None, 1, 2, 3,10],
#              'min_samples_split': [2,3,10,10],
#              'min_samples_leaf': [1,2,3,10],
#              'max_features': ['auto', 1,2, 'log2','sqrt'],
#              'max_leaf_nodes': [None, 2,10,100]} 
# extra_tree_search = RandomizedSearchCV(estimator=extra_tree, # input estimator from above
#                       param_distributions = param_grid, # set parameter distributions to values we want to test
#                       cv=cv, # state CV to above shufflesplit one
#                       scoring='neg_mean_squared_error', # set scoring mode
#                       n_iter=50,
#                       random_state=RANDOM_SEED,
#                       refit=True) # don't refit to the best parameters
# extra_tree_search.fit(lim_X_train, lim_y_train)
# extra_tree_predictions = extra_tree_search.predict(lim_X_test)
# extra_tree_score = mean_squared_error(lim_y_test, extra_tree_predictions)
# print("Extra Trees Score: {}".format(np.sqrt(extra_tree_score)))

"""### Adaboost"""

adaboost = AdaBoostRegressor(random_state=RANDOM_SEED)
param_grid = {'n_estimators': [1000, 100, 10, 1], 
              'learning_rate': [.1,1,10, 100,100]} 
adaboost_search = RandomizedSearchCV(estimator=adaboost, # input estimator from above
                      param_distributions = param_grid, # set parameter distributions to values we want to test
                      cv=cv, # state CV to above shufflesplit one
                      scoring='neg_mean_squared_error', # set scoring mode
                      n_iter=50,
                      random_state=RANDOM_SEED,
                      refit=True) # don't refit to the best parameters
adaboost_search.fit(lim_X_train, lim_y_train)
adaboost_predictions = adaboost_search.predict(lim_X_test)
adaboost_score = mean_squared_error(lim_y_test, adaboost_predictions)
print("ADA Boosting Score: {}".format(np.sqrt(adaboost_score)))

"""# Regressor Conclusions

## Score Analysis
"""

estimator_names = list()
estimator_scores = list()

linear_regressors = models[0:4]

tree_names

tree_scores

tree_scores_copy = tree_scores.copy()

# Replace gradient boosting score with Noah's score
tree_scores_copy[2] = np.sqrt(gradientboost_score)

# Replace Adaboost score with Noah's
tree_scores_copy[4] = np.sqrt(adaboost_score)

tree_scores_copy

tree_scores_zip = zip(tree_names, tree_scores_copy)
list(tree_scores_zip)

# Compile complete list of names of regressors
for estimator in linear_regressors:
    estimator_names.append(estimator)
for tree in tree_names:
    estimator_names.append(tree)

estimator_names

# list of optimized linear regressors
optimized_linear = [final_lr, tuned_ridge, tuned_lasso, tuned_elasticnet]

# Add linear test score values to a list
for estimator in optimized_linear:
    estimator_scores.append(print_test_score(estimator, lim_X_test, lim_y_test))

# Add tree score values to list
for score in tree_scores_copy:
    estimator_scores.append(score)

estimator_scores

scores_dict = dict(zip(estimator_names, estimator_scores))
scores_dict

# Series of all tested scores on optimized estimators
scores = pd.Series(estimator_scores, index=estimator_names)
sorted_scores = scores.sort_values()

fig, ax = plt.subplots(figsize = (9,4), facecolor='w')
sns.barplot(y=sorted_scores.index, x=sorted_scores, color='blue')
plt.xlabel('Root Mean-Squared Error')
plt.ylabel('Optimized Estimators')
plt.title('Sorted RMSE of Optimized Estimators')
fig.savefig("plot-RMSE-Optimized-Estimators.pdf", bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', orientation='portrait', papertype=None, format=None, pad_inches=0.25)
plt.show()

"""The tuned Gradient Boosting regressor is shows the best RMSE score over the tuned random forests and extra trees regressors by a slim margin.

## Feature Importances
"""

from operator import itemgetter
# Feature names to extract: nox(4), rooms(5), dis(7), ptratio(10), lstat(11)
indices = [4, 5, 7, 10, 11]
columns = list(pscaled_features_df.columns)

extracted_columns = itemgetter(*indices)(columns)

gb_reg = GradientBoostingRegressor(n_estimators=1000, min_samples_split=3,
                                   min_samples_leaf=2, max_leaf_nodes= None,
                                   max_features=1, max_depth=10, learning_rate=0.1, random_state=seed)
gb_reg.fit(lim_X_train, lim_y_train)
feature_importances = gb_reg.feature_importances_

feature_importances.sort()

features_importances_dict = dict(zip(extracted_columns, feature_importances))
features_importances_dict

combined_features = list((extracted_columns,feature_importances))

# Convert list to DataFrame and sort by Feature Importance value
combined_features_df = pd.DataFrame(combined_features).T
combined_features_df.columns = ['Features', 'Importance']
combined_features_df.sort_values(by='Importance', inplace=True)

# Seaborn horizontal bar plot
fig, ax = plt.subplots(figsize = (9,3), facecolor='w')
sns.barplot(x='Importance', y='Features', data=combined_features_df, orient = 'h').set_title("GradiantBoostingReg - Sorted Feature Importance")
fig.savefig("plot-GradiantBoostingReg-Sorted-Feature-Importance.pdf", bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', orientation='portrait', papertype=None, format=None, pad_inches=0.25)

"""# Voting Ensemble Experimentation"""

# Ali's Attempt

# Used the best parameters for the algorithms in Voting Regressors

# Voting Regressor with GradiantBoosting, RandomForest and Linear
reg1 = GradientBoostingRegressor(n_estimators=100, random_state=seed, learning_rate=0.1, max_features ='log2')
reg2 = RandomForestRegressor(max_features="log2", n_estimators=100, random_state=seed)
reg3 = LinearRegression()
ereg = VotingRegressor(estimators=[('gb', reg1), ('rf', reg2), ('lr', reg3)])
ereg = ereg.fit(x, log_y)

reg1.fit(x, log_y)
reg2.fit(x, log_y)
reg3.fit(x, log_y)
ereg.fit(x, log_y)

# Score (higher the better) and RMSE (lower the better)
print('Score Voting Regressor with GradiantBoosting, RandomForest and Linear', ereg.score(x, log_y))
print('Voting Regressor with LinearReg RMSE: ', np.mean(np.sqrt(get_rmse(ereg, x, log_y))),'\n')
RMSE_values = []
RMSE_values.append(np.mean(np.sqrt(get_rmse(ereg, x, log_y))))
RMSE_names = []
RMSE_names.append('VotingReg GB, RF and Linear')
plot_ereg1 = np.mean(np.sqrt(get_rmse(ereg, x, log_y)))

# Voting Regressor with GradiantBoosting and RandomForest
reg1 = GradientBoostingRegressor(n_estimators=100, random_state=seed, learning_rate=0.1, max_features ='log2')
reg2 = RandomForestRegressor(max_features="log2", n_estimators=100, random_state=seed)
ereg = VotingRegressor(estimators=[('gb', reg1), ('rf', reg2)])
ereg = ereg.fit(x, log_y)

reg1.fit(x, log_y)
reg2.fit(x, log_y)
ereg.fit(x, log_y)

# Score (higher the better) and RMSE (lower the better)
print('Score Voting Regressor with GradiantBoosting and RandomForest', ereg.score(x, log_y))
print('Voting Regressor without LinearReg RMSE: ', np.mean(np.sqrt(get_rmse(ereg, x, log_y))),'\n')
RMSE_values.append(np.mean(np.sqrt(get_rmse(ereg, x, log_y))))
RMSE_names.append('VotingReg GB and RF')
plot_ereg2 = np.mean(np.sqrt(get_rmse(ereg, x, log_y)))

names = ['Voting w/ Linear', 'Voting w/o Linear']
values = [plot_ereg1, plot_ereg2]

# Append model results to list
for i,model in enumerate(reg_models):
  print('{} RMSE: {}'.format(models[i], np.mean(np.sqrt(get_rmse(model, x, log_y)))))
  RMSE_values.append(np.mean(np.sqrt(get_rmse(model, x, log_y))))
  RMSE_names.append(models[i])

# Combine the results from the models into a list
combined = list((RMSE_names,RMSE_values))

# Convert list to DataFrame and sort by RMSE value
rmse_df = pd.DataFrame(combined).T
rmse_df.columns = ['Algorithm', 'RMSE']
rmse_df.sort_values(by='RMSE', inplace=True)
print(rmse_df)

# Seaborn horizontal bar plot
fig, ax = plt.subplots(figsize=(9,5))
sns.barplot(x='RMSE', y='Algorithm', data=rmse_df, orient = 'h')
plt.xlabel('RMSE')
plt.ylabel('Models')
plt.title('Performance (RMSE) of various algorithms.')
fig.savefig("plot-RMSE-Performance-Models-Seaborn.pdf", bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', orientation='portrait', papertype=None, format=None, pad_inches=0.25)
plt.show()

# Horizontal bar plot
fig, ax = plt.subplots(figsize=(9,5))
ax.barh(rmse_df['Algorithm'], rmse_df['RMSE'], align='center')
ax.set_yticks(RMSE_names)
ax.set_yticklabels(RMSE_names)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('RMSE')
ax.set_title('Performance (RMSE) of various algorithms.')
fig.savefig("plot-RMSE-Performance-Models.pdf", bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', orientation='portrait', papertype=None, format=None, pad_inches=0.25)
plt.show()

vote1_start = time.time()
testclf = VotingRegressor([('lr', final_lr), ('tree', optimized_rf)])
testclf.fit(all_X_train, all_y_train)
vote1_end = time.time()

vote1_elapsed_time = (vote1_end - vote1_start) / 60
print("{0:.2f} minutes elapsed".format(vote1_elapsed_time))

test_preds = testclf.predict(all_X_test)
test_score = mean_squared_error(all_y_test, test_preds)
test_rmse_score = np.sqrt(test_score)
print(test_rmse_score)

# best linear regressor + best tree regressor
vote2_start_time = time.time()
testclf2 = VotingRegressor([('lr', tuned_lasso), ('tree', optimized_et)])
testclf2.fit(all_X_train, all_y_train)
vote2_end_time = time.time()

vote2_elapsed_time = (vote2_end_time - vote2_start_time) / 60
print("{0:.2f} minutes elapsed".format(vote2_elapsed_time))

test2_preds = testclf2.predict(all_X_test)
test2_score = mean_squared_error(all_y_test, test2_preds)
test2_rmse_score = np.sqrt(test2_score)
print(test2_rmse_score)

"""**Big caveat:** The optimized linear regressors are trained on a limited set of features. Using every feature created worse linear models. However, all tree models are optimized and then trained on all features. The dataset used in these tests is with all features.

**Second caveat:** If time is a concern, I may skip this all together. It takes forever to run, and optimizing trees already takes a long time.
"""

vote_et = ExtraTreesRegressor(n_estimators=10, max_features=3, max_depth=9, random_state=RANDOM_SEED)

vote3 = VotingRegressor([('lr', tuned_lasso), ('tree', vote_et)])
vote3.fit(lim_X_train, lim_y_train)

vote3_preds = vote3.predict(lim_X_test)
vote3_score = mean_squared_error(lim_y_test, vote3_preds)
vote3_rmse = np.sqrt(vote3_score)
print(vote3_rmse)

"""Using our selected features, the RMSE is still not as good as tree models themselves.

# Conclusion

A voting regressor doesn't improve RMSE beyond most of the ensemble tree methods. Would not recommend running ever again because they took forever.
"""

total_end = time.time()

total_runtime = (total_end - total_start) / 60
print("Total runtime: {0:.2f}".format(total_runtime)) # the part inside the brackets formats the decimals to 2 places

"""# Appendix

## Variables

* **x** - all variables scaled with StandardScaler
* **y** - median values scaled with StandardScaler
* **log_y** - log transformed median values scaled with StandardScaler
* **unscaled_x** - all variables without transformation
* **unscaled_y** - unscaled median values
* **unscaled_log_y** - unscaled log transformed median values
* **pscaled_x** -  nox, rooms, dis, ptratio, lstat scaled by PowerScaler
* **lim_X_train** - selected features (from pscaled_x) training set
* **lim_X_test** - selected features (from pscaled_x) test set
* **lim_y_train** - selected features (from pscaled_x) training predictions
* **lim_y_test** - selected features (from pscaled_x) true predictions
* **reg_regrssors** - list of linear regressors with random state set to RANDOM_SEED
* **all_X_train** - training set with all features
* **all_X_test** - testing set with all features
* **all_y_train** - associated training set with all features
* **all_y_test** - associated test set with all features
* **tree_regrssors** - list of tree regressors with random state set to RANDOM_SEED
* **tree_names** - string format names of tree regressors, same order as tree_regressors
* **estimator_names** - string names of all estimators
* **estimator_scores** - RMSE scores of associated estimators, corresponds to **estimator_names**

### Tuned Models

* **final_lr** - linear regressor
* **tuned_ridge** 
* **tuned_lasso**
* **tuned_elasticnet**
* optimized_bag
* optimized_rf
* optimized_gb
* optimized_et
* optimized_ada
"""