
# -*- coding: utf-8 -*-



# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

# %matplotlib inline

# Sklearn modules
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler, Normalizer, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.decomposition import PCA, NMF
from sklearn.metrics import f1_score

# Start time for entire document
script_start = time.time()

random_seed = 1

"""## Import data"""

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)

print(mnist.DESCR)

mnist.keys()

X, y = mnist['data'], mnist['target']

X.shape

y.shape

"""## Common functions"""

# time conversion
def runtime_in_min(start_time, end_time):
    """
    This function takes an end time point and a start time point
    and converts the result to minutes.
    """
    return((end_time - start_time) / 60)

# Pipeline function

"""## Data Split"""

# first 60000 as training set, final 10000 as held out testing set
from sklearn.model_selection import train_test_split

# Define Test Size (Train Size = 1 - Test Size)
test_size = 10000 / 70000

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                    train_size=(1-test_size),
                                                    random_state=random_seed,
                                                    shuffle=False)

X_train.shape

X_test.shape

"""# Requirement 1

Begin by fitting a random forest classifier using the full set of 784 explanatory variables and the model development set of 60,000 observations. Record the time it takes to fit the model and evaluate the model on the holdout data. Assess classification performance using the F1 score, which is the harmonic mean of precision and recall.
"""

base_start = time.time()

base_rf = RandomForestClassifier(bootstrap=True, max_features="sqrt", n_estimators=10, random_state=random_seed)
base_rf.fit(X_train, y_train)

base_end = time.time()

base_rf_preds = base_rf.predict(X_test)
base_rf_score = f1_score(y_test, base_rf_preds, average='micro')

# Computer Total Time
#time_minutes_elapsed = (time_end - time_start)/60
base_rf_runtime = runtime_in_min(base_start, base_end)
print('Model fit runtime in minutes (784 features, 60000 observations): ', round(base_rf_runtime, 4), '\n')

print('F1 score (784 variables, 60000 observations):', base_rf_score)

"""# Requirement 2

Execute principal components analysis (PCA) on the full set of 70,000, generating principal components that represent 95 percent of the variability in the explanatory variables. The number of principal components in the solution should be substantially fewer than the 784 explanatory variables. Record the time it takes to identify the principal components.
"""

# Use StandardScaler first
sscaler = StandardScaler()
ss_X_train = sscaler.fit_transform(X_train)
ss_X_test = sscaler.transform(X_test)
ss_X = sscaler.fit_transform(X)

full_pca_start = time.time()

full_pca = PCA(n_components=0.95, random_state=random_seed)
full_pca.fit_transform(ss_X)

full_pca_end = time.time()

full_pca_runtime = runtime_in_min(full_pca_start, full_pca_end)
full_pca_n_components = full_pca.n_components_
print("PCA (full dataset) runtime: {0:.3f} minutes".format(full_pca_runtime))
print("PCA (full dataset) n components: {}".format(full_pca_n_components))
full_explained_variance = full_pca.explained_variance_

"""# Requirement 3

Using the identified principal components from step (2), use the first 60,000 observations to build another random forest classifier. Record the time it takes to fit the model and to evaluate the model on the holdout data (the last 10,000 observations). Assess classification performance using the F1 score, which is the harmonic mean of  precision and recall.
"""

# transform first 60000 and last 10000
full_X_train = full_pca.transform(X_train)
full_X_test = full_pca.transform(X_test)

full_rf_start = time.time()
full_rf = RandomForestClassifier(bootstrap=True, n_estimators=10,
                                 random_state=random_seed, max_features='sqrt')
full_rf = full_rf.fit(full_X_train, y_train)
full_rf_end = time.time()

full_rf_preds = full_rf.predict(full_X_test)
full_rf_runtime = runtime_in_min(full_rf_start, full_rf_end)
full_rf_score = f1_score(y_test, full_rf_preds, average='micro')

print("RandomForest (PCA on full dataset) runtime: {0:.3f} minutes".format(full_rf_runtime))
print("RandomForest (PCA on full dataset) F1 score: {0:.3f}".format(full_rf_score))

"""# Requirement 4

Compare test set performance across the two modeling approaches: original 784-variable model versus the 95-percent-PCA model. Also evaluate the time required to perform (1) versus the time required to perform (2) and (3) together. Ensure that accurate measures are made of the total time it takes to execute each of the modeling approaches in training the models. Some guidance on the coding of benchmark studies may be found in the Python code under:
"""

# Compare scores
print('F1 score (784 variables, 60000 observations): {0:.5f}'.format(base_rf_score))
print("F1 Score RandomForest (PCA on full dataset): {0:.5f}".format(full_rf_score))

# Compare runtimes 
pca_model_fit_runtime = full_pca_runtime + full_rf_runtime
print('Model fit runtime in minutes (784 features, 60000 observations): ', round(base_rf_runtime, 4))
print('Model fit runtime in minutes (PCA fit, 332 features, 60000 observations): {0:.5f}'.format(pca_model_fit_runtime))

"""# Requirement 5

The experiment we have proposed has a design flaw. Identify the flaw. Fix it. Rerun the experiment in a way that is consistent with a training-and-test regimen.

**Answer**: The flaw is PCA was run on the the entire dataset. The test set should be unseen by the model until the very end. The fact that PCA was fit on the 10000 observations that are also in the test set make the scores flawed.
"""

# pipeline inputs
pca = PCA(n_components=0.95, random_state=random_seed)
rf = RandomForestClassifier(bootstrap=True, n_estimators=10,
                                 random_state=random_seed, max_features='sqrt')

pipe_start = time.time()
model_pipe = Pipeline([
                       ('scaler', StandardScaler()),
                       ('pca', pca),
                       ('classifier', rf)
])

model_pipe.fit(X_train, y_train)

pipe_end = time.time()

pipe_preds = model_pipe.predict(X_test)
pipe_score = f1_score(y_test, pipe_preds, average='micro')
pipe_runtime = runtime_in_min(pipe_start, pipe_end)

print("F1 Score Pipeline(Scaler, PCA, RandomForest): {0:.5f}".format(pipe_score))
print('Pipe runtime: {0:.5f}'.format(pipe_runtime))

def run_pipeline(seed):
    pca = PCA(n_components=0.95, random_state=random_seed)
    rf = RandomForestClassifier(bootstrap=True, n_estimators=10,
                                 random_state=seed, max_features='sqrt')
    model_pipe = Pipeline([
                       ('scaler', StandardScaler()),
                       ('pca', pca),
                       ('classifier', rf)
    ])

    model_pipe.fit(X_train, y_train)
    pipe_preds = model_pipe.predict(X_test)
    pipe_score = f1_score(y_test, pipe_preds, average='micro')
    print(pipe_score)

"""# Assignment Runtime"""

assignment_end = time.time()
assignment_runtime = runtime_in_min(script_start, assignment_end)
print("Assignment runtime: {0:3f} minutes".format(assignment_runtime))

"""# Exploration"""

seeds = np.random.randint(100, size = 10)

print(seeds)

for i in seeds:
    run_pipeline(i)

some_digit = X[5]
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap='binary')
plt.axis('off')
plt.savefig("plot-digit-image-example.pdf",
         bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
            orientation='portrait', papertype=None, format=None, pad_inches=0.25)
plt.show()


# Creates a dataframe of mnist data and feature names
mnist_df = pd.DataFrame(mnist.data, columns=mnist.feature_names)
mnist_df.head()

# Standarize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# PCA
from sklearn.decomposition import PCA 

# Start Timer
time_start = time.time()

pca = PCA(.95, random_state=random_seed)
X_train = pca.fit_transform(X_train) 
X_test = pca.transform(X_test) 

# End Timer
time_end = time.time()

# Computer Total Time
time_minutes_elapsed = (time_end - time_start)/60
print('Total Duration in Minutes: ', round(time_minutes_elapsed, 4), '\n')

variance = pca.explained_variance_ratio_

print("List of components and their variances: ", '\n', variance, '\n')

print("Number of Components to get to 95% Sum of Variance Ratio: ", pca.n_components_, '\n')

print("95-percent-PCA model:", pca.explained_variance_ratio_.sum(), '\n')

X_train.shape

X_test.shape

# import time
# time_start = time.time()

# rf = RandomForestClassifier(bootstrap=True, max_features="sqrt", n_estimators=10, random_state=random_seed)
# rf.fit(X_train, y_train)

# preds = rf.predict(X_test)

# from sklearn.metrics import f1_score

# score = f1_score(y_test, preds, average='micro')

# time_end = time.time()

# # Computer Total Time
# #time_minutes_elapsed = (time_end - time_start)/60
# time_elapsed = runtime_in_min(time_start, time_end)
# print('Total Duration in Minutes (Principal comp variables, 60000 observations): ', round(time_elapsed, 4), '\n')

# print('F1 score (Principal comp variables, 60000 observations):', score)

# Plot the variance of the n_components
fig, ax = plt.subplots(figsize = (12,4), facecolor='w')
plt.bar(x = range(1, len(variance)+1), height=variance, width=0.7)
plt.title("Individual variance for each component.")
plt.savefig("plot-variance-n_components.pdf",
         bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
            orientation='portrait', papertype=None, format=None, pad_inches=0.25)
plt.show()

percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_);
cum_var_explained = np.cumsum(percentage_var_explained)

# Plot the PCA spectrum for 95%
fig, ax = plt.subplots(figsize = (12,4), facecolor='w')
plt.plot(cum_var_explained, linewidth=2)
plt.axis("tight")
plt.grid()
plt.xlabel("n_components")
plt.ylabel("Cumulative_explained_variance")
plt.title("Cumulative Explained Variance for 95%.")
plt.savefig("plot-spectrum-pca-95%.pdf",
         bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
            orientation='portrait', papertype=None, format=None, pad_inches=0.25)
plt.show()

# Plot the PCA spectrum for all dataset
fig, ax = plt.subplots(figsize = (12,4), facecolor='w')
#pca = PCA().fit(mnist_df)
plt.plot(np.cumsum(pca.explained_variance_ratio_), linewidth=2)
plt.title('Cumulative Explained Variance over Principal Components')
plt.ylabel('Cumulative Explained variance')
plt.xlabel('Principal components')
plt.axis("tight")
plt.grid()
plt.axhline(y = .95, color='r', linestyle='--', label = '95% Explained Variance')
plt.axhline(y = .90, color='g', linestyle='--', label = '90% Explained Variance')
plt.axhline(y = .85, color='y', linestyle='--', label = '85% Explained Variance')
plt.legend(loc='best')
plt.savefig("plot-spectrum-pca-pct-explained-variance.pdf",
         bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
            orientation='portrait', papertype=None, format=None, pad_inches=0.25)
plt.show()

"""## Pipeline example and explanation

## What the code is doing

First, start with a fresh train, test split with no scaling, no pca, no fitting to anything.
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                    train_size=(1-test_size),
                                                    random_state=random_seed,
                                                    shuffle=False)

"""Next, apply the scaler. 

1. Remember to fit AND transform to the training set.
2. Then transform the test set which was fitted to the training set. 

This is because you don't want to standardize your test set. You want to make it fit into how your training set is standardized.
"""

scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

"""Now, set PCA to 0.95 variance explained and maintain the random_state."""

pca = PCA(n_components=0.95, random_state=random_seed)
pca_X_train = pca.fit_transform(scaled_X_train)
pca_X_test = pca.transform(scaled_X_test)

"""The classifier parameters are arbitrary here. I'm just showing it as an example."""

rf = RandomForestClassifier(n_estimators=10, random_state=random_seed,
                            max_features='sqrt')
rf.fit(pca_X_train, y_train)
rf_preds = rf.predict(pca_X_test)

"""And test score!"""

f1_score(y_test, rf_preds, average='micro')

"""## With Pipeline library

These are copied and pasted from the above code section.
"""

scaler = StandardScaler()
pca = PCA(n_components=0.65, random_state=random_seed)
rf = RandomForestClassifier(random_state=random_seed,
                            max_features='sqrt')

"""Make the pipeline. It's a list of tuples. The estimator should be last."""

pipe = Pipeline([
                 ('scaler', StandardScaler()),
                 ('pca', pca),
                 ('rf', rf)
])

pipe.fit(X_train, y_train)
pipe_preds = pipe.predict(X_test)

f1_score(y_test, pipe_preds, average='micro')

test_start = time.time()
pca = PCA(n_components=0.5, random_state=random_seed)

rf = RandomForestClassifier(n_estimators=25, random_state=random_seed,
                            max_features='sqrt')

pipe = Pipeline([
                 ('scaler', StandardScaler()),
                 ('pca', pca),
                 ('rf', rf)
])

pipe.fit(X_train, y_train)
pipe_preds = pipe.predict(X_test)
print(f1_score(y_test, pipe_preds, average='micro'))

test_end = time.time()
print(runtime_in_min(time_start, time_end))

"""## Pipeline and GridSearchCV

Feel free to comment this back in to run it. It took me about 8 minutes just to run this small grid. I didn't bother running anything with more estimators because we have so many rows of data.
"""

# from sklearn.model_selection import GridSearchCV

# grid_start = time.time()

# param_grid = {
#     'pca__n_components':[0.95, 0.85, 0.65], # you can adjust things along the pipe with a big parameter grid
#     'rf__n_estimators': [15, 25],
#     'rf__max_features': ['sqrt']
# }

# pipe = Pipeline([
#                  ('scaler', StandardScaler()),
#                  ('pca', pca),
#                  ('rf', rf)
# ])

# search = GridSearchCV(pipe, param_grid=param_grid, 
#                       cv=3, # standard is regular ole kfold
#                       scoring='f1_micro',
#                       refit=True)
# search.fit(X_train, y_train)
# #search_preds = search.predict(X_test)

# grid_end = time.time()
# print(runtime_in_min(grid_start, grid_end))

"""Just a few things took about 8 minutes to run. I don't want to do this again, so I think we should skip GridSearch overall."""

# print(search.best_params_)
# search_preds = search.predict(X_test)
# f1_score(y_test, search_preds, average='micro')

"""## Other model testing"""

gb_start = time.time()

pca = PCA(n_components=0.75, random_state=random_seed)
pca_X_train = pca.fit_transform(scaled_X_train)
pca_X_test = pca.transform(scaled_X_test)

gb = GradientBoostingClassifier(n_estimators=20, max_features='sqrt', random_state=random_seed)
gb.fit(X_train, y_train)
gb_preds = gb.predict(X_test)

print(f1_score(y_test, gb_preds, average='micro'))

gb_end = time.time()
print(runtime_in_min(gb_start, gb_end))

gb_start = time.time()
gb = GradientBoostingClassifier(n_estimators=20, max_features='sqrt', random_state=random_seed)
gb.fit(pca_X_train, y_train)
gb_preds = gb.predict(pca_X_test)

print(f1_score(y_test, gb_preds, average='micro'))

gb_end = time.time()
print(runtime_in_min(gb_start, gb_end))

et = ExtraTreesClassifier(n_estimators=20, max_features='sqrt', random_state=random_seed)
et.fit(X_train, y_train)
et_preds = et.predict(X_test)

print(f1_score(y_test, et_preds, average='micro'))

pca = PCA(n_components=0.95, random_state=random_seed)
pca_X_train = pca.fit_transform(scaled_X_train)
pca_X_test = pca.transform(scaled_X_test)

et = ExtraTreesClassifier(n_estimators=20, max_features='sqrt', random_state=random_seed)
et.fit(pca_X_train, y_train)
et_preds = et.predict(pca_X_test)

print(f1_score(y_test, et_preds, average='micro'))

"""## GridSearch 

* Search between 3 different scalers
* Use PCA explained variance 0.95, 0.85, 0.7
* Try randomforest, extratrees, and gradientboosting classifiers
"""

# # Scalers

# ss = StandardScaler()
# pt = PowerTransformer()
# norm = Normalizer()

# # Dim reduction
# pca = PCA(n_components=0.95, random_state=random_seed)

# # Classifiers
# rf = RandomForestClassifier(n_estimators=20, random_state=random_seed,
#                             bootstrap=True, max_features='sqrt')
# et = ExtraTreesClassifier(n_estimators=20, random_state=random_seed,
#                           bootstrap=True, max_features='sqrt')
# gb = GradientBoostingClassifier(n_estimators=20, 
#                                 max_features='sqrt', random_state=random_seed)

# # pipeline 
# pipe = Pipeline([
#                  ('standardizer', ss),
#                  ('pca', pca),
#                  ('classifier', rf)
# ])

# # parameters to test
# param_grid = [{
#     'standardizer': [ss, pt, norm],
#     'pca__n_components': [0.95, 0.85, 0.7],
#     'classifier': [rf, et, gb],
# }]

# pipe_grid_start = time.time()

# pipe_search = GridSearchCV(pipe, cv=3, param_grid=param_grid,
#                            scoring = 'f1_micro', refit=True)

# pipe_search.fit(X_train, y_train)
# print(pipe_search.best_params_)
# pipe_preds = pipe_search.predict(X_test)

# pipe_grid_end = time.time()
# print(runtime_in_min(pipe_grid_start, pipe_grid_end))

"""Output:

{'classifier': ExtraTreesClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=None, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=20, n_jobs=None, oob_score=False, random_state=1, verbose=0, warm_start=False), 'pca__n_components': 0.7, 'standardizer': Normalizer(copy=True, norm='l2')}

Runtime: 51.98222145636876
"""

# f1_score(y_test, pipe_preds, average='micro')

"""Score: 0.9489

# Script Total Run Time
"""

# Script timer end point
script_stop = time.time()

# Calculate total run time
script_runtime = script_stop - script_start
print(runtime_in_min(script_start, script_stop))

"""# Identify flaw:

5. The experiment we have proposed has a design flaw. Identify the flaw. Fix it. Rerun the experiment in a way that is consistent with a training-and-test regimen.

I think the flaw is in step 2. PCA was fitted to the entire dataset whereas you would want the test set to be held out until the very, very end. You also want to fit PCA to only the training set and transform the test set with that. The issue is fitting PCA the entire dataset is the wrong approach.

# Variables Appendix

* script_start = timer start for entire script
* script_stop = timer end point for entire script
* script_runtime = total runtime for entire script
* base_rf = base random forest
* base_rf_score = base rf f1 score
* base_rf_runtime = base rf runtime
* full_pca_runtime = runtime in minutes by running PCA on full dataset
* full_pca_n_components = number of components left by running PCA on full dataset
"""