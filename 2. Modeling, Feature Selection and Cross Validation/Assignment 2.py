# -*- coding: utf-8 -*-


# Commented out IPython magic to ensure Python compatibility.
# Jump-Start for the Bank Marketing Study
# as described in Marketing Data Science: Modeling Techniques
# for Predictive Analytics with R and Python (Miller 2015)

# jump-start code revised by Thomas W. Milller (2018/10/07)

# Scikit Learn documentation for this assignment:
# http://scikit-learn.org/stable/auto_examples/classification/
#   plot_classifier_comparison.html
# http://scikit-learn.org/stable/modules/generated/
#   sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB.score
# http://scikit-learn.org/stable/modules/generated/
#   sklearn.linear_model.LogisticRegression.html
# http://scikit-learn.org/stable/modules/model_evaluation.html 
# http://scikit-learn.org/stable/modules/generated/
#  sklearn.model_selection.KFold.html

# prepare for Python version 3x features and functions
# comment out for Python 3.x execution
# from __future__ import division, print_function
# from future_builtins import ascii, filter, hex, map, oct, zip

# seed value for random number generators to obtain reproducible results
RANDOM_SEED = 5

# import base packages into the namespace for this program
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

# initial work with the smaller data set
file_url = ""
bank = pd.read_csv(file_url, sep = ';')  # start with smaller data set
# examine the shape of original input data
print(bank.shape)

# drop observations with missing data, if any
bank.dropna()
# examine the shape of input data after dropping missing data
print(bank.shape)

# look at the list of column names, note that y is the response
list(bank.columns.values)

# look at the beginning of the DataFrame
bank.head()

# mapping function to convert text no/yes to integer 0/1
convert_to_binary = {'no' : 0, 'yes' : 1}

# define binary variable for having credit in default
default = bank['default'].map(convert_to_binary)

# define binary variable for having a mortgage or housing loan
housing = bank['housing'].map(convert_to_binary)

# define binary variable for having a personal loan
loan = bank['loan'].map(convert_to_binary)

# define response variable to use in the model
response = bank['response'].map(convert_to_binary)

# gather three explanatory variables and response into a numpy array 
# here we use .T to obtain the transpose for the structure we want
model_data = np.array([np.array(default), np.array(housing), np.array(loan), 
    np.array(response)]).T

# examine the shape of model_data, which we will use in subsequent modeling
print(model_data.shape)

"""**OUR CODE HERE**"""

X = [row[0:3] for row in model_data]
Y = [row[3] for row in model_data]

# the rest of the program should set up the modeling methods
# and evaluation within a cross-validation design

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=0.25)

"""## Data Exploration"""

#plot the number of clients susbscribed to term deposits, we can see in this case the majority are no.
fig, ax = plt.subplots()
bank['response'].value_counts().plot(kind="bar")
plt.show()
fig.savefig('plot-client-term-dep-y-n.pdf',
        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
        orientation='portrait', papertype=None, format=None, 
        pad_inches=0.25, frameon=None)

bank['response'].value_counts() # remove this or bar chart later

# All Customers
distributions = bank[['age', 'balance', 'duration', 'pdays']].hist( bins = 5, figsize=(15, 5), layout=(1,4))
plt.savefig("plot-distribution-all-customers.pdf",
        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
        orientation='portrait', papertype=None, format=None, 
        pad_inches=0.25, frameon=None)

# Customers who responded yes to response variable
distributions = bank[response == 1][['age', 'balance', 'duration', 'pdays']].hist( bins = 5, figsize=(15, 5), layout=(1,4))
plt.savefig("plot-distribution-all-yes-customers.pdf",
        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
        orientation='portrait', papertype=None, format=None, 
        pad_inches=0.25, frameon=None)

"""Comparing the top vs bottom graphs we can see as these variables start to increase more likely to lead to a yes response lets look at averages to confirm"""

# What does the average 'yes' person look like?
pd.options.display.float_format = '{:,.2f}'.format
bank.groupby('response').mean()

"""Looking at the average response and graphs above of those clients subscribed to term deposits who said yes, we can see:
1. Average age is slightly higher and larger amount of yes in above 60 age groups
2. Average yearly balance is slightly higher but the datset is likely skewed from graphs above
3. Date of month contacted has no bearing on response, but significantly longer time spent on prior call for those interested
4. More likely to see a yes response if not first time contacting client.  However, contacting a client too often may impact likelihood of interest as days since last contacted were significantly higher in those who said yes.
"""

# age versus response
x1 = list(bank[bank['response'] == 'yes']['age'])
x2 = list(bank[bank['response'] == 'no']['age'])
plt.figure(figsize=(15,10))
plt.hist([x1, x2], label=['yes','no'], bins=15, color= ['blue','tan'])
plt.legend()
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Response type by age')
plt.savefig("plot-age-versus-response.pdf",
        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
        orientation='portrait', papertype=None, format=None, 
        pad_inches=0.25, frameon=None)

# Split age group in 5 year increments ratio's based on age
bank['age_group'] = pd.cut(bank.age, bins=[g for g in range(20, 90, 5)], include_lowest=True)
age_y = bank[response == 1]['age_group'].value_counts()
age_n =bank[response == 0]['age_group'].value_counts()
age_y/(age_n+age_y)

# stacked bar of ratio
age_ratios = age_y/(age_n+age_y) 
age_ratios_df = pd.DataFrame(age_ratios) # put the series into a dataframe so I can add another column
age_ratios_df['no'] = 1-  age_ratios_df.age_group # create a column with the complementary values
age_ratios_df

# code for bar chart in seaborn
x = age_ratios_df.index
y1 = age_ratios_df.age_group
y2 = age_ratios_df.no

plt.figure(figsize=(16,5)) 

# no bar first
sns.barplot(x=x, y=y2, color="#EC8013") 
sns.barplot(x=x, y=y1, color="#137FEC") # this draws one chart on top of the other
plt.savefig("plot-no-bar-first.pdf",
        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
        orientation='portrait', papertype=None, format=None, 
        pad_inches=0.25, frameon=None)

# bar chart of age ratios
plt.figure(figsize=(16,5)) # Widen the to be able to see age bins
sns.barplot(x=x , y=y1, color="#137FEC") # set a color because seaborn likes to have different bars be different colors
plt.savefig("plot-bar-chart-age-ratios.pdf",
        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
        orientation='portrait', papertype=None, format=None, 
        pad_inches=0.25, frameon=None)

"""When reviewing the results of customers who have subscribed to a term deposit, we notice at the 20-25 age group and the older age groups above 60 there is a much larger likelihood of customers to subcribe to term deposits.  The amount of clients in these bucket appear smaller from clients where datset could potentially be skewed.  25-30 also returned a slightly higher result than the 30-60 age range with a good frequency of clients in this age group."""

# Education versus response
x1 = list(bank[bank['response'] == 'yes']['education'])
x2 = list(bank[bank['response'] == 'no']['education'])

plt.figure(figsize=(15,10))
plt.hist([x1, x2], label=['yes','no'], bins=5, color= ['blue','tan'])
plt.legend()
plt.xlabel('Education')
plt.ylabel('Frequency')
plt.title('Education and Response Type')
plt.savefig("plot-education-versus-response.pdf",
        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
        orientation='portrait', papertype=None, format=None, 
        pad_inches=0.25, frameon=None)

# ratio of yes responses based on education
education_y = bank[response == 1]['education'].value_counts()
education_n =bank[response == 0]['education'].value_counts()
education_y/(education_n+education_y)

"""Tertiary education potentially leads to a higher slighlty high response rate a few % above the normal but very little difference between secondary primary and unknown education."""

# Job versus response
x1 = list(bank[bank['response'] == 'yes']['job'])
x2 = list(bank[bank['response'] == 'no']['job'])
plt.figure(figsize=(20,10))
plt.hist([x1, x2], label=['yes','no'], bins=15, color= ['blue','tan'])
plt.legend()
plt.xlabel('job')
plt.ylabel('Frequency')
plt.title('Response by Job type')
plt.savefig("plot-job-versus-response.pdf",
        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
        orientation='portrait', papertype=None, format=None, 
        pad_inches=0.25, frameon=None)

# ratio of yes responses based on job
job_y = bank[response == 1]['job'].value_counts()
job_n =bank[response == 0]['job'].value_counts()
job_y/(job_n+job_y)

"""1. Those classified as having a blue-collar job scored significantly below the average and should be avoided for future promotions (largest client list).  
2. Management scored fairly well and had a large amount of clients inside it (twice as likely as blue collar to subscribe).
3. I would reccomend gathering a large sample size first but students, retired and clients with unknown job also scored on the higher end of the spectrum as a consideration
"""

# Marital versus response
x1 = list(bank[bank['response'] == 'yes']['marital'])
x2 = list(bank[bank['response'] == 'no']['marital'])

plt.figure(figsize=(15,10))
plt.hist([x1, x2], label=['yes','no'], bins=3, color= ['blue','tan'])
plt.legend()
plt.xlabel('Marital')
plt.ylabel('Frequency')
plt.title('Response by Marital status')
plt.savefig("plot-marital-versus-response.pdf",
        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
        orientation='portrait', papertype=None, format=None, 
        pad_inches=0.25, frameon=None)

# ratio of yes responses based on maritial status
married_y = bank[response == 1]['marital'].value_counts()
married_n =bank[response == 0]['marital'].value_counts()
married_y/(married_n+married_y)

"""Married clients are less likely to apply to term desposits than single & divorced.  Sample size may want to be increased in the single and divorced buckets to confirm assumptions

**Breakout of predefined binary variables on model**
"""

#% of yes responses
( response == 1 ).sum() / len(response) * 100

# break-down of has credit in default for yes response group 
defaulted = bank[response == 1]['default'].value_counts()
defaulted

# ratio of yes responses based on has credit in default
default_y = bank[response == 1]['default'].value_counts()
default_n = bank[response == 0]['default'].value_counts()
print(default_y/(default_n+default_y))

"""Its specified in the assignment which binary explanatory variables to use to predict the response variable.  Even though we dont have control of what variables we chose, its still beneficial to understand it's impact on the response variable before building the model.

Results above are showing that having credit in default doesnt impact likelihood to subscribe to a term deposit where close to same ratio for those who have credit in default for both those subscribed and not subscribed to a term deposit
"""

# break-down of has housing loan for yes response group 
house = bank[response == 1]['housing'].value_counts()
house

# ratio of yes responses based on has house loan
house_y = bank[response == 1]['housing'].value_counts()
house_n =bank[response == 0]['housing'].value_counts()
house_y/(house_n+house_y)

"""Having a housing loan lowers likelihood to a subscribe to a term deposit by a fair amount"""

# break-down of has personal loan for yes response group 
persloan = bank[response == 1]['loan'].value_counts()
persloan

# ratio of yes responses based on has personal loan
pers_y = bank[response == 1]['loan'].value_counts()
pers_n =bank[response == 0]['loan'].value_counts()
pers_y/(pers_n+pers_y)

"""Having a personal loan lowers likelihood to subscribe to a term deposit by a fair amount"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
tree_classifier = DecisionTreeClassifier(random_state=42)
rf_classifier = RandomForestClassifier(random_state=42)

#all_features = bank.iloc[:,0:-1] # need to convert it to a number to use them in trees
#targets = bank.iloc[:,-1]

tree_classifier.fit(X, Y)
rf_classifier.fit(X,Y)

tree_classifier.feature_importances_

rf_classifier.feature_importances_

"""Both DecisionTreeClassifier and ensemble method, RandomForestClassifier suggest that the second feature, housing is the most important of the three. Both EDA and classifiers suggest housing should be weighted more heavily in modeling.

**Build the model based on default, housing, and loan**

##Logistic Regression Model
"""

#run the logistic regression model with the dataset split 75 % training date and 25 % test data
#3 Binary variables are defined above housing, default and loan to the response variable Y, has client subscribed to 

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=1)

logmodel = LogisticRegression()

logmodel.fit(X_train, Y_train)

prediction = logmodel.predict(X_test)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print(classification_report(Y_test, prediction))

print(confusion_matrix(Y_test, prediction))

print(accuracy_score(Y_test, prediction))

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=0.25)

lr_clf = LogisticRegression(C=100, solver='liblinear') # C = 100
lr_clf.fit(X_train, y_train)

lr_clf.score(X_train, y_train)

lr_clf.score(X_test, y_test)

y_pred = lr_clf.predict_proba(X_test) # taking a look at probabilities
print(y_pred[:10])

y_pred.shape

no_probs = [row[0] for row in y_pred]
no_probs[:10]

yes_probs = [row[1] for row in y_pred]
yes_probs[:10]

threshold_altering = ['yes' if prob >= 0.5 else 'no' for prob in yes_probs]
threshold_altering[:10]

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit

# logistic regression cv results
#cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv = ShuffleSplit(n_splits=5, test_size=0.2, train_size=0.8, random_state=42)

def cv_results(clf, scoring):
  results = cross_val_score(estimator=clf, X=X, y=Y, cv=cv,
                            scoring=scoring)
  return results

cv_results(lr_clf, 'accuracy')

print(cv_results(lr_clf, 'roc_auc'))

print(cv_results(lr_clf, 'roc_auc').mean())

print(cv_results(lr_clf, 'f1'))

"""Since everything is predicted as no, the warning message says precision and recall may be poor metrics for this situation."""

clf = LogisticRegression(C=1, solver='liblinear') # C = 1
clf.fit(X_train, y_train)

clf.score(X_train, y_train)

clf.score(X_test, y_test)

mean_lr_results = cv_results(clf, 'roc_auc').mean() # roc auc for logistic regression, C=1
mean_lr_results

mean_lr_acc_results = cv_results(clf, 'accuracy').mean() # accuracy for logistic regression, C=1
mean_lr_acc_results

"""Conclusion: Tuning the C parameter does not seem to change the results much.

## Bayesian Classifiers

### GaussianNB
"""

from sklearn.naive_bayes import GaussianNB
gbayes_clf = GaussianNB()
gbayes_clf.fit(X_train, y_train)

gbayes_clf.score(X_train, y_train)

gbayes_clf.score(X_test, y_test)

gbayes_prediction = gbayes_clf.predict(X_test)
gbayes_prediction

from collections import Counter
Counter(gbayes_prediction)

confusion_matrix(y_test, gbayes_prediction)

"""983 True positives (response = no, prediction = no)

23 False positives (response = no, prediction = yes)

124 False negatives (response = yes, prediction = no)

1 True negative (response = yes, prediction = yes)

So, model's pretty awful at predicting 'yes.' It only predicted one yes correctly, and the rest are wrong. I think this is good evidence that we can throw this model out all together. This makes sense because Gaussian NB is generally used on normally distributed data. 

Ali - this is a case where you would want to use StandardScaler before fitting data to the classifier. StandardScaler converts everything to z-scores.
"""

cv_results(gbayes_clf, 'accuracy')

mean_gbayes_results = cv_results(gbayes_clf, 'roc_auc').mean()
mean_gbayes_results

"""### BernoulliNB"""

from sklearn.naive_bayes import BernoulliNB 
bnb_clf = BernoulliNB(alpha=1)
bnb_clf.fit(X_train, y_train)

bnb_clf.score(X_train, y_train)

bnb_clf.score(X_test, y_test)

Counter(bnb_clf.predict(X_test)) # Counting how many 1's and 0's are in the predictions

mean_bnb_acc_results = cv_results(bnb_clf, 'accuracy').mean()
mean_bnb_acc_results

mean_bnb_results = cv_results(bnb_clf, 'roc_auc').mean()
mean_bnb_results

bnb_prediction = bnb_clf.predict(X_test)

print(confusion_matrix(Y_test, bnb_prediction)) # Bernoulli NB confusion matrix

# plot ROC curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

lr_scores = clf.predict_proba(X_test)[:, 1]
lr_fpr, lr_tpr, lr_thresholds = roc_curve(y_test, lr_scores)

gbayes_pred = gbayes_clf.predict_proba(X_test)
gbayes_scores = gbayes_pred[:, 1]
gbayes_fpr, gbayes_tpr, gbayes_thresholds = roc_curve(y_test, gbayes_scores)

bnb_pred = bnb_clf.predict_proba(X_test)
bnb_scores = bnb_pred[:, 1]
bnb_fpr, bnb_tpr, bnb_thresholds = roc_curve(y_test, bnb_scores)

plt.figure(figsize=(12,8))
p1 = plt.plot(lr_fpr, lr_tpr)
p2 = plt.plot(gbayes_fpr,gbayes_tpr)
p3 = plt.plot(bnb_fpr, bnb_tpr)
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.title("ROC Curve of 3 classifier")
plt.legend((p1[0], p2[0], p3[0]), ('Logistic Regression', 'GaussianNB', 'BernoulliNB'))
plt.savefig("plot-TPR-FPR-ROC-3-Classifier.pdf",
        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
        orientation='portrait', papertype=None, format=None, 
        pad_inches=0.25, frameon=None)
plt.show()

"""Based on the plotted ROC curve, logistric regression and Gaussian NB classifiers curves almost overlap with one another."""

print("Mean Logistic Regression ROC AUC results: {:.5f}".format(mean_lr_results))
print("Mean Bernoulli Naive Bayes ROC AUC results: {:.5f}".format(mean_bnb_results))

print("Mean Logistic Regression Accuracy results: {:.5f}".format(mean_lr_acc_results))
print("Mean Bernoulli Naive Bayes Accuracy results: {:.5f}".format(mean_bnb_acc_results))
