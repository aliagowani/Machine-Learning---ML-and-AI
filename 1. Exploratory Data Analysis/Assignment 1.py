#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Jump-Start Example: Python analysis of MSPA Software Survey

# Update 2017-09-21 by Tom Miller and Kelsey O'Neill
# Update 2018-06-30 by Tom Miller v005 transformation code added

# tested under Python 3.6.1 :: Anaconda custom (x86_64)
# on Windows 10.0 and Mac OS Sierra 10.12.2 

# shows how to read in data from a comma-delimited text file
# manipuate data, create new count variables, define categorical variables,
# work with dictionaries and lambda mapping functions for recoding data 

# visualizations in this program are routed to external pdf files
# so they may be included in printed or electronic reports

# prepare for Python version 3x features and functions
# these two lines of code are needed for Python 2.7 only
# commented out for Python 3.x versions
# from __future__ import division, print_function
# from future_builtins import ascii, filter, hex, map, oct, zip

# external libraries for visualizations and data manipulation
# ensure that these packages have been installed prior to calls
import pandas as pd  # data frame operations  
import numpy as np  # arrays and math functions
import matplotlib.pyplot as plt  # static plotting
import seaborn as sns  # pretty plotting, including heat map
import os
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# correlation heat map setup for seaborn
def corr_chart(df_corr):
    corr=df_corr.corr()
    #screen top half to get a triangle
    top = np.zeros_like(corr, dtype=np.bool)
    top[np.triu_indices_from(top)] = True
    fig=plt.figure()
    fig, ax = plt.subplots(figsize=(12,12))
    sns.heatmap(corr, mask=top, cmap='coolwarm', 
        center = 0, square=True, 
        linewidths=.5, cbar_kws={'shrink':.5}, 
        annot = True, annot_kws={'size': 9}, fmt = '.3f')           
    plt.xticks(rotation=45) # rotate variable labels on columns (x axis)
    plt.yticks(rotation=0) # use horizontal variable labels on rows (y axis)
    plt.title('Correlation Heat Map')   
    plt.savefig('plot-corr-map.pdf', 
        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
        orientation='portrait', papertype=None, format=None, 
        transparent=True, pad_inches=0.25, frameon=None)      

np.set_printoptions(precision=3)


# In[3]:


# read in comma-delimited text file, creating a pandas DataFrame object
# note that IPAddress is formatted as an actual IP address
# but is actually a random-hash of the original IP address


###############################################################################
### Define path to where file is stored.  Note for windows users, maintain the 
### r in front of the path.  For mac users, the r is not required.
###############################################################################
#path=""
#os.chdir(path)

# from github repo, raw file
#file_path = ""

# from github gist
file_path = ""

valid_survey_input = pd.read_csv(file_path)

# use the RespondentID as label for the rows... the index of DataFrame
valid_survey_input.set_index('RespondentID', drop = True, inplace = True)


# In[4]:


# examine the structure of the DataFrame object
print('\nContents of initial survey data ---------------')


# In[5]:


# could use len() or first index of shape() to get number of rows/observations
print('\nNumber of Respondents =', len(valid_survey_input)) 


# In[6]:


# show the column/variable names of the DataFrame
# note that RespondentID is no longer present
print(valid_survey_input.columns)


# In[7]:


# abbreviated printing of the first five rows of the data frame
print(pd.DataFrame.head(valid_survey_input)) 


# In[8]:


# shorten the variable/column names for software preference variables
survey_df = valid_survey_input.rename(index=str, columns={
    'Personal_JavaScalaSpark': 'My_Java',
    'Personal_JavaScriptHTMLCSS': 'My_JS',
    'Personal_Python': 'My_Python',
    'Personal_R': 'My_R',
    'Personal_SAS': 'My_SAS',
    'Professional_JavaScalaSpark': 'Prof_Java',
    'Professional_JavaScriptHTMLCSS': 'Prof_JS',
    'Professional_Python': 'Prof_Python',
    'Professional_R': 'Prof_R',
    'Professional_SAS': 'Prof_SAS',
    'Industry_JavaScalaSpark': 'Ind_Java',
    'Industry_JavaScriptHTMLCSS': 'Ind_JS',
    'Industry_Python': 'Ind_Python',
    'Industry_R': 'Ind_R',
    'Industry_SAS': 'Ind_SAS'})


# In[9]:


# define subset DataFrame for analysis of software preferences 
software_df = survey_df.loc[:, 'My_Java':'Ind_SAS']


# In[10]:


# single scatter plot example
fig, axis = plt.subplots()
axis.set_xlabel('Personal Preference for R')
axis.set_ylabel('Personal Preference for Python')
plt.title('R and Python Perferences')
scatter_plot = axis.scatter(survey_df['My_R'], 
    survey_df['My_Python'],
    facecolors = 'none', 
    edgecolors = 'blue') 
plt.savefig('plot-scatter-r-python.pdf', 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='portrait', papertype=None, format=None, 
    transparent=True, pad_inches=0.25, frameon=None)  

survey_df_labels = [
    'Personal Preference for Java/Scala/Spark',
    'Personal Preference for Java/Script/HTML/CSS',
    'Personal Preference for Python',
    'Personal Preference for R',
    'Personal Preference for SAS',
    'Professional Java/Scala/Spark',
    'Professional JavaScript/HTML/CSS',
    'Professional Python',
    'Professional R',
    'Professional SAS',
    'Industry Java/Scala/Spark',
    'Industry Java/Script/HTML/CSS',
    'Industry Python',
    'Industry R',
    'Industry SAS'        
]  


# In[11]:


survey_df = valid_survey_input.rename(index=str, columns={
    'Personal_JavaScalaSpark': 'My_Java',
    'Personal_JavaScriptHTMLCSS': 'My_JS',
    'Personal_Python': 'My_Python',
    'Personal_R': 'My_R',
    'Personal_SAS': 'My_SAS',
    'Professional_JavaScalaSpark': 'Prof_Java',
    'Professional_JavaScriptHTMLCSS': 'Prof_JS',
    'Professional_Python': 'Prof_Python',
    'Professional_R': 'Prof_R',
    'Professional_SAS': 'Prof_SAS',
    'Industry_JavaScalaSpark': 'Ind_Java',
    'Industry_JavaScriptHTMLCSS': 'Ind_JS',
    'Industry_Python': 'Ind_Python',
    'Industry_R': 'Ind_R',
    'Industry_SAS': 'Ind_SAS'})

# create a set of scatter plots for personal preferences
for i in range(5):
  for j in range(5):
    if i != j:
      file_title = survey_df.columns[i] + '_and_' + survey_df.columns[j]
      plot_title = survey_df.columns[i] + ' and ' + survey_df.columns[j]
      fig, axis = plt.subplots()
      axis.set_xlabel(survey_df_labels[i])
      axis.set_ylabel(survey_df_labels[j])
      plt.title(plot_title)
      scatter_plot = axis.scatter(survey_df[survey_df.columns[i]], 
      survey_df[survey_df.columns[j]],
      facecolors = 'none', 
      edgecolors = 'blue') 
      plt.savefig(file_title + '.pdf',             
                  bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
                  orientation='portrait', papertype=None, format=None, 
                  transparent=True, pad_inches=0.25, frameon=None)  


# In[12]:


# examine intercorrelations among software preference variables
# with correlation matrix/heat map
corr_chart(df_corr = software_df) 


# In[13]:


# descriptive statistics for software preference variables
print('\nDescriptive statistics for survey data ---------------')
print(software_df.describe())


# In[14]:


# descriptive statistics for one variable
print('\nDescriptive statistics for courses completed ---------------')
print(survey_df['Courses_Completed'].describe())


# In[15]:


# ----------------------------------------------------------
# transformation code added with version v005
# ----------------------------------------------------------
# transformations a la Scikit Learn
# documentation at http://scikit-learn.org/stable/auto_examples/
#                  preprocessing/plot_all_scaling.html#sphx-glr-auto-
#                  examples-preprocessing-plot-all-scaling-py
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


# In[16]:


# transformations a la Scikit Learn
# select variable to examine, eliminating missing data codes
X = survey_df['Courses_Completed'].dropna()
print(X.shape)
# Seaborn provides a convenient way to show the effects of transformations
# on the distribution of values being transformed
# Documentation at https://seaborn.pydata.org/generated/seaborn.distplot.html


# In[17]:


unscaled_fig, ax = plt.subplots()
sns.distplot(X).set_title('Unscaled')
unscaled_fig.savefig('Transformation-Unscaled' + '.pdf', 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='portrait', papertype=None, format=None, 
    transparent=True, pad_inches=0.25, frameon=None)  


# In[18]:


log_fig, ax = plt.subplots()
sns.distplot(np.log(X)).set_title('NaturalLog')
log_fig.savefig('Transformation-NaturalLog' + '.pdf', 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='portrait', papertype=None, format=None, 
    transparent=True, pad_inches=0.25, frameon=None)  


# In[19]:


X=np.array(X).reshape(-1,1)
standard_fig, ax = plt.subplots()
sns.distplot(StandardScaler().fit_transform(X)).set_title('StandardScaler')
standard_fig.savefig('Transformation-StandardScaler' + '.pdf', 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='portrait', papertype=None, format=None, 
    transparent=True, pad_inches=0.25, frameon=None)  


# 

# In[20]:


minmax_fig, ax = plt.subplots()
sns.distplot(MinMaxScaler().fit_transform(X)).set_title('MinMaxScaler')
#sns.distplot(MinMaxScaler().fit_transform(X)).set_title('MinMaxScaler')
minmax_fig.savefig('Transformation-MinMaxScaler' + '.pdf', 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='portrait', papertype=None, format=None, 
    transparent=True, pad_inches=0.25, frameon=None) 


# # Our code starts here

# In[21]:


#Classes offered per program language, does it align with students preferences?
language_counts = {"R": 5, "Python": 3, "SAS": 2} # Dictionary of counts per language


# In[22]:


valid_survey_input.columns


# **Programming language analysis**

# In[23]:


#Boxplot of the points across the five software options based on students' desire to learn each language or software system.

#Create dataframe for each category (Student Desire, Profession and Industry)
my_df = survey_df[['My_Java', 'My_JS', 'My_Python', 'My_R', 'My_SAS']]
prof_df = survey_df[['Prof_Java', 'Prof_JS', 'Prof_Python', 'Prof_R', 'Prof_SAS']]
ind_df = survey_df[['Ind_Java', 'Ind_JS', 'Ind_Python', 'Ind_R', 'Ind_SAS']]

# Not needed but shows the total value for each category
my_df.sum()
prof_df.sum()
ind_df.sum()

#Boxplots Side by Side (1 x 3)
plt.rcParams['figure.figsize']=(15,5)
fig, ax = plt.subplots(1,3)
sns.boxplot(y=None, data=my_df, ax=ax[0]).set_title('Students Desire to Learn')
sns.boxplot(y=None, data=ind_df, ax=ax[1]).set_title('Students Professional Need')
sns.boxplot(y=None, data=prof_df, ax=ax[2]).set_title('Importance in Industry')

# Save chart
fig.savefig('Assign1_Boxplot_Student_Des_Prof_Ind' + '.pdf', 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='portrait', papertype=None, format=None, 
    transparent=True, pad_inches=0.25)


# When we generate boxplots based on the students' desire to learn, students profesional needs and the students' perspective on these languages in the industry, we can gain few insights: 
# 
# 1.   Python and R seem to gain most points from students who want to learn them.
# 2.   Students feel that while Python and R meet their profesional needs, several respondants feel the same way for SAS, if not more so. There are several outliers that are increasing the mean for students profesional need for SAS. This would mean that these students see SAS in great professional need and potentially future demand but it is not reflective of the total population and just few students (outliers).
# 3.   Students feel that the industry is demanding Python and R as programmatic langauges, perhaps foretelling future demand.
# 4.   While R seems to lead in both the students desire to learn this langague and their perspective of the dmeand for R in the industry, it is interesting to note that their current profesional need does not require additional students to learn R as much as they desire or what they feel the market demands. This may suggest that there is sufficient demand in the market place for R related work. 
# 5. JavaScript scores the lowest in every category. So perhaps even proficient JS users do not find it useful to expand on the tool.
# 
# 
# 

# In[24]:


# Shows a box plot of the total points allocated by student for each programming language by category (Student Desire, Profession, and Industry)
plt.rcParams['figure.figsize']=(15,5)
fig, ax = plt.subplots(1,1)

# Joins the seperate dataframes into a new dataframe
new_order = pd.concat([my_df, prof_df, ind_df])

# Orders the columns in the dataframe by programming language and ensures the color scheme matches the boxplot above with the programming language.
new_order = new_order[['My_Java', 'Prof_Java', 'Ind_Java', 'My_JS', 'Prof_JS', 'Ind_JS', 'My_Python', 'Prof_Python', 'Ind_Python', 'My_R', 'Prof_R', 'Ind_R', 'My_SAS', 'Prof_SAS', 'Ind_SAS']]
new_order_type_colors = ['#3274A1', '#3274A1', '#3274A1', '#E1812C', '#E1812C', '#E1812C', '#3A923A', '#3A923A', '#3A923A', '#9B3132', '#9B3132', '#9B3132', '#9372B2', '#9372B2', '#9372B2']

# Displays the barplot
sns.barplot(data=new_order, estimator=sum, palette=new_order_type_colors, ci=None).set_title('Total Points by Prog. Language')
plt.xlabel("Programming Language by My(Student) Preference, Prof. and Industry")
plt.ylabel("Total Points")

# Save chart
fig.savefig('Assign1_Barplot_Student_Des_Prof_Ind' + '.pdf', 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='portrait', papertype=None, format=None, 
    transparent=True, pad_inches=0.25) 


# **Analysis on new course interest**

# Perception: Personal perference is what students want to learn. Professional perception is what they need to improve. Industry is what they think is most important where they are going.

# In[25]:


course_interest = valid_survey_input.iloc[:, 15:19].copy()
course_interest.head()


# These courses minus Python course do not have a set language yet.

# In[26]:


plt.rcParams['figure.figsize']=(12,6)
fig, ax = plt.subplots(1,1)
p1 = sns.boxplot(y=None, data=course_interest)
p1.set_title("Boxplots of Course Interests")

# Save chart
fig.savefig('Assign1_Boxplot_Course_Interests' + '.pdf', 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='portrait', papertype=None, format=None, 
    transparent=True, pad_inches=0.25) 


# In[27]:


mean_course_interest = course_interest.mean()
median_course_interest = course_interest.median()


# In[28]:


columns = mean_course_interest.index
mean_values = mean_course_interest.values

fig, ax = plt.subplots(1,1)
p2 = sns.barplot(x = columns, y = mean_values)
p2.set_title("Mean Values of Course Interest")

# Save chart
fig.savefig('Assign1_Barplot_Course_Interests_MeanValues' + '.pdf', 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='portrait', papertype=None, format=None, 
    transparent=True, pad_inches=0.25) 


# So it's obvious that an introductory Python course has strong interest. Combined with the speculation that Python is necessary in the industry and professionally, this could mean that incoming students at the time feel a need to learn Python to continue a career in analytics. 
# 
# This could all point to the fact that the following courses if taught in Python could require some type of Python prerequisite or have Python weaved into the classes, progressively getting more advanced as students go through the program.

# **Does progress in program change student preferences?**

# In[29]:


# survey was conducted Dec '16
# Split respondents graduating within a year versus ones graduating later and compare preferences there.
valid_survey_input.Graduate_Date.value_counts()


# In[30]:


# Perhaps number of courses completed is a better measurement of progress through program
# Pandas cut function
valid_survey_input.Courses_Completed.dropna().value_counts().sort_index()


# In[31]:


#split courses completed 0-4 into Early in program, 5-8 middle of program, 9-12 late or close to graduating
progress_df = valid_survey_input.copy()
progress_df['progress'] = pd.cut(x=valid_survey_input['Courses_Completed'].dropna(), bins=[0,4,8,12], labels=['Early', 'Middle', 'Late'])
progress_df.progress.value_counts()


# In[32]:


#sequential ordering
progress_counts = progress_df.progress.value_counts().reindex(['Early', 'Middle', 'Late'])
progress_counts


# In[33]:


# bar plot for value counts
plt.rcParams['figure.figsize']=(10,4)
fig, ax = plt.subplots(1,1)

progress_counts_bar = sns.barplot(x=progress_counts.index, y=progress_counts)
progress_counts_bar.set_title("Bar plot of surveyed progress counts")

# Save chart
fig.savefig('Assign1_Boxplot_Student_Progress_Counts' + '.pdf', 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='portrait', papertype=None, format=None, 
    transparent=True, pad_inches=0.25) 


# In[34]:


# scatter plot of R vs Python, colored by stage
plt.rcParams['figure.figsize']=(12,9)
fig, ax = plt.subplots(1,1)

pref_scatter = sns.scatterplot(x = progress_df['Personal_Python'], 
                              y = progress_df['Personal_R'],
                              hue = progress_df['progress'])

# Save chart
fig.savefig('Assign1_Scatter_Student_Progress' + '.pdf', 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='portrait', papertype=None, format=None, 
    transparent=True, pad_inches=0.25) 


# In[35]:


fig, ax = plt.subplots(1,1)

python_course_interest = sns.barplot(x=progress_df['progress'], y=progress_df['Python_Course_Interest'])
python_course_interest.set_title("Python Course Interest by Program Progress Levels")

# Save chart
fig.savefig('Assign1_Barplot_Python_Interest_by_Progress' + '.pdf', 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='portrait', papertype=None, format=None, 
    transparent=True, pad_inches=0.25) 


# Most students personally split software preferences somewhere between R and Python. There is nothing that definitive in this chart.
# 
# I chose personal preference over industry and professional because I figured personal preference ties into both industry and professional preference but is also students' predicting what skills they need most later on.

# In[36]:


#correlating course completion by course interest
corr_matrix=survey_df.corr()
corr_matrix["Courses_Completed"].sort_values(ascending=False)


# Overall, when correlating courses completed and preference for software language, we find that course completion is not strong indicator for software preference. 
# 

# In[37]:


#scatter plot 1
from pandas.plotting import scatter_matrix

attributes = ["Courses_Completed","Graduate_Date", "Python_Course_Interest", "Foundations_DE_Course_Interest",
       "Analytics_App_Course_Interest", "Systems_Analysis_Course_Interest"]
Assign1_ScatMatrix_Course_Interest = scatter_matrix(survey_df[attributes], figsize=(18, 15)) # Ben: making this bigger to be able to see it

# Save chart
plt.savefig('Assign1_ScatMatrix_Course_Interest')


# In[38]:


#scatter plot 2
from pandas.plotting import scatter_matrix

attributes = ["Courses_Completed", "Prof_SAS", "Prof_Python", "Prof_R"]
Assign1_ScatMatrix_Course_Course_Completed = scatter_matrix(survey_df[attributes], figsize=(18, 16))

# Save chart
plt.savefig('Assign1_ScatMatrix_Course_Course_Completed')


# The above scatter plots indicate that overall, course completion is not a strong indicator of professional interest for some of the s/w languages (mainly R, Python and SAS), however, across those 3 langauages, R and Python have the stronger co-relations.
# 
# 

# In[39]:


#electives what is the prevalent software?
### DO we see favorability to Python, R, SAS?


#Create dataframe for each category (Student Desire, Profession and Industry)
python_df = survey_df[['PREDICT452','PREDICT453','OtherPython']]
r_df = survey_df[['PREDICT413', 'PREDICT450', 'PREDICT451', 'PREDICT454', 'PREDICT455','PREDICT456', 'PREDICT457', 'OtherR']]
sas_df = survey_df['OtherSAS']

python_val = python_df.count()
r_val = r_df.count()
sas_val = sas_df.count()

print('The count of R electives taken is:', r_val.sum())
print('The count of Python electives taken is:', python_val.sum())
print('The count of SAS electives taken is:', sas_val.sum())


# Interesting to see most students take R courses as electives over python, but could just be due to the number of courses offered in R is much larger.  Very few students taking SAS as an elective
