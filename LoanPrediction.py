# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 16:30:23 2019

@author: Vana
"""


# Import libraries

# linear algebra
import numpy as np 

# data processing
import pandas as pd 
pd.set_option('display.max_columns',13)

# data visualization
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style


# Load training and test data
import os
path="Documents\online teaching\Projects from Internet\Loan Prediction\Python"
os.chdir(path)



# Making a list of missing value types
missing_values = ["n/a", "na", "--", "", " "]

train_set = pd.read_csv("train_u6lujuX_CVtuZ9i.csv", na_values = missing_values)
test_set = pd.read_csv("test_Y3wMUE5_7gLdaTN.csv", na_values = missing_values)

# Preview the data
train_set.head()


# Find how many instances (rows) and how many attributes (columns) the data contains
# shape
print(train_set.shape)
# (614, 13)
print(test_set.shape)
# (367, 12)


# The training set has 614 observations and 13 variables and the test set has 
# 367 observations and 12 variables, which means that the traning set has 1 extra variable. 
# Check which variable is missing from the test set. 

colnames_check = np.setdiff1d(train_set.columns.values,test_set.columns.values)
# array(['Loan_Status'], dtype=object)

# As we can see we are missing the "Loan_Status" variable in the test set, 
# which is something that was expected, since we must predict this by creating a model


# more info on the data
print(train_set.info())

#<class 'pandas.core.frame.DataFrame'>
#RangeIndex: 614 entries, 0 to 613
#Data columns (total 13 columns):
#Loan_ID              614 non-null object
#Gender               601 non-null object
#Married              611 non-null object
#Dependents           599 non-null object
#Education            614 non-null object
#Self_Employed        582 non-null object
#ApplicantIncome      614 non-null int64
#CoapplicantIncome    614 non-null float64
#LoanAmount           592 non-null float64
#Loan_Amount_Term     600 non-null float64
#Credit_History       564 non-null float64
#Property_Area        614 non-null object
#Loan_Status          614 non-null object
#dtypes: float64(4), int64(1), object(8)
#memory usage: 62.4+ KB
#None





# Statistical Summary
# We can take a look at a summary of each attribute.
# This includes the count, mean, the min and max values and some percentiles.

print(train_set.describe())

#       ApplicantIncome  CoapplicantIncome  LoanAmount  Loan_Amount_Term  \
#count       614.000000         614.000000  592.000000         600.00000   
#mean       5403.459283        1621.245798  146.412162         342.00000   
#std        6109.041673        2926.248369   85.587325          65.12041   
#min         150.000000           0.000000    9.000000          12.00000   
#25%        2877.500000           0.000000  100.000000         360.00000   
#50%        3812.500000        1188.500000  128.000000         360.00000   
#75%        5795.000000        2297.250000  168.000000         360.00000   
#max       81000.000000       41667.000000  700.000000         480.00000   
#
#       Credit_History  
#count      564.000000  
#mean         0.842199  
#std          0.364878  
#min          0.000000  
#25%          1.000000  
#50%          1.000000  
#75%          1.000000  
#max          1.000000 



# For categorical variables
print(train_set.describe(include='O'))

#         Loan_ID Gender Married Dependents Education Self_Employed  \
#count        614    601     611        599       614           582   
#unique       614      2       2          4         2             2   
#top     LP002794   Male     Yes          0  Graduate            No   
#freq           1    489     398        345       480           500   
#
#       Property_Area Loan_Status  
#count            614         614  
#unique             3           2  
#top        Semiurban           Y  
#freq             233         422  


##      Exploratory Analysis


# Class Distribution: Examine the number and percentage of customers whose loan was approved

print(train_set.groupby('Loan_Status').size())
#Loan_Status
#N    192
#Y    422
#dtype: int64


# Class Distribution: Percentage
print(train_set.groupby('Loan_Status').size().apply(lambda x: float(x) / train_set.groupby('Loan_Status').size().sum()*100))
#Loan_Status
#N    31.270358
#Y    68.729642
#dtype: float64


# Class Distribution: Frequency and Percentage
print(pd.DataFrame(data = {'freq': train_set.groupby('Loan_Status').size(), 
                           'percentage':train_set.groupby('Loan_Status').size().apply(lambda x: float(x) / train_set.groupby('Loan_Status').size().sum()*100)}))
#             freq  percentage
#Loan_Status                  
#N             192   31.270358
#Y             422   68.729642


#### Visual univariate analysis

# Separate the variables into two lists: "cat" for the categorical variables 
# and "cont" for the continuous variables. 
 
# Loan_Amount_Term and Credit_History have only a few unique values and can be considered categorical
cat =  ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 
        'Loan_Amount_Term', 'Credit_History']
cont = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']

# Distribution for training set only
fig = plt.figure(figsize=(20, 20))
plt.subplots_adjust(hspace = .45)
for i in range (0,len(cat)):
    fig.add_subplot(5,3,i+1)
    sns.countplot(x=cat[i], data=train_set);  

for col in cont:
    fig.add_subplot(5,3,i + 2)
    sns.distplot(train_set[col].dropna());
    i += 1
    fig.add_subplot(5,3,i + 2)
    sns.boxplot(train_set[col].dropna(), orient = 'v')
    i += 1
    
plt.show()
fig.clear()


# Distribution for training and test sets for categorical variables
fig = plt.figure(figsize=(20, 20))
plt.subplots_adjust(hspace = 1.55)
i = 1
j= 2
for col in cat:
    fig.add_subplot(8,2,i)
    sns.countplot(x=train_set[col].dropna());  
    plt.title('Training Set')
    i += 2
    fig.add_subplot(8,2,j)
    sns.countplot(x=test_set[col].dropna());  
    plt.title('Test Set')
    j += 2

plt.show()
fig.clear()

# Distribution for training and test sets for continuous variables
fig = plt.figure(figsize=(15, 15))
plt.subplots_adjust(hspace = .95)
i = 1
j= 2    
for col in cont:
    fig.add_subplot(6,2,i)
    sns.distplot(train_set[col].dropna());
    plt.title('Training Set')
    i += 2
    fig.add_subplot(6,2,j)
    sns.distplot(test_set[col].dropna());
    plt.title('Test Set')
    j += 2
    fig.add_subplot(6,2,i)
    sns.boxplot(train_set[col].dropna(), orient = 'v')
    plt.title('Training Set')
    i += 2
    fig.add_subplot(6,2,j)
    sns.boxplot(test_set[col].dropna(), orient = 'v')
    plt.title('Test Set')
    j += 2
    
plt.show()
fig.clear()

# From all these plots we observe that:

#   Most applicants are males in both the training and test sets.
#print(pd.DataFrame(data = {'Freq-Train': train_set.groupby('Gender').size(), 
#                           'Percentage-Train':train_set.groupby('Gender').size().apply(lambda x: float(x) / train_set.groupby('Gender').size().sum()*100),
#                           'Freq-Test': test_set.groupby('Gender').size(), 
#                           'Percentage-Test':test_set.groupby('Gender').size().apply(lambda x: float(x) / test_set.groupby('Gender').size().sum()*100)
#                           }))
#
#        Freq-Train  Percentage-Train  Freq-Test  Percentage-Test
#Gender                                                          
#Female         112         18.635607         70        19.662921
#Male           489         81.364393        286        80.337079   
 
#   Most applicants are married in both the training and test sets.
#print(pd.DataFrame(data = {'Freq-Train': train_set.groupby('Married').size(), 
#                           'Percentage-Train':train_set.groupby('Married').size().apply(lambda x: float(x) / train_set.groupby('Married').size().sum()*100),
#                           'Freq-Test': test_set.groupby('Married').size(), 
#                           'Percentage-Test':test_set.groupby('Married').size().apply(lambda x: float(x) / test_set.groupby('Married').size().sum()*100)
#                           }))
#         Freq-Train  Percentage-Train  Freq-Test  Percentage-Test
#Married                                                          
#No              213         34.860884        134        36.512262
#Yes             398         65.139116        233        63.487738

#   Most applicants do not have dependents in both the training and test sets.
#print(pd.DataFrame(data = {'Freq-Train': train_set.groupby('Dependents').size(), 
#                           'Percentage-Train':train_set.groupby('Dependents').size().apply(lambda x: float(x) / train_set.groupby('Dependents').size().sum()*100),
#                           'Freq-Test': test_set.groupby('Dependents').size(), 
#                           'Percentage-Test':test_set.groupby('Dependents').size().apply(lambda x: float(x) / test_set.groupby('Dependents').size().sum()*100)
#                           }))
#             Freq-Train  Percentage-Train  Freq-Test  Percentage-Test
#Dependents                                                          
#0                  345         57.595993        200        56.022409
#1                  102         17.028381         58        16.246499
#2                  101         16.861436         59        16.526611
#3+                  51          8.514190         40        11.204482   

#   Most applicants are graduates in both the training and test sets.
#print(pd.DataFrame(data = {'Freq-Train': train_set.groupby('Education').size(), 
#                           'Percentage-Train':train_set.groupby('Education').size().apply(lambda x: float(x) / train_set.groupby('Education').size().sum()*100),
#                           'Freq-Test': test_set.groupby('Education').size(), 
#                           'Percentage-Test':test_set.groupby('Education').size().apply(lambda x: float(x) / test_set.groupby('Education').size().sum()*100)
#                           }))
#              Freq-Train  Percentage-Train  Freq-Test  Percentage-Test
#Education                                                             
#Graduate             480         78.175896        283        77.111717
#Not Graduate         134         21.824104         84        22.888283

#   The vast majority of applicants are not self employed in both the training and test sets.
#print(pd.DataFrame(data = {'Freq-Train': train_set.groupby('Self_Employed').size(), 
#                           'Percentage-Train':train_set.groupby('Self_Employed').size().apply(lambda x: float(x) / train_set.groupby('Self_Employed').size().sum()*100),
#                           'Freq-Test': test_set.groupby('Self_Employed').size(), 
#                           'Percentage-Test':test_set.groupby('Self_Employed').size().apply(lambda x: float(x) / test_set.groupby('Self_Employed').size().sum()*100)
#                           }))
#               Freq-Train  Percentage-Train  Freq-Test  Percentage-Test
#Self_Employed                                                          
#No                    500         85.910653        307        89.244186
#Yes                    82         14.089347         37        10.755814

#   Most applicants have a credit history that meets guidelines in both the training and test sets.
#print(pd.DataFrame(data = {'Freq-Train': train_set.groupby('Credit_History').size(), 
#                           'Percentage-Train':train_set.groupby('Credit_History').size().apply(lambda x: float(x) / train_set.groupby('Credit_History').size().sum()*100),
#                           'Freq-Test': test_set.groupby('Credit_History').size(), 
#                           'Percentage-Test':test_set.groupby('Credit_History').size().apply(lambda x: float(x) / test_set.groupby('Credit_History').size().sum()*100)
#                           }))
#                    Freq-Train  Percentage-Train  Freq-Test  Percentage-Test
#Credit_History                                                          
#0.0                     89         15.780142         59        17.455621
#1.0                    475         84.219858        279        82.544379

#   Property area is the only predictor variable whose distribution looks different between the 
# training and test sets.
#print(pd.DataFrame(data = {'Freq-Train': train_set.groupby('Property_Area').size(), 
#                           'Percentage-Train':train_set.groupby('Property_Area').size().apply(lambda x: float(x) / train_set.groupby('Property_Area').size().sum()*100),
#                           'Freq-Test': test_set.groupby('Property_Area').size(), 
#                           'Percentage-Test':test_set.groupby('Property_Area').size().apply(lambda x: float(x) / test_set.groupby('Property_Area').size().sum()*100)
#                           }))
#               Freq-Train  Percentage-Train  Freq-Test  Percentage-Test
#Property_Area                                                          
#Rural                 179         29.153094        111        30.245232
#Semiurban             233         37.947883        116        31.607629
#Urban                 202         32.899023        140        38.147139


#   The applicant income, co-applicant income and the loan amount have many outliers and 
# the distributions are right asymmetric.

#   The majority of loans have a term of 360 months.
#print(pd.DataFrame(data = {'Freq': train_set.groupby('Loan_Amount_Term').size(), 
#                           'Percentage':train_set.groupby('Loan_Amount_Term').size().apply(lambda x: float(x) / train_set.groupby('Loan_Amount_Term').size().sum()*100)
#                           }))
#                  Freq  Percentage
#Loan_Amount_Term                  
#12.0                 1    0.166667
#36.0                 2    0.333333
#60.0                 2    0.333333
#84.0                 4    0.666667
#120.0                3    0.500000
#180.0               44    7.333333
#240.0                4    0.666667
#300.0               13    2.166667
#360.0              512   85.333333
#480.0               15    2.500000
    
#### Visual bivariate analysis

# The next charts show the loan approval (and non-approval) numbers for each variable.

# Count of loan approval

fig = plt.figure(figsize=(20, 20))
plt.subplots_adjust(hspace = .45)
i = 1
for col in cat:
    if col != 'Loan_Status':
        fig.add_subplot(5,3,i)
        sns.countplot(x=col, data=train_set,hue='Loan_Status')
        i += 1

# Box plot Loan_Status vs ApplicantIncome
fig.add_subplot(5,3,9)
sns.swarmplot(x="Loan_Status", y="ApplicantIncome", hue="Gender", data=train_set)
fig.add_subplot(5,3,10)
sns.boxplot(x="Loan_Status", y="ApplicantIncome", data=train_set)

# Loan_Status vs CoapplicantIncome
fig.add_subplot(5,3,11)
sns.boxplot(x="Loan_Status", y="CoapplicantIncome", data=train_set)

# Loan_Status vs LoanAmount
fig.add_subplot(5,3,12)
sns.boxplot(x="Loan_Status", y="LoanAmount", data=train_set)


# For the barplots and the correlations, we need to convert 'Y', 'N' of Loan_Status to 1,0, respectively
train_set2 = train_set.copy()
train_set2.Loan_Status = pd.Series(np.where(train_set2.Loan_Status.values=='Y',1,0), train_set2.index)


# Correlations
corr = train_set2.drop(['Loan_ID'], axis=1).corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)
fig.add_subplot(5,3,13)
sns.heatmap(corr, mask=mask, cmap=cmap, cbar_kws={"shrink": .5})
plt.show()
fig.clear()



# Percentages of loan approval

fig = plt.figure(figsize=(20, 20))
plt.subplots_adjust(hspace = .45)
i = 1
for col in cat:
    if col != 'Loan_Status':
        fig.add_subplot(4,3,i)
        sns.barplot(x=col, data=train_set2,y='Loan_Status')
        i += 1

# Loan_Status vs ApplicantIncome
fig.add_subplot(4,3,9)
sns.regplot(x='ApplicantIncome', y='Loan_Status', data=train_set2)

# Loan_Status vs CoapplicantIncome
fig.add_subplot(4,3,10)
sns.regplot(x='CoapplicantIncome', y='Loan_Status', data=train_set2)

# Loan_Status vs LoanAmount
fig.add_subplot(4,3,11)
sns.regplot(x='LoanAmount', y='Loan_Status', data=train_set2)


plt.show()
fig.clear()

# ApplicantIncome-Gender-Loan_Status plot
sns.lmplot(x='ApplicantIncome', y='Loan_Status',hue='Gender', data=train_set2, palette='Set1')

# From all these plots we observe that:

#   A slightly larger proportion of female applicants are refused than male ones.
#print(pd.DataFrame(data = {'Freq': train_set.groupby(['Gender'])['Loan_Status'].value_counts(),
#'Percentage': train_set.groupby(['Gender'])['Loan_Status'].value_counts(normalize=True)}))
#                    Freq  Percentage
#Gender Loan_Status                  
#Female Y              75    0.669643
#       N              37    0.330357
#Male   Y             339    0.693252
#       N             150    0.306748

#   A larger proportion of unmarried applicants are refused than married ones.
#                     Freq  Percentage
#Married Loan_Status                  
#No      Y             134    0.629108
#        N              79    0.370892
#Yes     Y             285    0.716080
#        N             113    0.283920

#   A smaller proportion of applicants with 2 dependents are refused than applicants with other 
# number of dependents.
#print(pd.DataFrame(data = {'Freq': train_set.groupby(['Dependents'])['Loan_Status'].value_counts(),
#'Percentage': train_set.groupby(['Dependents'])['Loan_Status'].value_counts(normalize=True)}))
#                        Freq  Percentage
#Dependents Loan_Status                  
#0          Y             238    0.689855
#           N             107    0.310145
#1          Y              66    0.647059
#           N              36    0.352941
#2          Y              76    0.752475
#           N              25    0.247525
#3+         Y              33    0.647059
#           N              18    0.352941    

#   A larger proportion of non graduates are refused than graduates.
#print(pd.DataFrame(data = {'Freq': train_set.groupby(['Education'])['Loan_Status'].value_counts(),
#'Percentage': train_set.groupby(['Education'])['Loan_Status'].value_counts(normalize=True)}))
#                          Freq  Percentage
#Education    Loan_Status                  
#Graduate     Y             340    0.708333
#             N             140    0.291667
#Not Graduate Y              82    0.611940
#             N              52    0.388060
             
#   Applicants that are self employed have a slightly worse approved rate.
#print(pd.DataFrame(data = {'Freq': train_set.groupby(['Self_Employed'])['Loan_Status'].value_counts(),
#'Percentage': train_set.groupby(['Self_Employed'])['Loan_Status'].value_counts(normalize=True)}))
#                           Freq  Percentage
#Self_Employed Loan_Status                  
#No            Y             343    0.686000
#              N             157    0.314000
#Yes           Y              56    0.682927
#              N              26    0.317073
    

#   Credit_History seems to be a very important predictor variable. The vast majority 
# of applicants whose credit history doesn't meet guidelines are refused.
#print(pd.DataFrame(data = {'Freq': train_set.groupby(['Credit_History'])['Loan_Status'].value_counts(),
#'Percentage': train_set.groupby(['Credit_History'])['Loan_Status'].value_counts(normalize=True)}))
#                                Freq  Percentage
#Credit_History Loan_Status                  
#0.0            N              82    0.921348
#               Y               7    0.078652
#1.0            Y             378    0.795789
#               N              97    0.204211

#   It's easier to get a loan if the property is semi-urban and harder if it's rural.
#print(pd.DataFrame(data = {'Freq': train_set.groupby(['Property_Area'])['Loan_Status'].value_counts(),
#'Percentage': train_set.groupby(['Property_Area'])['Loan_Status'].value_counts(normalize=True)}))
#                           Freq  Percentage
#Property_Area Loan_Status                  
#Rural         Y             110    0.614525
#              N              69    0.385475
#Semiurban     Y             179    0.768240
#              N              54    0.231760
#Urban         Y             133    0.658416
#              N              69    0.341584    

#   ApplicantIncome doesn't seem to have a significant influence on loan approval.

#   CoapplicantIncome has some influence on loan approval.

#   LoanAmount has some influence on loan approval.


# As concerning the correlation between the features, we can see that the stronger 
# correlation with Loan_Status is for variable Credit_History. 
# This shows that people whose credit history meets guidelines have higher probability to take the loan.


#### Visualizations  

# We will plot separately all the above plots.

# Number of approved loans by Gender of Applicant


from matplotlib.colors import LinearSegmentedColormap
dict=(0,1)
colors = sns.color_palette("Set1", n_colors=len(dict))
cmap1 = LinearSegmentedColormap.from_list("my_colormap", colors)
colors2 = colors[::-1]
cmap2 = LinearSegmentedColormap.from_list("my_colormap", colors2)



# Number of approved loans by Gender of Applicant

plt.figure(figsize=(10,10))
sns.countplot(x='Gender', hue="Loan_Status", data=train_set.loc[0:train_set.shape[0],], palette=colors[::-1])
plt.xlabel('Gender')
plt.ylabel('Number of approved loans')
plt.title('Loan Status by Gender of Applicant')
plt.legend(labels={'Approved','Rejected'})


# Percentage of approved loans by Gender of Applicant

# We need to compute a new column 'Refused' as 1 - 'Approved'
train_set3 = train_set2.copy()
train_set3['Refused'] = 1 - train_set3['Loan_Status']


train_set3.loc[0:train_set3.shape[0],].groupby('Gender').agg('mean')[['Loan_Status', 'Refused']].plot(kind='bar', figsize=(10, 10),
                                                          stacked=True, colormap = cmap2)
plt.title('Loan Status by Gender of Applicant')
plt.xlabel('Gender')
plt.ylabel('Percentage of approved loans')
plt.legend(labels={'Approved','Rejected'}, loc='upper right', bbox_to_anchor=(0.9, 0.88),
           bbox_transform=plt.gcf().transFigure)


# We see that a slightly larger proportion of female applicants are refused than male ones




# Number of approved loans by Marital Status of Applicant

plt.figure(figsize=(10,10))
sns.countplot(x='Married', hue="Loan_Status", data=train_set.loc[0:train_set.shape[0],], palette=colors[::-1])
plt.xlabel('Married')
plt.ylabel('Number of approved loans')
plt.title('Loan Status by Marital Status of Applicant')
plt.legend(labels={'Approved','Rejected'})


# Percentage of approved loans by Marital Status of Applicant

train_set3.loc[0:train_set3.shape[0],].groupby('Married').agg('mean')[['Loan_Status', 'Refused']].plot(kind='bar', figsize=(10, 10),
                                                          stacked=True, colormap = cmap2)
plt.title('Loan Status by Marital Status of Applicant')
plt.xlabel('Married')
plt.ylabel('Percentage of approved loans')
plt.legend(labels={'Approved','Rejected'}, loc='upper right', bbox_to_anchor=(0.9, 0.88),
           bbox_transform=plt.gcf().transFigure)

# We see that a larger proportion of unmarried applicants are refused than married ones





# Number of approved loans by number of Dependents of Applicant

plt.figure(figsize=(10,10))
sns.countplot(x='Dependents', hue="Loan_Status", data=train_set.loc[0:train_set.shape[0],], palette=colors[::-1])
plt.xlabel('Dependents')
plt.ylabel('Number of approved loans')
plt.title('Loan Status by number of Dependents of Applicant')
plt.legend(labels={'Approved','Rejected'})



# Percentage of approved loans by number of Dependents of Applicant

train_set3.loc[0:train_set3.shape[0],].groupby('Dependents').agg('mean')[['Loan_Status', 'Refused']].plot(kind='bar', figsize=(10, 10),
                                                          stacked=True, colormap = cmap2)
plt.title('Loan Status by number of Dependents of Applicant')
plt.xlabel('Dependents')
plt.ylabel('Percentage of approved loans')
plt.legend(labels={'Approved','Rejected'}, loc='upper right', bbox_to_anchor=(0.9, 0.88),
           bbox_transform=plt.gcf().transFigure)


# We see that a smaller proportion of applicants with 2 dependents are refused than applicants 
# with other number of dependents





# Number of approved loans by Education of Applicant

plt.figure(figsize=(10,10))
sns.countplot(x='Education', hue="Loan_Status", data=train_set.loc[0:train_set.shape[0],], palette=colors[::-1])
plt.xlabel('Education')
plt.ylabel('Number of approved loans')
plt.title('Loan Status by Education of Applicant')
plt.legend(labels={'Approved','Rejected'})


# Percentage of approved loans by Education of Applicant

train_set3.loc[0:train_set3.shape[0],].groupby('Education').agg('mean')[['Loan_Status', 'Refused']].plot(kind='bar', figsize=(10, 10),
                                                          stacked=True, colormap = cmap2)
plt.title('Loan Status by number of Dependents of Applicant')
plt.xlabel('Education')
plt.ylabel('Percentage of approved loans')
plt.legend(labels={'Approved','Rejected'}, loc='upper right', bbox_to_anchor=(0.9, 0.88),
           bbox_transform=plt.gcf().transFigure)


# We see that a larger proportion of non graduates are refused than graduates





# Number of approved loans by Employment status of Applicant

plt.figure(figsize=(10,10))
sns.countplot(x='Self_Employed', hue="Loan_Status", data=train_set.loc[0:train_set.shape[0],], palette=colors[::-1])
plt.xlabel('Self Employed')
plt.ylabel('Number of approved loans')
plt.title('Loan Status by Employment status of Applicant')
plt.legend(labels={'Approved','Rejected'})


# Percentage of approved loans by Employment status of Applicant

train_set3.loc[0:train_set3.shape[0],].groupby('Self_Employed').agg('mean')[['Loan_Status', 'Refused']].plot(kind='bar', figsize=(10, 10),
                                                          stacked=True, colormap = cmap2)
plt.title('Loan Status by Employment status of Applicant')
plt.xlabel('Self Employed')
plt.ylabel('Percentage of approved loans')
plt.legend(labels={'Approved','Rejected'}, loc='upper right', bbox_to_anchor=(0.9, 0.88),
           bbox_transform=plt.gcf().transFigure)


# We see that applicants that are self employed have a slightly worse approved rate




# Number of approved loans by Terms of Loan

plt.figure(figsize=(10,10))
sns.countplot(x='Loan_Amount_Term', hue="Loan_Status", data=train_set.loc[0:train_set.shape[0],], palette=colors[::-1])
plt.xlabel('Loan Amount Term')
plt.ylabel('Number of approved loans')
plt.title('Loan Status by Terms of Loan')
plt.legend(labels={'Approved','Rejected'})


# Percentage of approved loans by Terms of Loan

train_set3.loc[0:train_set3.shape[0],].groupby('Loan_Amount_Term').agg('mean')[['Loan_Status', 'Refused']].plot(kind='bar', figsize=(10, 10),
                                                          stacked=True, colormap = cmap2)
plt.title('Loan Status by Terms of Loan')
plt.xlabel('Loan Amount Term')
plt.ylabel('Percentage of approved loans')
plt.legend(labels={'Approved','Rejected'}, loc='upper right', bbox_to_anchor=(0.9, 0.88),
           bbox_transform=plt.gcf().transFigure)


# From these plots we can only infer that the majority of loans have a term of 360 months 





# Number of approved loans by Credit History of Applicant

plt.figure(figsize=(10,10))
sns.countplot(x='Credit_History', hue="Loan_Status", data=train_set.loc[0:train_set.shape[0],], palette=colors[::-1])
plt.xlabel('Credit History')
plt.ylabel('Number of approved loans')
plt.title('Loan Status by Credit History')
plt.legend(labels={'Approved','Rejected'})



# Percentage of approved loans by Credit History of Applicant

train_set3.loc[0:train_set3.shape[0],].groupby('Credit_History').agg('mean')[['Loan_Status', 'Refused']].plot(kind='bar', figsize=(10, 10),
                                                          stacked=True, colormap = cmap2)
plt.title('Loan Status by Credit History of Applicant')
plt.xlabel('Credit History')
plt.ylabel('Percentage of approved loans')
plt.legend(labels={'Approved','Rejected'}, loc='upper right', bbox_to_anchor=(0.9, 0.88),
           bbox_transform=plt.gcf().transFigure)


# Credit_History seems to be a very important predictor variable. 
# The vast majority of applicants whose credit history doesn't meet guidelines are refused.





# Number of approved loans by Property Area

plt.figure(figsize=(10,10))
sns.countplot(x='Property_Area', hue="Loan_Status", data=train_set.loc[0:train_set.shape[0],], palette=colors[::-1])
plt.xlabel('Property Area')
plt.ylabel('Number of approved loans')
plt.title('Loan Status by Property Area')
plt.legend(labels={'Approved','Rejected'})



# Percentage of approved loans by Property Area

train_set3.loc[0:train_set3.shape[0],].groupby('Property_Area').agg('mean')[['Loan_Status', 'Refused']].plot(kind='bar', figsize=(10, 10),
                                                          stacked=True, colormap = cmap2)
plt.title('Loan Status by Property Area')
plt.xlabel('Property_Area')
plt.ylabel('Percentage of approved loans')
plt.legend(labels={'Approved','Rejected'}, loc='upper right', bbox_to_anchor=(0.9, 0.88),
           bbox_transform=plt.gcf().transFigure)


# We see that it's easier to get a loan if the property is Semiurban and harder if it's Rural.




# Number of approved loans by Applicant Income

plt.figure(figsize=(10,10))
sns.boxplot(x="Loan_Status", y="ApplicantIncome", data=train_set)
plt.title('Loan Status by Applicant Income')

# Approved loans by Applicant Income with regression line

# Add a regression line that predicts the response variable Loan_Status as a function of 
# the explanatory variable ApplicantIncome using linear model (lm)

plt.figure(figsize=(10,10))
sns.regplot(x='ApplicantIncome', y='Loan_Status', data=train_set2)
plt.title('Loan Status by Applicant Income')

# We see that ApplicantIncome doesn't seem to have a significant influence on loan approval





# Number of approved loans by Coapplicant Income

plt.figure(figsize=(10,10))
sns.boxplot(x="Loan_Status", y="CoapplicantIncome", data=train_set)
plt.title('Loan Status by Coapplicant Income')


# Approved loans by Applicant Income with regression line

# Add a regression line that predicts the response variable Loan_Status as a function of 
# the explanatory variable ApplicantIncome using linear model (lm)

plt.figure(figsize=(10,10))
sns.regplot(x='CoapplicantIncome', y='Loan_Status', data=train_set2)
plt.title('Loan Status by Coapplicant Income')


# We see that CoapplicantIncome has some influence on loan approval




# Number of approved loans by Loan Amount

plt.figure(figsize=(10,10))
sns.boxplot(x="Loan_Status", y="LoanAmount", data=train_set)
plt.title('Loan Status by Loan Amount')


# Approved loans by Loan Amount with regression line

# Add a regression line that predicts the response variable Loan_Status as a function of 
# the explanatory variable LoanAmount using linear model (lm)

plt.figure(figsize=(10,10))
sns.regplot(x='LoanAmount', y='Loan_Status', data=train_set2)
plt.title('Loan Status by Loan Amount')


# We see that LoanAmount has some influence on loan approval




# Compute Predictor Importance

# For categorical predictor variables: Predictor Importance = 1 - PVal 
# from Pearson Chi-square test 
# For numerical predictor variables: Predictor Importance = 1 - PVal 
# from ANOVA F-Test for Equality of Mean


# Initialize the predictor importance data.frame
pr_imp = pd.DataFrame(index=range(0,len(cat)+len(cont)), columns=['Variable','Importance'])

# Predictor Importance for categorical variables
from scipy import stats
j=0
for i in cat:

    contingency_table = pd.crosstab(  # frequency count table
    train_set[i],
    train_set.Loan_Status,
    margins = True)

    f_obs = np.array(contingency_table.iloc[0:contingency_table.shape[0]-1,0:contingency_table.shape[1]-1])

    pr_imp.loc[j,:] = [i,1-stats.chi2_contingency(f_obs)[1]]
    j += 1
    
    
# Predictor Importance for continuous variables    
import statsmodels.api as sm
from statsmodels.formula.api import ols
for i in cont:
    mod = ols('Loan_Status ~ ' + i, data=train_set2).fit()
    aov_table = sm.stats.anova_lm(mod, typ=2)
    pr_imp.loc[j,:] = [i,1- aov_table['PR(>F)'][0]]
    j += 1

# Sort the data frame from the variable with the highest importance to the variable with the 
# lowest importance
pr_imp = pr_imp.sort_values(by='Importance', ascending=False)

#             Variable Importance
#7      Credit_History          1
#5       Property_Area   0.997864
#1             Married   0.965606
#3           Education     0.9569
#6    Loan_Amount_Term   0.878142
#9   CoapplicantIncome   0.857052
#10         LoanAmount   0.635264
#2          Dependents   0.632149
#0              Gender   0.291347
#8     ApplicantIncome  0.0927122
#4       Self_Employed  0.0579961

# From the above univariate analysis, we see that Credit_History, Property_Area, Married and
# Education are the most significant predictors.
# Surprisingly, ApplicantIncome is a very weak predictor, while CoapplicantIncome is a
# significant predictor.


# After this analysis, we will not ignore the outliers from ApplicantIncome, CoapplicantIncome
# and LoanAmount as most significant predictors are all categorical.





# Check for missing values (empty or NA) in the training set

total = train_set.isnull().sum().sort_values(ascending=False)
percent_1 = train_set.isnull().sum()/train_set.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
train_set_missing = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])

#                   Total    %
#Credit_History        50  8.1
#Self_Employed         32  5.2
#LoanAmount            22  3.6
#Dependents            15  2.4
#Loan_Amount_Term      14  2.3
#Gender                13  2.1
#Married                3  0.5
#Loan_Status            0  0.0
#Property_Area          0  0.0
#CoapplicantIncome      0  0.0
#ApplicantIncome        0  0.0
#Education              0  0.0
#Loan_ID                0  0.0


# Check for missing values (empty or NA) in the test set
total = test_set.isnull().sum().sort_values(ascending=False)
percent_1 = test_set.isnull().sum()/test_set.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
test_set_missing = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])

#                   Total    %
#Credit_History        29  7.9
#Self_Employed         23  6.3
#Gender                11  3.0
#Dependents            10  2.7
#Loan_Amount_Term       6  1.6
#LoanAmount             5  1.4
#Property_Area          0  0.0
#CoapplicantIncome      0  0.0
#ApplicantIncome        0  0.0
#Education              0  0.0
#Married                0  0.0
#Loan_ID                0  0.0


# We see that we have missing values in Gender, Married, Dependents, Self_Employed, LoanAmount, 
# Loan_Amount_Term and Credit_History in the training set and 
# Gender, Dependents, Self_Employed, LoanAmount, Loan_Amount_Term and Credit_History in 
# the test set.
    
# To tackle this problem, we are going to predict the missing values with the full data set, 
# which means that we need to combine the training and test sets together.


# Combine training and test sets 
full_set = train_set.append(test_set, sort= 'False', ignore_index=True)

# Check for missing values (empty or NA) in the full set (training + test)
total = full_set.isnull().sum().sort_values(ascending=False)
percent_1 = full_set.isnull().sum()/full_set.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
full_set_missing = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])

#                   Total     %
#Loan_Status          367  37.4
#Credit_History        79   8.1
#Self_Employed         55   5.6
#LoanAmount            27   2.8
#Dependents            25   2.5
#Gender                24   2.4
#Loan_Amount_Term      20   2.0
#Married                3   0.3
#Property_Area          0   0.0
#Loan_ID                0   0.0
#Education              0   0.0
#CoapplicantIncome      0   0.0
#ApplicantIncome        0   0.0


# Compute a summary of the full set

full_set.describe()
#       ApplicantIncome  CoapplicantIncome  Credit_History  LoanAmount  \
#count       981.000000         981.000000      902.000000  954.000000   
#mean       5179.795107        1601.916330        0.835920  142.511530   
#std        5695.104533        2718.772806        0.370553   77.421743   
#min           0.000000           0.000000        0.000000    9.000000   
#25%        2875.000000           0.000000        1.000000  100.000000   
#50%        3800.000000        1110.000000        1.000000  126.000000   
#75%        5516.000000        2365.000000        1.000000  162.000000   
#max       81000.000000       41667.000000        1.000000  700.000000   
#
#       Loan_Amount_Term  
#count        961.000000  
#mean         342.201873  
#std           65.100602  
#min            6.000000  
#25%          360.000000  
#50%          360.000000  
#75%          360.000000  
#max          480.000000  

# For categorical variables
full_set.describe(include='O')
#       Dependents Education Gender   Loan_ID Loan_Status Married  \
#count         956       981    957       981         614     978   
#unique          4         2      2       981           2       2   
#top             0  Graduate   Male  LP001486           Y     Yes   
#freq          545       763    775         1         422     631   
#
#       Property_Area Self_Employed  
#count            981           926  
#unique             3             2  
#top        Semiurban            No  
#freq             349           807  




####  Missing values Imputation


####  Variable "Married"

# Find which passengers have missing Married variables

married_missing_rows = np.where(full_set['Married'].isnull() | (full_set['Married']=="") | (full_set['Married']==" "))
full_set.loc[full_set.index[married_missing_rows], :]

#     ApplicantIncome  CoapplicantIncome  Credit_History Dependents Education  \
#104             3816              754.0             1.0        NaN  Graduate   
#228             4758                0.0             1.0        NaN  Graduate   
#435            10047                0.0             1.0        NaN  Graduate   
#
#     Gender  LoanAmount  Loan_Amount_Term   Loan_ID Loan_Status Married  \
#104    Male       160.0             360.0  LP001357           Y     NaN   
#228    Male       158.0             480.0  LP001760           Y     NaN   
#435  Female         NaN             240.0  LP002393           Y     NaN   
#
#    Property_Area Self_Employed  
#104         Urban            No  
#228     Semiurban            No  
#435     Semiurban            No 

# We will infer the missing Married values based on present data that seem relevant: 
# CoapplicantIncome

# We will consider the Married variable as "No", when the coapplicant income is zero, and
# "Yes" otherwise.

full_set.loc[(full_set['Married'].isnull()) & (full_set['CoapplicantIncome']==0), 'Married'] = 'No'
full_set.loc[(full_set['Married'].isnull()) & (full_set['CoapplicantIncome']!=0), 'Married'] = 'Yes'


####  Variables Gender and Dependents


# Replacing the missing values from "Gender" and "Dependents" with the most frequent category 
# might not be the best idea, since they may differ by groups of applicants.


# We will first examine if there are rows with both Gender and Dependents missing

full_set.loc[(full_set['Gender'].isnull()) & (full_set['Dependents'].isnull()), :]

#     ApplicantIncome  CoapplicantIncome  Credit_History Dependents Education  \
#752             3333             1250.0             1.0        NaN  Graduate   
#
#    Gender  LoanAmount  Loan_Amount_Term   Loan_ID Loan_Status Married  \
#752    NaN       110.0             360.0  LP001769         NaN      No   
#
#    Property_Area Self_Employed  
#752     Semiurban            No  

# We see that there is only one applicant with both these values missing. 
# This applicant is not Married but has higher income than the coapplicant.

# Let's investigate which gender has higher income

full_set.groupby(['Gender'])[['ApplicantIncome']].median()

#        ApplicantIncome
#Gender                 
#Female           3634.5
#Male             3865.0

# Since males have higher income, we will consider the missing value as "Male"
full_set.loc[(full_set['Gender'].isnull()) & (full_set['Dependents'].isnull()), 'Gender'] = 'Male'

# So all the other missing observations have only one of these variables missing. 

# First, we will examine the missing values of Dependents.

####  Variable Dependents

# Display variables for missing values of Dependents that seem relevant.

full_set_missing_dep = full_set.loc[full_set['Dependents'].isnull(), 
                                    ['Gender','Married', 'ApplicantIncome',
                                     'CoapplicantIncome', 'LoanAmount',
                                     'Loan_Amount_Term', 'Property_Area']]

# For continuous variables
full_set_missing_dep.describe()

#       ApplicantIncome  CoapplicantIncome  LoanAmount  Loan_Amount_Term
#count          25.0000          25.000000   22.000000         24.000000
#mean         5206.1200        1023.320000  123.545455        343.500000
#std          3355.3042        1312.081099   35.450472         80.444958
#min          2066.0000           0.000000   70.000000         84.000000
#25%          3250.0000           0.000000   96.250000        360.000000
#50%          3863.0000           0.000000  119.500000        360.000000
#75%          5417.0000        1750.000000  156.500000        360.000000
#max         14987.0000        4490.000000  180.000000        480.000000
                                                      
# For categorical variables
full_set_missing_dep.describe(include='O')

#       Gender Married Property_Area
#count      25      25            25
#unique      2       2             3
#top      Male     Yes         Urban
#freq       20      14            11

# We can examine the Dependents variable in relation to the Married variable

full_set.groupby(["Married", "Dependents"]).size().unstack(fill_value=0)
#Dependents    0    1    2  3+
#Married                      
#No          276   36   14  12
#Yes         269  124  146  79

# Probability table of Dependents in relation to Married 

full_set.groupby(['Married'])['Dependents'].value_counts(normalize=True).unstack(fill_value=0)

#Dependents         0         1         2        3+
#Married                                           
#No          0.816568  0.106509  0.041420  0.035503
#Yes         0.435275  0.200647  0.236246  0.127832


# The majority of unmarried people do not have dependents. However, ~ 19% of unmarried people
# have 1 or more dependents, so we can't impute missing dependents with zero for unmarried people.


############################################################################################


# We are going to predict missing Dependents variables using three different methods, in order 
# to investigate which one achieves the best results. 

########################################################################################

# Modal imputation
# For categorical variables, an easy way to impute the values is to use modal imputation, 
# or impute cases with the mode, or most common value. 

# Find the most common value

tmp1 = full_set.copy()

# Impute the cases

tmp1['Dependents'] = tmp1['Dependents'].fillna(tmp1['Dependents'].mode()[0])

# Check the distribution of Dependents before and after imputation

full_set['Dependents'].value_counts()

#0     545
#1     160
#2     160
#3+     91
#Name: Dependents, dtype: int64

tmp1['Dependents'].value_counts()

#0     570
#1     160
#2     160
#3+     91
#Name: Dependents, dtype: int64



# IterativeImputer


from fancyimpute import IterativeImputer

# We will use IterativeImputer (from fancyimpute package) to predict the missing Dependents 
# values

## For installation
#conda install ecos
#conda install CVXcanon
#pip install fancyimpute



# The imputation methods included in fancyimput require numerical data.
# In order to use them for categorical data, we have to assign a number to each level,
# apply the imputation method and then convert the numbers back to their corresponding levels.

# Specifically, we need to follow these steps:
# 1. Subset all categorical variables into another data frame
# 2. Change np.nan into an object data type, e.g. 'None' or 'Unknown'.
# 3. Encode the data frame.
# 4. Change back the value of encoded 'None' into np.nan
# 5. Use IterativeImpute (from fancyimpute package) to impute the missing values
# 6. Round the imputed values to convert them to the respective categorical values
# 7. Re-map the encoded data frame to its initial names


tmp2 = full_set.copy()

# Drop the unnecessary columns
tmp2.drop(['Loan_ID', 'Loan_Status'], axis=1, inplace=True)

# Create a boolean mask for categorical columns
categorical_feature_mask = tmp2.dtypes == object

# Get list of categorical column names
categorical_columns = tmp2.columns[categorical_feature_mask].tolist()

# Create a new data frame with categorical variables only (it contains the nan values) 
tmp2_obj = tmp2.loc[:,categorical_columns]

# Take a copy
tmp2_obj_None = tmp2_obj.copy()

# Then replace all nan values with None
tmp2_obj_None = tmp2_obj_None.fillna('None')

# Encode tha data (convert them into numerical data)
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
d = defaultdict(LabelEncoder)
tmp2_obj_encoded = tmp2_obj_None.apply(lambda x: d[x.name].fit_transform(x))

# Change back the value of encoded 'None' into np.nan
tmp2_obj_encoded[tmp2_obj_None[:] == 'None'] = tmp2_obj

# Get list of non-categorical column names
non_categorical_columns = tmp2.columns[~categorical_feature_mask].tolist()

# Create a new data frame with numerical variables only (it contains the nan values) 
tmp2_num = tmp2.loc[:,non_categorical_columns]

# Create a new data frame with all variables (numerical + categorical)
tmp2_all = pd.concat([tmp2_obj_encoded, tmp2_num], axis=1)

#fancy impute removes column names
imp_cols = tmp2_all.columns.values

# Model each feature with missing values as a function of other features, and use this
# estimate for imputation.
tmp2_imputed = pd.DataFrame(IterativeImputer(verbose=False).fit_transform(tmp2_all), columns= imp_cols)

# Round the imputed values to convert them to the respective categorical values
tmp2_imputed = np.round(tmp2_imputed)

# Drop numerical columns again
tmp2_imputed.drop(non_categorical_columns, axis=1, inplace=True)

# Encode the data again, in order to be able to reverse them back
tmp2_obj_encoded_new = tmp2_imputed.apply(lambda x: d[x.name].fit_transform(x))

# Inverse the encoded data
tmp2_obj_encoded_new =tmp2_obj_encoded_new.apply(lambda x: d[x.name].inverse_transform(x))

# Check the labels
tmp2_obj_encoded_new['Dependents'].unique()
#array([0., 1., 2., 3.])

# We see that the labels are not the same as the original

# Take the old labels and replace the labels into tmp2_obj_encoded_new

le=LabelEncoder()
le.fit(tmp2_obj['Dependents'].astype('str'))
# Create a dataframe with the mapped labels
labels_df = pd.DataFrame(data = {'en':range(len(le.classes_)), 'cl':le.classes_ })
#   en   cl
#0   0    0
#1   1    1
#2   2    2
#3   3   3+
#4   4  nan

# Convert them to dicionary without taking the nan label, since it has been imputed in the new 
# data frame
labels_dict = labels_df.set_index('en')['cl'][0:labels_df.shape[0]-1].to_dict()
#{0: '0', 1: '1', 2: '2', 3: '3+'}

tmp2_obj_encoded_new['Dependents'] = tmp2_obj_encoded_new['Dependents'].map(labels_dict)



# Check the distribution of Dependents before and after imputation

full_set['Dependents'].value_counts()

#0     545
#1     160
#2     160
#3+     91
#Name: Dependents, dtype: int64

tmp2_obj_encoded_new['Dependents'].value_counts()

#0     555
#1     175
#2     160
#3+     91
#Name: Dependents, dtype: int64




# SoftImpute


from fancyimpute import  SoftImpute

# We will use SoftImpute (from fancyimpute package) to predict the missing Dependents 
# values


tmp3 = full_set.copy()

# Drop the unnecessary columns
tmp3.drop(['Loan_ID', 'Loan_Status'], axis=1, inplace=True)

# Create a boolean mask for categorical columns
categorical_feature_mask = tmp3.dtypes == object

# Get list of categorical column names
categorical_columns = tmp3.columns[categorical_feature_mask].tolist()

# Create a new data frame with categorical variables only (it contains the nan values) 
tmp3_obj = tmp3.loc[:,categorical_columns]

# Take a copy
tmp3_obj_None = tmp3_obj.copy()

# Then replace all nan values with None
tmp3_obj_None = tmp3_obj_None.fillna('None')

# Encode tha data (convert them into numerical data)
#from sklearn.preprocessing import LabelEncoder
#from collections import defaultdict
d = defaultdict(LabelEncoder)
tmp3_obj_encoded = tmp3_obj_None.apply(lambda x: d[x.name].fit_transform(x))

# Change back the value of encoded 'None' into np.nan
tmp3_obj_encoded[tmp3_obj_None[:] == 'None'] = tmp3_obj

# Get list of non-categorical column names
non_categorical_columns = tmp3.columns[~categorical_feature_mask].tolist()

# Create a new data frame with numerical variables only (it contains the nan values) 
tmp3_num = tmp3.loc[:,non_categorical_columns]

# Create a new data frame with all variables (numerical + categorical)
tmp3_all = pd.concat([tmp3_obj_encoded, tmp3_num], axis=1)

#fancy impute removes column names
imp_cols = tmp3_all.columns.values

# Impute missing values
tmp3_imputed = pd.DataFrame(SoftImpute(verbose=False).fit_transform(tmp3_all), columns= imp_cols)

# Round the imputed values to convert them to the respective categorical values
tmp3_imputed = np.round(tmp3_imputed)

# Drop numerical columns again
tmp3_imputed.drop(non_categorical_columns, axis=1, inplace=True)

# Encode the data again, in order to be able to reverse them back
tmp3_obj_encoded_new = tmp3_imputed.apply(lambda x: d[x.name].fit_transform(x))

# Inverse the encoded data
tmp3_obj_encoded_new =tmp3_obj_encoded_new.apply(lambda x: d[x.name].inverse_transform(x))

# Check the labels
tmp3_obj_encoded_new['Dependents'].unique()
#array([0., 1., 2., 3.])

# We see that the labels are not the same as the original

# Take the old labels and replace the labels into tmp2_obj_encoded_new

le=LabelEncoder()
le.fit(tmp3_obj['Dependents'].astype('str'))
# Create a dataframe with the mapped labels
labels_df = pd.DataFrame(data = {'en':range(len(le.classes_)), 'cl':le.classes_ })
#   en   cl
#0   0    0
#1   1    1
#2   2    2
#3   3   3+
#4   4  nan

# Convert them to dicionary without taking the nan label, since it has been imputed in the new 
# data frame
labels_dict = labels_df.set_index('en')['cl'][0:labels_df.shape[0]-1].to_dict()
#{0: '0', 1: '1', 2: '2', 3: '3+'}

tmp3_obj_encoded_new['Dependents'] = tmp3_obj_encoded_new['Dependents'].map(labels_dict)



# Check the distribution of Dependents before and after imputation

full_set['Dependents'].value_counts()

#0     545
#1     160
#2     160
#3+     91
#Name: Dependents, dtype: int64

tmp3_obj_encoded_new['Dependents'].value_counts()

#0     558
#1     172
#2     160
#3+     91
#Name: Dependents, dtype: int64




# KNN


from fancyimpute import KNN

# We will use KNN (from fancyimpute package) to predict the missing Dependents 
# values


tmp4 = full_set.copy()

# Drop the unnecessary columns
tmp4.drop(['Loan_ID', 'Loan_Status'], axis=1, inplace=True)

# Create a boolean mask for categorical columns
categorical_feature_mask = tmp4.dtypes == object

# Get list of categorical column names
categorical_columns = tmp4.columns[categorical_feature_mask].tolist()

# Create a new data frame with categorical variables only (it contains the nan values) 
tmp4_obj = tmp4.loc[:,categorical_columns]

# Take a copy
tmp4_obj_None = tmp4_obj.copy()

# Then replace all nan values with None
tmp4_obj_None = tmp4_obj_None.fillna('None')

# Encode tha data (convert them into numerical data)
#from sklearn.preprocessing import LabelEncoder
#from collections import defaultdict
d = defaultdict(LabelEncoder)
tmp4_obj_encoded = tmp4_obj_None.apply(lambda x: d[x.name].fit_transform(x))

# Change back the value of encoded 'None' into np.nan
tmp4_obj_encoded[tmp4_obj_None[:] == 'None'] = tmp4_obj

# Get list of non-categorical column names
non_categorical_columns = tmp4.columns[~categorical_feature_mask].tolist()

# Create a new data frame with numerical variables only (it contains the nan values) 
tmp4_num = tmp4.loc[:,non_categorical_columns]

# Create a new data frame with all variables (numerical + categorical)
tmp4_all = pd.concat([tmp4_obj_encoded, tmp4_num], axis=1)

#fancy impute removes column names
imp_cols = tmp4_all.columns.values

# Use 3 nearest rows which have a feature to fill in each row's missing features
tmp4_imputed = pd.DataFrame(KNN(k=3,verbose=False).fit_transform(tmp4_all), columns= imp_cols)

# Round the imputed values to convert them to the respective categorical values
tmp4_imputed = np.round(tmp4_imputed)

# Drop numerical columns again
tmp4_imputed.drop(non_categorical_columns, axis=1, inplace=True)

# Encode the data again, in order to be able to reverse them back
tmp4_obj_encoded_new = tmp4_imputed.apply(lambda x: d[x.name].fit_transform(x))

# Inverse the encoded data
tmp4_obj_encoded_new =tmp4_obj_encoded_new.apply(lambda x: d[x.name].inverse_transform(x))

# Check the labels
tmp4_obj_encoded_new['Dependents'].unique()
#array([0., 1., 2., 3.])

# We see that the labels are not the same as the original

# Take the old labels and replace the labels into tmp2_obj_encoded_new

le=LabelEncoder()
le.fit(tmp4_obj['Dependents'].astype('str'))
# Create a dataframe with the mapped labels
labels_df = pd.DataFrame(data = {'en':range(len(le.classes_)), 'cl':le.classes_ })
#   en   cl
#0   0    0
#1   1    1
#2   2    2
#3   3   3+
#4   4  nan

# Convert them to dicionary without taking the nan label, since it has been imputed in the new 
# data frame
labels_dict = labels_df.set_index('en')['cl'][0:labels_df.shape[0]-1].to_dict()
#{0: '0', 1: '1', 2: '2', 3: '3+'}

tmp4_obj_encoded_new['Dependents'] = tmp4_obj_encoded_new['Dependents'].map(labels_dict)



# Check the distribution of Dependents before and after imputation

full_set['Dependents'].value_counts()

#0     545
#1     160
#2     160
#3+     91
#Name: Dependents, dtype: int64

tmp4_obj_encoded_new['Dependents'].value_counts()

#0     557
#1     169
#2     164
#3+     91
#Name: Dependents, dtype: int64




# Compare the original distribution of Dependents with the predicted using modal imputation,
# IterativeImputer, SoftImpute and KNN imputation in order to select the best prediction.


# Create a function that adds labels to plots
def add_value_labels(ax, spacing=5):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label.
        space = spacing
        # Vertical alignment for positive values
        vertical_alignment = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            vertical_alignment = 'top'

        # Use Y value as label and format number with four decimal places
        label = "{:.4f}".format(y_value)

        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=vertical_alignment)      # Vertically align label differently for
                                        # positive and negative values.



# Plot Dependents distributions


fig1, [[axis1, axis2, axis3],[axis4, axis5, axis6]] = plt.subplots(2,3,figsize=(15,4))
plt.subplots_adjust(hspace = .7)
axis1.set_title('Original Dependents values',pad=20)
axis2.set_title('New Dependents values using modal imputation',pad=20)
axis3.set_title('New Dependents values using IterativeImputer',pad=20)
axis4.set_title('New Dependents values using SoftImpute',pad=20)
axis5.set_title('New Dependents values using KNN imputation',pad=20)

# plot original Dependents values
(full_set.Dependents.value_counts()/(len(full_set))).sort_index().plot(kind="bar", ax=axis1, rot=0)
add_value_labels(axis1)
axis1.set_ylim([0,0.75])
# plot imputed Dependents values
(tmp1.Dependents.value_counts()/(len(tmp1))).sort_index().plot(kind="bar",ax=axis2, rot=0)
add_value_labels(axis2)
axis2.set_ylim([0,0.75])
(tmp2_obj_encoded_new.Dependents.value_counts()/(len(tmp2_obj_encoded_new))).sort_index().plot(kind="bar",ax=axis3, rot=0)
add_value_labels(axis3)
axis3.set_ylim([0,0.75])
(tmp3_obj_encoded_new.Dependents.value_counts()/(len(tmp3_obj_encoded_new))).sort_index().plot(kind="bar",ax=axis4, rot=0)
add_value_labels(axis4)
axis4.set_ylim([0,0.75])
(tmp4_obj_encoded_new.Dependents.value_counts()/(len(tmp4_obj_encoded_new))).sort_index().plot(kind="bar",ax=axis5, rot=0)
add_value_labels(axis5)
axis5.set_ylim([0,0.75])


# We see that the best prediction is probably achieved using KNN imputation, so we will use these 
# values to replace the missing Dependents values.

# Replace missing Dependents values with the predicted 
full_set['Dependents'] = tmp4_obj_encoded_new['Dependents']

# Show number of missing Dependents values
full_set['Dependents'].isnull().sum()
#0




####  Variable Gender 



# We will compare again some imputation methods for the missing Gender values


# IterativeImputer

tmp1 = full_set.copy()

# Drop the unnecessary columns
tmp1.drop(['Loan_ID', 'Loan_Status'], axis=1, inplace=True)

# Create a boolean mask for categorical columns
categorical_feature_mask = tmp1.dtypes == object

# Get list of categorical column names
categorical_columns = tmp1.columns[categorical_feature_mask].tolist()

# Create a new data frame with categorical variables only (it contains the nan values) 
tmp1_obj = tmp1.loc[:,categorical_columns]

# Take a copy
tmp1_obj_None = tmp1_obj.copy()

# Then replace all nan values with None
tmp1_obj_None = tmp1_obj_None.fillna('None')

# Encode tha data (convert them into numerical data)
d = defaultdict(LabelEncoder)
tmp1_obj_encoded = tmp1_obj_None.apply(lambda x: d[x.name].fit_transform(x))

# Change back the value of encoded 'None' into np.nan
tmp1_obj_encoded[tmp1_obj_None[:] == 'None'] = tmp1_obj

# Get list of non-categorical column names
non_categorical_columns = tmp1.columns[~categorical_feature_mask].tolist()

# Create a new data frame with numerical variables only (it contains the nan values) 
tmp1_num = tmp1.loc[:,non_categorical_columns]

# Create a new data frame with all variables (numerical + categorical)
tmp1_all = pd.concat([tmp1_obj_encoded, tmp1_num], axis=1)

#fancy impute removes column names
imp_cols = tmp1_all.columns.values

# Model each feature with missing values as a function of other features, and use this
# estimate for imputation.
tmp1_imputed = pd.DataFrame(IterativeImputer(verbose=False).fit_transform(tmp1_all), columns= imp_cols)

# Round the imputed values to convert them to the respective categorical values
tmp1_imputed = np.round(tmp1_imputed)

# If any value is greater than 1, which is the greatest number in this case representing a level,
# then replace that value, with 1
tmp1_imputed['Gender'].values[tmp1_imputed['Gender'] > 1] = 1

# Drop numerical columns again
tmp1_imputed.drop(non_categorical_columns, axis=1, inplace=True)

# Encode the data again, in order to be able to reverse them back
tmp1_obj_encoded_new = tmp1_imputed.apply(lambda x: d[x.name].fit_transform(x))

# Inverse the encoded data
tmp1_obj_encoded_new =tmp1_obj_encoded_new.apply(lambda x: d[x.name].inverse_transform(x))

# Check the labels
tmp1_obj_encoded_new['Gender'].unique()
#array([1., 0.])

# We see that the labels are not the same as the original

# Take the old labels and replace the labels into tmp1_obj_encoded_new

le=LabelEncoder()
le.fit(tmp1_obj['Gender'].astype('str'))
# Create a dataframe with the mapped labels
labels_df = pd.DataFrame(data = {'en':range(len(le.classes_)), 'cl':le.classes_ })
#   en      cl
#0   0  Female
#1   1    Male
#2   2     nan

# Convert them to dicionary without taking the nan label, since it has been imputed in the new 
# data frame
labels_dict = labels_df.set_index('en')['cl'][0:labels_df.shape[0]-1].to_dict()
#{0: 'Female', 1: 'Male'}

tmp1_obj_encoded_new['Gender'] = tmp1_obj_encoded_new['Gender'].map(labels_dict)



# Check the distribution of Dependents before and after imputation

full_set['Gender'].value_counts()

#Male      776
#Female    182
#Name: Gender, dtype: int64

tmp1_obj_encoded_new['Gender'].value_counts()

#Male      799
#Female    182
#Name: Gender, dtype: int64




# SoftImpute



tmp2 = full_set.copy()

# Drop the unnecessary columns
tmp2.drop(['Loan_ID', 'Loan_Status'], axis=1, inplace=True)

# Create a boolean mask for categorical columns
categorical_feature_mask = tmp2.dtypes == object

# Get list of categorical column names
categorical_columns = tmp2.columns[categorical_feature_mask].tolist()

# Create a new data frame with categorical variables only (it contains the nan values) 
tmp2_obj = tmp2.loc[:,categorical_columns]

# Take a copy
tmp2_obj_None = tmp2_obj.copy()

# Then replace all nan values with None
tmp2_obj_None = tmp2_obj_None.fillna('None')

# Encode tha data (convert them into numerical data)
#from sklearn.preprocessing import LabelEncoder
#from collections import defaultdict
d = defaultdict(LabelEncoder)
tmp2_obj_encoded = tmp2_obj_None.apply(lambda x: d[x.name].fit_transform(x))

# Change back the value of encoded 'None' into np.nan
tmp2_obj_encoded[tmp2_obj_None[:] == 'None'] = tmp2_obj

# Get list of non-categorical column names
non_categorical_columns = tmp2.columns[~categorical_feature_mask].tolist()

# Create a new data frame with numerical variables only (it contains the nan values) 
tmp2_num = tmp2.loc[:,non_categorical_columns]

# Create a new data frame with all variables (numerical + categorical)
tmp2_all = pd.concat([tmp2_obj_encoded, tmp2_num], axis=1)

#fancy impute removes column names
imp_cols = tmp2_all.columns.values

# Impute missing values
tmp2_imputed = pd.DataFrame(SoftImpute(verbose=False).fit_transform(tmp2_all), columns= imp_cols)

# Round the imputed values to convert them to the respective categorical values
tmp2_imputed = np.round(tmp2_imputed)

# If any value is greater than 1, which is the greatest number in this case representing a level,
# then replace that value, with 1
tmp2_imputed['Gender'].values[tmp2_imputed['Gender'] > 1] = 1

# Drop numerical columns again
tmp2_imputed.drop(non_categorical_columns, axis=1, inplace=True)

# Encode the data again, in order to be able to reverse them back
tmp2_obj_encoded_new = tmp2_imputed.apply(lambda x: d[x.name].fit_transform(x))

# Inverse the encoded data
tmp2_obj_encoded_new =tmp2_obj_encoded_new.apply(lambda x: d[x.name].inverse_transform(x))

# Check the labels
tmp2_obj_encoded_new['Gender'].unique()
#array([1., 0.])

# We see that the labels are not the same as the original

# Take the old labels and replace the labels into tmp2_obj_encoded_new

le=LabelEncoder()
le.fit(tmp2_obj['Gender'].astype('str'))
# Create a dataframe with the mapped labels
labels_df = pd.DataFrame(data = {'en':range(len(le.classes_)), 'cl':le.classes_ })
#   en      cl
#0   0  Female
#1   1    Male
#2   2     nan

# Convert them to dicionary without taking the nan label, since it has been imputed in the new 
# data frame
labels_dict = labels_df.set_index('en')['cl'][0:labels_df.shape[0]-1].to_dict()
#{0: 'Female', 1: 'Male'}

tmp2_obj_encoded_new['Gender'] = tmp2_obj_encoded_new['Gender'].map(labels_dict)



# Check the distribution of Dependents before and after imputation

full_set['Gender'].value_counts()

#Male      776
#Female    182
#Name: Gender, dtype: int64

tmp2_obj_encoded_new['Gender'].value_counts()

#Male      790
#Female    191
#Name: Gender, dtype: int64




# KNN



tmp3 = full_set.copy()

# Drop the unnecessary columns
tmp3.drop(['Loan_ID', 'Loan_Status'], axis=1, inplace=True)

# Create a boolean mask for categorical columns
categorical_feature_mask = tmp3.dtypes == object

# Get list of categorical column names
categorical_columns = tmp3.columns[categorical_feature_mask].tolist()

# Create a new data frame with categorical variables only (it contains the nan values) 
tmp3_obj = tmp3.loc[:,categorical_columns]

# Take a copy
tmp3_obj_None = tmp3_obj.copy()

# Then replace all nan values with None
tmp3_obj_None = tmp3_obj_None.fillna('None')

# Encode tha data (convert them into numerical data)
#from sklearn.preprocessing import LabelEncoder
#from collections import defaultdict
d = defaultdict(LabelEncoder)
tmp3_obj_encoded = tmp3_obj_None.apply(lambda x: d[x.name].fit_transform(x))

# Change back the value of encoded 'None' into np.nan
tmp3_obj_encoded[tmp3_obj_None[:] == 'None'] = tmp3_obj

# Get list of non-categorical column names
non_categorical_columns = tmp3.columns[~categorical_feature_mask].tolist()

# Create a new data frame with numerical variables only (it contains the nan values) 
tmp3_num = tmp3.loc[:,non_categorical_columns]

# Create a new data frame with all variables (numerical + categorical)
tmp3_all = pd.concat([tmp3_obj_encoded, tmp3_num], axis=1)

#fancy impute removes column names
imp_cols = tmp3_all.columns.values

# Use 3 nearest rows which have a feature to fill in each row's missing features
tmp3_imputed = pd.DataFrame(KNN(k=3,verbose=False).fit_transform(tmp3_all), columns= imp_cols)

# Round the imputed values to convert them to the respective categorical values
tmp3_imputed = np.round(tmp3_imputed)

# If any value is greater than 1, which is the greatest number in this case representing a level,
# then replace that value, with 1
tmp3_imputed['Gender'].values[tmp3_imputed['Gender'] > 1] = 1

# Drop numerical columns again
tmp3_imputed.drop(non_categorical_columns, axis=1, inplace=True)

# Encode the data again, in order to be able to reverse them back
tmp3_obj_encoded_new = tmp3_imputed.apply(lambda x: d[x.name].fit_transform(x))

# Inverse the encoded data
tmp3_obj_encoded_new =tmp3_obj_encoded_new.apply(lambda x: d[x.name].inverse_transform(x))

# Check the labels
tmp3_obj_encoded_new['Gender'].unique()
#array([1., 0.])

# We see that the labels are not the same as the original

# Take the old labels and replace the labels into tmp2_obj_encoded_new

le=LabelEncoder()
le.fit(tmp3_obj['Gender'].astype('str'))
# Create a dataframe with the mapped labels
labels_df = pd.DataFrame(data = {'en':range(len(le.classes_)), 'cl':le.classes_ })
#   en      cl
#0   0  Female
#1   1    Male
#2   2     nan

# Convert them to dicionary without taking the nan label, since it has been imputed in the new 
# data frame
labels_dict = labels_df.set_index('en')['cl'][0:labels_df.shape[0]-1].to_dict()
#{0: 'Female', 1: 'Male'}

tmp3_obj_encoded_new['Gender'] = tmp3_obj_encoded_new['Gender'].map(labels_dict)



# Check the distribution of Dependents before and after imputation

full_set['Gender'].value_counts()

#Male      776
#Female    182
#Name: Gender, dtype: int64

tmp3_obj_encoded_new['Gender'].value_counts()

#Male      798
#Female    183
#Name: Gender, dtype: int64




# Compare the original distribution of Gender with the predicted using 
# IterativeImputer, SoftImpute and KNN imputation in order to select the best prediction.

  
# Plot Gender distributions

fig1, [[axis1, axis2],[axis3, axis4]] = plt.subplots(2,2,figsize=(15,4))
plt.subplots_adjust(hspace = .7)
axis1.set_title('Original Gender values',pad=20)
axis2.set_title('New Gender values using IterativeImputer',pad=20)
axis3.set_title('New Gender values using SoftImpute',pad=20)
axis4.set_title('New Gender values using KNN imputation',pad=20)

# plot original Gender values
(full_set.Gender.value_counts()/(len(full_set))).sort_index().plot(kind="bar", ax=axis1, rot=0)
add_value_labels(axis1)
axis1.set_ylim([0,1.05])
# plot imputed Gender values
(tmp1_obj_encoded_new.Gender.value_counts()/(len(tmp1_obj_encoded_new))).sort_index().plot(kind="bar",ax=axis2, rot=0)
add_value_labels(axis2)
axis2.set_ylim([0,1.05])
(tmp2_obj_encoded_new.Gender.value_counts()/(len(tmp2_obj_encoded_new))).sort_index().plot(kind="bar",ax=axis3, rot=0)
add_value_labels(axis3)
axis3.set_ylim([0,1.05])
(tmp3_obj_encoded_new.Gender.value_counts()/(len(tmp3_obj_encoded_new))).sort_index().plot(kind="bar",ax=axis4, rot=0)
add_value_labels(axis4)
axis4.set_ylim([0,1.05])



# We see that the best prediction  (i.e., percentages of Gender before and after the prediction 
# look very similar) is probably achieved using KNN imputation, so we will use these 
# values to replace the missing Dependents values.

# Replace missing Dependents values with the predicted 
full_set['Gender'] = tmp3_obj_encoded_new['Gender']

# Show number of missing Dependents values
full_set['Gender'].isnull().sum()
#0



####  Variable Self_Employed

# Examine what percentage of applicants is self employed 

print(pd.DataFrame(data = {'Freq': full_set.groupby('Self_Employed').size(), 
                           'Percentage':full_set.groupby('Self_Employed').size().apply(lambda x: float(x) / full_set.groupby('Self_Employed').size().sum()*100)}))
#               Freq  Percentage
#Self_Employed                  
#No              807   87.149028
#Yes             119   12.850972


# Since ~87% of applicants are not self employed, it would be safe to impute the missing values 
# as "No", as there is a high probability of success.

# However we can investigate if there are any hidden patterns. 

# We will examine the relationship between the Self_Employed variable and the data that seem 
# relevant: Gender and Education

full_set.groupby(['Education','Gender','Self_Employed']).size()
#Education     Gender  Self_Employed
#Graduate      Female  No               124
#                      Yes               16
#              Male    No               502
#                      Yes               78
#Not Graduate  Female  No                28
#                      Yes                4
#              Male    No               153
#                      Yes               21
#dtype: int64

plt.figure(figsize=(10,10))
full_set.groupby(['Education','Gender','Self_Employed']).size().plot(kind="bar", title=
                'Self Employed Applicants in relation to Gender and Education')

# We see that the vast majority of applicants are not self employed regardless of their
# gender and education.
# Therefore, we can impute the missing Self_Employed values using the mode="No".


# Find the most common value

tmp1 = full_set.copy()

# Impute the cases

tmp1['Self_Employed'] = tmp1['Self_Employed'].fillna(tmp1['Self_Employed'].mode()[0])

# Check the distribution of Self_Employed before and after imputation

print(pd.DataFrame(data = {'Freq': full_set.groupby('Self_Employed').size(), 
                           'Percentage':full_set.groupby('Self_Employed').size().apply(lambda x: float(x) / full_set.groupby('Self_Employed').size().sum()*100)}))

#               Freq  Percentage
#Self_Employed                  
#No              807   87.149028
#Yes             119   12.850972


# Examine what percentage of applicants is self employed after the replacement
print(pd.DataFrame(data = {'Freq': tmp1.groupby('Self_Employed').size(), 
                           'Percentage':tmp1.groupby('Self_Employed').size().apply(lambda x: float(x) / tmp1.groupby('Self_Employed').size().sum()*100)}))

#               Freq  Percentage
#Self_Employed                  
#No              862   87.869521
#Yes             119   12.130479    
    
    
# We see that the percentages of self employed applicants before and after the prediction
# look very similar, which implies that our prediction was correct.    
    
    
# Replace missing Self_Employed values with the predicted 
full_set['Self_Employed'] = tmp1['Self_Employed']    
    



####  Variable Credit_History

# Credit History is a high impact variable. If credit history is not available, it possibly
# means that the applicant has not had many credit activities in the past. 
# The safest approach is to treat this variable as a separate category.
# Therefore, we will replace the missing values with "Not Available".


# Replace all NA values with Not Available
full_set['Credit_History'] = full_set['Credit_History'].fillna('Not Available')
full_set['Credit_History'] = full_set['Credit_History'].astype(str)




####  Variable LoanAmount


# Compute a basic summary of LoanAmount

full_set['LoanAmount'].describe()

#count    954.000000
#mean     142.511530
#std       77.421743
#min        9.000000
#25%      100.000000
#50%      126.000000
#75%      162.000000
#max      700.000000
#Name: LoanAmount, dtype: float64

# We will examine the relationship between the LoanAmount variable and the data that seem 
# relevant: Education and Self_Employed

lo_ed_em = full_set.pivot_table(values='LoanAmount', index=['Education'], columns=['Self_Employed'], aggfunc=np.median)

#Self_Employed     No    Yes
#Education                  
#Graduate       130.0  150.0
#Not Graduate   117.0  130.0


# Define function to return value of this pivot table
def mloan(x):
    return lo_ed_em[x['Self_Employed']][x['Education']]


# Replace missing values based on their category
full_set['LoanAmount'].fillna(full_set[full_set['LoanAmount'].isnull()].apply(mloan, axis=1), inplace=True)


# Compute a summary of the LoanAmount variable after prediction to ensure that everything is ok
full_set['LoanAmount'].describe()

#count    981.000000
#mean     142.122324
#std       76.399416
#min        9.000000
#25%      101.000000
#50%      128.000000
#75%      160.000000
#max      700.000000
#Name: LoanAmount, dtype: float64





####  Variable Loan_Amount_Term

# Compute a basic summary of Loan_Amount_Term

full_set['Loan_Amount_Term'].describe()

#count    961.000000
#mean     342.201873
#std       65.100602
#min        6.000000
#25%      360.000000
#50%      360.000000
#75%      360.000000
#max      480.000000
#Name: Loan_Amount_Term, dtype: float64


# We see that the 1st quartile,median and 3rd quartile values are 360, which means that the
# majority of Loan_Amount_Term values are 360.
# Let's check it out.

full_set.groupby('Loan_Amount_Term').size() # returns a Series
# full_set.groupby(['Loan_Amount_Term']).size().reset_index(name='Freq') # returns a DataFrame 

#Loan_Amount_Term
#6.0        1
#12.0       2
#36.0       3
#60.0       3
#84.0       7
#120.0      4
#180.0     66
#240.0      8
#300.0     20
#350.0      1
#360.0    823
#480.0     23
#dtype: int64


# We will examine the relationship between the Loan_Amount_Term variable and the data that seem 
# relevant: LoanAmount 


lo_te  = full_set.groupby(['Loan_Amount_Term'])[['LoanAmount']].median() 

#                  LoanAmount
#Loan_Amount_Term            
#6.0                     95.0
#12.0                   185.5
#36.0                   118.0
#60.0                   139.0
#84.0                   108.0
#120.0                   25.0
#180.0                  117.0
#240.0                  100.0
#300.0                  135.5
#350.0                  133.0
#360.0                  130.0
#480.0                  113.0



# There seems to be no linear relationship between LoanAmount and Loan_Amount_Term.

# Create a dataframe with the columns Loan_Amount_Term, median LoanAmount and number of 
# observations for each unique Loan_Amount_Term
pd.concat([lo_te.reset_index(), full_set.groupby(['Loan_Amount_Term']).size().reset_index(name='Freq')['Freq']], axis=1)

#    Loan_Amount_Term  LoanAmount  Freq
#0                6.0        95.0     1
#1               12.0       185.5     2
#2               36.0       118.0     3
#3               60.0       139.0     3
#4               84.0       108.0     7
#5              120.0        25.0     4
#6              180.0       117.0    66
#7              240.0       100.0     8
#8              300.0       135.5    20
#9              350.0       133.0     1
#10             360.0       130.0   823
#11             480.0       113.0    23

# We see that the LoanAmount does not have a big influence on the Loan_Amount_Term as it
# could be expected.

# Since the vast majority of the loans had a term of 360 months, we will replace the missing
# values with the mode, or most common value, which is 360.

full_set['Loan_Amount_Term'] = full_set['Loan_Amount_Term'].fillna(full_set['Loan_Amount_Term'].mode()[0])

# and since there are only a few unique Loan_Amount_Term values, we will convert this variable
# from integer to categorical.
full_set['Loan_Amount_Term'] = full_set['Loan_Amount_Term'].astype(str)








####  Create new variables that will help in the prediction

# Some variables should be combined or transformed someway in order to build a better model.
# We will make several graphs and computations to determine which transformation of variables
# can be added as new features.


####  Variable TotalIncome


# It is possible that some applicants have lower income but strong support co-applicants.
# So it might be a good idea to combine both incomes as total income.


full_set["TotalIncome"] = full_set.ApplicantIncome  + full_set.CoapplicantIncome


# Number of approved loans by TotalIncome

plt.figure(figsize=(10,10))
sns.boxplot(x="Loan_Status", y="TotalIncome", data=full_set.loc[0:train_set.shape[0],])
plt.title('Loan Status by Total Income')


# Approved loans by TotalIncome with regression line

# Add a regression line that predicts the response variable Loan_Status as a function of 
# the explanatory variable TotalIncome using linear model (lm)

# For the barplots, we need to convert 'Y', 'N' of Loan_Status to 1,0, respectively
full_set2 = full_set.copy()
full_set2.Loan_Status = pd.Series(np.where(full_set2.Loan_Status.values=='Y',1,0), full_set2.index)

plt.figure(figsize=(10,10))
sns.regplot(x='TotalIncome', y='Loan_Status', data=full_set2.loc[0:train_set.shape[0],])
plt.title('Loan Status by Total Income')


# We see that TotalIncome has some influence on loan approval, but it is not very significant.




####  Variable Loan_by_TotalIncome

# Create a variable as the loan amount divided by the sum of applicant and coapplicant income
# (TotalIncome). This variable gives an idea of how well the applicant is suited to pay back
# his loan.


full_set["Loan_by_TotalIncome"] = full_set.LoanAmount/full_set.TotalIncome



# Number of approved loans by Loan_by_TotalIncome

plt.figure(figsize=(10,10))
sns.boxplot(x="Loan_Status", y="Loan_by_TotalIncome", data=full_set.loc[0:train_set.shape[0],])
plt.title('Loan Status by Loan_by_TotalIncome')

# Approved loans by Loan_by_TotalIncome with regression line

# Add a regression line that predicts the response variable Loan_Status as a function of 
# the explanatory variable Loan_by_TotalIncome using linear model (lm)

# For the barplots, we need to convert 'Y', 'N' of Loan_Status to 1,0, respectively
full_set3 = full_set.copy()
full_set3.Loan_Status = pd.Series(np.where(full_set3.Loan_Status.values=='Y',1,0), full_set3.index)


plt.figure(figsize=(10,10))
sns.regplot(x='Loan_by_TotalIncome', y='Loan_Status', data=full_set3.loc[0:train_set.shape[0],])
plt.title('Loan Status by Loan_by_TotalIncome')


# There seems to be a strong negative relationship between loan amount/total income and 
# approved rate.



####  Variable Coapplicant

# Create a variable that indicates whether there is coapplicant or not.

# A coapplicant exists if the CoapplicantIncome is larger than zero, or the applicant is married.


full_set['Coapplicant'] = 0
full_set.loc[ ((full_set['CoapplicantIncome']>0) | (full_set['Married']=='Yes')) , 'Coapplicant'] = 1

# We will convert this variablefrom integer to categorical.
full_set['Coapplicant'] = full_set['Coapplicant'].astype(str)


# Number of approved loans by the existence or not of Coapplicant

plt.figure(figsize=(10,10))
sns.countplot(x='Coapplicant', hue="Loan_Status", data=full_set.loc[0:train_set.shape[0],], palette=colors)
plt.xlabel('Coapplicant')
plt.ylabel('Number of approved loans')
plt.title('Loan Status by the existence or not of Coapplicant')
plt.legend(labels={'Approved','Rejected'})

# We see that most applicants have a co-applicant 


# Percentage of approved loans by the existence or not of Coapplicant
# We need to compute a new column 'Refused' as 1 - 'Approved'
full_set4 = full_set.copy()
full_set4.Loan_Status = pd.Series(np.where(full_set4.Loan_Status.values=='Y',1,0), full_set4.index)

full_set4['Refused'] = 1 - full_set4['Loan_Status']

full_set4.loc[0:full_set4.shape[0],].groupby('Coapplicant').agg('mean')[['Loan_Status', 'Refused']].plot(kind='bar', figsize=(10, 10),
                                                          stacked=True, colormap = cmap1, rot=0)
plt.title('Loan Status by the existence or not of Coapplicant')
plt.xlabel('Coapplicant')
plt.ylabel('Percentage of approved loans')
plt.legend(labels={'Approved','Rejected'}, loc='upper right', bbox_to_anchor=(0.9, 0.88),
           bbox_transform=plt.gcf().transFigure)


# We see that a loan is slightly more likely to be approved if there is a coapplicant.



####  Variable FamilySize


# FamilySize contains the applicant itself, the coapplicant (if exists) and the number of 
# dependents.
numDependents  = full_set.Dependents.copy()
numDependents = numDependents.map({'0':0, '1':1, '2':2, '3+': 3})


full_set['FamilySize'] = numDependents+1
full_set.loc[ ((full_set['CoapplicantIncome']>0) | (full_set['Married']=='Yes')) , 'FamilySize'] = numDependents+2

full_set.groupby(['FamilySize']).size().reset_index(name='Freq')

#   FamilySize  Freq
#0           1   183
#1           2   399
#2           3   155
#3           4   160
#4           5    84


# Change FamilySize for those with  family sizes (>=3) to 3+ and finally convert it to categorical
full_set['DFamilySize'] = full_set['FamilySize'].map({0:'0', 1:'1', 2:'2', 3:'3+', 4:'3+', 5:'3+'})

full_set.groupby(['DFamilySize']).size().reset_index(name='Freq')

#  DFamilySize  Freq
#0           1   183
#1           2   399
#2          3+   399



# Number of approved loans by FamilySize

plt.figure(figsize=(10,10))
sns.countplot(x='DFamilySize', hue="Loan_Status", data=full_set.loc[0:train_set.shape[0],], 
              order=['1','2', '3+'], palette=colors)
plt.xlabel('DFamilySize')
plt.ylabel('Number of approved loans')
plt.title('Loan Status by FamilySize')
plt.legend(labels={'Approved','Rejected'})



# Percentage of approved loans by FamilySize
# We need to compute a new column 'Refused' as 1 - 'Approved'
full_set5 = full_set.copy()
full_set5.Loan_Status = pd.Series(np.where(full_set5.Loan_Status.values=='Y',1,0), full_set5.index)

full_set5['Refused'] = 1 - full_set5['Loan_Status']

full_set5.loc[0:full_set5.shape[0],].groupby('DFamilySize').agg('mean')[['Loan_Status', 'Refused']].plot(kind='bar', figsize=(10, 10),
                                                          stacked=True, colormap = cmap1, rot=0)
plt.title('Loan Status by FamilySize')
plt.xlabel('DFamilySize')
plt.ylabel('Percentage of approved loans')
plt.legend(labels={'Approved','Rejected'}, loc='upper right', bbox_to_anchor=(0.9, 0.88),
           bbox_transform=plt.gcf().transFigure)



# We see that if an applicant has a coapplicant or one dependent (DFamilySize=2) 
# or has a family of size >=3, it is slightly less likely to take the loan, but 
# if the family size is 1, i.e., if there is only the applicant, it helps to 
# take the loan.




####  Variable FamilyIncome

# Create a variable as the sum of applicant and coapplicant income (TotalIncome) divided by 
# the family size. This variable gives an idea of the actual income of a family.


full_set["FamilyIncome"] = full_set.TotalIncome/full_set.FamilySize



# Number of approved loans by FamilyIncome

plt.figure(figsize=(10,10))
sns.boxplot(x="Loan_Status", y="FamilyIncome", data=full_set.loc[0:train_set.shape[0],])
plt.title('Loan Status by Family Income')


# Add a regression line that predicts the response variable Loan_Status as a function of 
# the explanatory variable Loan_by_TotalIncome using linear model (lm)

# For the barplots, we need to convert 'Y', 'N' of Loan_Status to 1,0, respectively
full_set6 = full_set.copy()
full_set6.Loan_Status = pd.Series(np.where(full_set6.Loan_Status.values=='Y',1,0), full_set6.index)


plt.figure(figsize=(10,10))
sns.regplot(x='FamilyIncome', y='Loan_Status', data=full_set6.loc[0:train_set.shape[0],])
plt.title('Loan Status by Family Income')


# There seems to be a strong negative relationship between family income and approved rate.




####  Variable Loan_by_FamilyIncome

# Create a variable as the loan amount divided by family income.
# This variable gives an idea of how well a family is suited to pay back its loan.



full_set["Loan_by_FamilyIncome"] = full_set.LoanAmount/full_set.FamilyIncome



# Number of approved loans by Loan_by_FamilyIncome

plt.figure(figsize=(10,10))
sns.boxplot(x="Loan_Status", y="Loan_by_FamilyIncome", data=full_set.loc[0:train_set.shape[0],])
plt.title('Loan Status by Loan_by_FamilyIncome')

# Approved loans by Loan_by_FamilyIncome with regression line

# Add a regression line that predicts the response variable Loan_Status as a function of 
# the explanatory variable Loan_by_FamilyIncome using linear model (lm)

# For the barplots, we need to convert 'Y', 'N' of Loan_Status to 1,0, respectively
full_set7 = full_set.copy()
full_set7.Loan_Status = pd.Series(np.where(full_set7.Loan_Status.values=='Y',1,0), full_set7.index)


plt.figure(figsize=(10,10))
sns.regplot(x='Loan_by_FamilyIncome', y='Loan_Status', data=full_set7.loc[0:train_set.shape[0],])
plt.title('Loan Status by Loan_by_FamilyIncome')


# There seems to be a weak negative relationship between loan amount/family income and 
# approved rate.





####  Variable LoanPerMonth 

# Create a variable as the loan amount divided by loan term. It is the monthly installment
# made by a borrower to a lender. We will ignore the interest rate per month here.

full_set["LoanPerMonth"] = full_set.LoanAmount/full_set.Loan_Amount_Term.astype(float)


# Number of approved loans by LoanPerMonth

plt.figure(figsize=(10,10))
sns.boxplot(x="Loan_Status", y="LoanPerMonth", data=full_set.loc[0:train_set.shape[0],])
plt.title('Loan Status by LoanPerMonth')


# Add a regression line that predicts the response variable Loan_Status as a function of 
# the explanatory variable LoanPerMonth using linear model (lm)

# For the barplots, we need to convert 'Y', 'N' of Loan_Status to 1,0, respectively
full_set8 = full_set.copy()
full_set8.Loan_Status = pd.Series(np.where(full_set8.Loan_Status.values=='Y',1,0), full_set8.index)


plt.figure(figsize=(10,10))
sns.regplot(x='LoanPerMonth', y='Loan_Status', data=full_set8.loc[0:train_set.shape[0],])
plt.title('Loan Status by LoanPerMonth')


# Remove the outlier just from the plot
full_set9 = full_set.copy()
full_set9 = full_set9.drop(full_set9.index[full_set9['LoanPerMonth'].idxmax()])

full_set9.Loan_Status = pd.Series(np.where(full_set9.Loan_Status.values=='Y',1,0), full_set9.index)

plt.figure(figsize=(10,10))
sns.regplot(x='LoanPerMonth', y='Loan_Status', data=full_set9.loc[0:train_set.shape[0],])
plt.title('Loan Status by LoanPerMonth')


# There seems to be a weak negative relationship between loan per month and approved rate.





####  Extreme values treatment



# Extreme values treatment in distribution of LoanAmount, TotalIncome, Loan_by_TotalIncome,
# FamilyIncome, Loan_by_FamilyIncome and LoanPerMonth

# In this problem, the extreme values are practically possible, i.e. some people might apply 
# for high  value loans due to specific needs. So instead of treating them as outliers, 
# it's probably a good idea to take a log transformation of the monetary variables to 
# nullify their effect.



####  Variable LoanAmount 

full_set.LoanAmount.describe()

#count    981.000000
#mean     142.122324
#std       76.399416
#min        9.000000
#25%      101.000000
#50%      128.000000
#75%      160.000000
#max      700.000000
#Name: LoanAmount, dtype: float64


full_set['LogLoanAmount'] = np.log(full_set['LoanAmount'])


# Compare the original distribution of LoanAmount with its log version 

fig1, (axis1,axis2) = plt.subplots(1,2,figsize=(12,10))
axis1.set_title('Original LoanAmount')
axis2.set_title('Log LoanAmount')

full_set['LoanAmount'].hist(bins=30, ax=axis1, figure=fig1)
full_set['LogLoanAmount'].hist(bins=30, ax=axis2, figure=fig1)


full_set['LogLoanAmount'].describe()

#count    981.000000
#mean       4.846066
#std        0.467902
#min        2.197225
#25%        4.615121
#50%        4.852030
#75%        5.075174
#max        6.551080
#Name: LogLoanAmount, dtype: float64

# After log transformation the distribution looks much closer to normal and the effect of 
# extreme values has been significantly subsided.



####  Variable TotalIncome

# Since ApplicantIncome and CoapplicantIncome are better predictors when they are combined in 
# one variable, we will take the log transformation of TotalIncome.

full_set.TotalIncome.describe()
#count      981.000000
#mean      6781.711437
#std       6023.952550
#min       1442.000000
#25%       4166.000000
#50%       5314.000000
#75%       7308.000000
#max      81000.000000
#Name: TotalIncome, dtype: float64


full_set['LogTotalIncome'] = np.log(full_set['TotalIncome'])


# Compare the original distribution of TotalIncome with its log version 

fig1, (axis1,axis2) = plt.subplots(1,2,figsize=(12,10))
axis1.set_title('Original TotalIncome')
axis2.set_title('Log TotalIncome')

full_set['TotalIncome'].hist(bins=30, ax=axis1, figure=fig1)
full_set['LogTotalIncome'].hist(bins=30, ax=axis2, figure=fig1)


full_set['LogTotalIncome'].describe()
#count    981.000000
#mean       8.649904
#std        0.520593
#min        7.273786
#25%        8.334712
#50%        8.578100
#75%        8.896725
#max       11.302204
#Name: LogTotalIncome, dtype: float64


# After log transformation the distribution looks much closer to normal and the effect of 
# extreme values has been significantly subsided.



####  Variable Loan_by_TotalIncome


full_set.Loan_by_TotalIncome.describe()
#count    981.000000
#mean       0.024076
#std        0.009000
#min        0.001905
#25%        0.019231
#50%        0.024131
#75%        0.028432
#max        0.102273
#Name: Loan_by_TotalIncome, dtype: float64


plt.figure(figsize=(10,10))
full_set['Loan_by_TotalIncome'].hist(bins=30)

# This one looks symmetric, so we will not take the log transformation.



####  Variable FamilyIncome

full_set.FamilyIncome.describe()

#count      981.000000
#mean      3179.411240
#std       2882.540445
#min        457.500000
#25%       1648.333333
#50%       2476.333333
#75%       3644.000000
#max      31668.500000
#Name: FamilyIncome, dtype: float64



full_set['LogFamilyIncome'] = np.log(full_set['FamilyIncome'])


# Compare the original distribution of FamilyIncome with its log version 

fig1, (axis1,axis2) = plt.subplots(1,2,figsize=(12,10))
axis1.set_title('Original FamilyIncome')
axis2.set_title('Log FamilyIncome')

full_set['FamilyIncome'].hist(bins=30, ax=axis1, figure=fig1)
full_set['LogFamilyIncome'].hist(bins=30, ax=axis2, figure=fig1)


full_set['LogFamilyIncome'].describe()

#count    981.000000
#mean       7.830484
#std        0.644508
#min        6.125777
#25%        7.407520
#50%        7.814534
#75%        8.200837
#max       10.363078
#Name: LogFamilyIncome, dtype: float64


# After log transformation the distribution looks much closer to normal and the effect of 
# extreme values has been significantly subsided.



####  Variable Loan_by_FamilyIncome


full_set.Loan_by_FamilyIncome.describe()

#count    981.000000
#mean       0.061361
#std        0.038156
#min        0.003490
#25%        0.033201
#50%        0.051430
#75%        0.082687
#max        0.330847
#Name: Loan_by_FamilyIncome, dtype: float64

full_set['LogLoan_by_FamilyIncome'] = np.log(1000*full_set['Loan_by_FamilyIncome'])


# Compare the original distribution of Loan_by_FamilyIncome with its log version 

fig1, (axis1,axis2) = plt.subplots(1,2,figsize=(12,10))
axis1.set_title('Original Loan_by_FamilyIncome')
axis2.set_title('Log Loan_by_FamilyIncome')

full_set['Loan_by_FamilyIncome'].hist(bins=30, ax=axis1, figure=fig1)
full_set['LogLoan_by_FamilyIncome'].hist(bins=30, ax=axis2, figure=fig1)


full_set['LogLoan_by_FamilyIncome'].describe()

#count    981.000000
#mean       3.923337
#std        0.652381
#min        1.249922
#25%        3.502566
#50%        3.940219
#75%        4.415066
#max        5.801655
#Name: LogLoan_by_FamilyIncome, dtype: float64



# After log transformation the distribution looks much closer to normal and the effect of 
# extreme values has been significantly subsided.




####  Variable LoanPerMonth 


full_set.LoanPerMonth.describe()

#count    981.000000
#mean       0.490322
#std        0.943670
#min        0.025000
#25%        0.288889
#50%        0.361111
#75%        0.500000
#max       21.666667
#Name: LoanPerMonth, dtype: float64

full_set['LogLoanPerMonth'] = np.log(1000*full_set['LoanPerMonth'])


# Compare the original distribution of Loan_by_FamilyIncome with its log version 

fig1, (axis1,axis2) = plt.subplots(1,2,figsize=(12,10))
axis1.set_title('Original LoanPerMonth')
axis2.set_title('Log LoanPerMonth')

full_set['LoanPerMonth'].hist(bins=30, ax=axis1, figure=fig1)
full_set['LogLoanPerMonth'].hist(bins=30, ax=axis2, figure=fig1)


full_set['LogLoanPerMonth'].describe()

#count    981.000000
#mean       5.953151
#std        0.554539
#min        3.218876
#25%        5.666042
#50%        5.889186
#75%        6.214608
#max        9.983530
#Name: LogLoanPerMonth, dtype: float64


# After log transformation the distribution looks much closer to normal and the effect of 
# extreme values has been significantly subsided.



# Since log transformations of LoanAmount, TotalIncome, FamilyIncome, Loan_by_FamilyIncome,
# LoanPerMonth make the effect of extreme values less intense we will keep these variables
# and delete their corresponding original values from the data frame. 
# We will also delete the ApplicantIncome and CoapplicantIncome since the new variable
# LogTotalIncome is a better predictor. 
# Finally, we will delete the variable FamilySize, since we will keep its discretized version,
# DFamilySize.  



full_set.drop(["LoanAmount", "TotalIncome", "FamilyIncome", "Loan_by_FamilyIncome",
               "LoanPerMonth", "ApplicantIncome", "CoapplicantIncome", "FamilySize"], 
            axis=1, inplace=True)




####  Examine Correlation of Continuous Variables

# Examine the continuous variables separately, and remove any that are highly correlated.


# Determine numeric and integer variables in the data frame




nums = full_set.columns[~(full_set.dtypes == object)].tolist()
numVars = full_set.loc[:,full_set.columns[~(full_set.dtypes == object)].tolist()]

# Create correlation matrix
corr_matrix = numVars.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

upper.unstack().dropna().sort_values(ascending=False)
 
#LogLoanPerMonth          LogLoanAmount              0.791239
#LogLoan_by_FamilyIncome  LogFamilyIncome            0.739728
#LogFamilyIncome          LogTotalIncome             0.650087
#LogTotalIncome           LogLoanAmount              0.629461
#LogLoan_by_FamilyIncome  Loan_by_TotalIncome        0.580639
#LogLoanPerMonth          LogTotalIncome             0.542641
#LogTotalIncome           Loan_by_TotalIncome        0.485795
#LogFamilyIncome          Loan_by_TotalIncome        0.376948
#LogLoan_by_FamilyIncome  LogLoanAmount              0.375336
#LogFamilyIncome          LogLoanAmount              0.346063
#LogLoanPerMonth          LogLoan_by_FamilyIncome    0.298011
#LogLoanAmount            Loan_by_TotalIncome        0.290342
#LogLoanPerMonth          LogFamilyIncome            0.272775
#LogLoan_by_FamilyIncome  LogTotalIncome             0.190778
#LogLoanPerMonth          Loan_by_TotalIncome        0.170122
#dtype: float64

# Find index of feature columns with correlation greater than 0.8
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]

# Drop Highly Correlated Features (there aren't any here)
numVars = numVars.drop(numVars.columns[to_drop], axis=1)



# Compute Predictor Importance for the new features



# For categorical predictor variables: Predictor Importance = 1 - PVal 
# from Pearson Chi-square test 
# For numerical predictor variables: Predictor Importance = 1 - PVal 
# from ANOVA F-Test for Equality of Mean


full_set10 = full_set.copy()

# Drop the unnecessary columns
full_set10.drop(['Loan_ID', 'Loan_Status'], axis=1, inplace=True)


# Create a boolean mask for categorical columns
categorical_feature_mask = full_set10.dtypes == object

# Get list of categorical column names
categorical_columns = full_set10.columns[categorical_feature_mask].tolist()
#['Credit_History',
# 'Dependents',
# 'Education',
# 'Gender',
# 'Loan_Amount_Term',
# 'Married',
# 'Property_Area',
# 'Self_Employed',
# 'Coapplicant',
# 'DFamilySize']

# Get list of non-categorical column names
non_categorical_columns = full_set10.columns[~categorical_feature_mask].tolist()
#['Loan_by_TotalIncome',
# 'LogLoanAmount',
# 'LogTotalIncome',
# 'LogFamilyIncome',
# 'LogLoan_by_FamilyIncome',
# 'LogLoanPerMonth']


# Initialize the predictor importance data.frame
pr_imp = pd.DataFrame(index=range(0,len(categorical_columns)+len(non_categorical_columns)), columns=['Variable','Importance'])

# Predictor Importance for categorical variables
j=0
for i in categorical_columns:

    contingency_table = pd.crosstab(  # frequency count table
    full_set[i],
    full_set.Loan_Status,
    margins = True)

    f_obs = np.array(contingency_table.iloc[0:contingency_table.shape[0]-1,0:contingency_table.shape[1]-1])

    pr_imp.loc[j,:] = [i,1-stats.chi2_contingency(f_obs)[1]]
    j += 1
    
 

# We need to convert 'Y', 'N' of Loan_Status to 1,0, respectively
full_set11 = full_set.copy()
full_set11.Loan_Status = pd.Series(np.where(full_set11.Loan_Status.values=='Y',1,0), full_set11.index)
    
    
# Predictor Importance for continuous variables    
for i in non_categorical_columns:
    mod = ols('Loan_Status ~ ' + i, data=full_set11).fit()
    aov_table = sm.stats.anova_lm(mod, typ=2)
    pr_imp.loc[j,:] = [i,1- aov_table['PR(>F)'][0]]
    j += 1

# Sort the data frame from the variable with the highest importance to the variable with the 
# lowest importance
pr_imp = pr_imp.sort_values(by='Importance', ascending=False)

#                   Variable Importance
#0            Credit_History          1
#6             Property_Area   0.997864
#8               Coapplicant   0.962658
#10      Loan_by_TotalIncome   0.960594
#5                   Married   0.960259
#2                 Education     0.9569
#4          Loan_Amount_Term   0.869415
#9               DFamilySize   0.791054
#12           LogTotalIncome   0.753303
#1                Dependents   0.635799
#14  LogLoan_by_FamilyIncome   0.481643
#13          LogFamilyIncome   0.460989
#3                    Gender   0.206364
#15          LogLoanPerMonth  0.0695498
#11            LogLoanAmount  0.0433363
#7             Self_Employed  0.0289254



# From the above analysis, we see that Credit_History, Property_Area, Married and
# Education and the new features Coapplicant and Loan_by_TotalIncome are the most significant 
# predictors.








#### PREDICTION

# Encode Categorical features


# We will use Binary Encoding to convert the categorical variables into numeric.
# In this technique, the categories are first encoded as ordinal, then those integers are converted 
# into binary code, then the digits from that binary string are split into separate columns.
# This encodes the data in fewer dimensions than one-hot encoding.


import category_encoders as ce

encoder = ce.BinaryEncoder(cols=categorical_columns, impute_missing='False', handle_unknown='ignore')
full_set_binary = encoder.fit_transform(full_set)


## Scaling numerical variables
#
#from sklearn.preprocessing import StandardScaler 
#scale = StandardScaler().fit(full_set_binary[non_categorical_columns])
#full_set_binary[non_categorical_columns] = scale.transform(full_set_binary[non_categorical_columns])


# Split the data back into the original training and testing sets

train_set_new = full_set_binary.loc[0:train_set.shape[0]-1,]
test_set_new = full_set_binary.loc[train_set.shape[0]:full_set.shape[0],]


# Set the seed to ensure reproduceability
np.random.seed(1)


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import ShuffleSplit
shuffle_validator = ShuffleSplit(n_splits=20, test_size=0.2, random_state=0)

from sklearn.model_selection import GridSearchCV
import multiprocessing
n_jobs= multiprocessing.cpu_count()-1


# Create a list of predictors

predictors = [ ['Credit_History_0', 'Credit_History_1', 'Credit_History_2',
                'Property_Area_0', 'Property_Area_1', 'Property_Area_2',   
                'Coapplicant_0', 'Coapplicant_1', 'Loan_by_TotalIncome', 
                'Married_0', 'Married_1', 'Education_0', 'Education_1'] , # 1
               ['Credit_History_0', 'Credit_History_1', 'Credit_History_2',
                'Property_Area_0', 'Property_Area_1', 'Property_Area_2',   
                'Coapplicant_0', 'Coapplicant_1', 'Loan_by_TotalIncome', 
                'Married_0', 'Married_1', 'Education_0', 'Education_1','Loan_Amount_Term_0',
                'Loan_Amount_Term_1', 'Loan_Amount_Term_2', 'Loan_Amount_Term_3', 'Loan_Amount_Term_4',
                'DFamilySize_0', 'DFamilySize_1', 'DFamilySize_2', 'LogTotalIncome',
                'Dependents_0', 'Dependents_1', 'Dependents_2'], # 2
                ['Credit_History_0', 'Credit_History_1', 'Credit_History_2',
                 'Property_Area_0', 'Property_Area_1', 'Property_Area_2',   
                 'Coapplicant_0', 'Coapplicant_1', 'Loan_by_TotalIncome', 
                 'Married_0', 'Married_1', 'Education_0', 'Education_1','Loan_Amount_Term_0',
                 'Loan_Amount_Term_1', 'Loan_Amount_Term_2', 'Loan_Amount_Term_3', 'Loan_Amount_Term_4',
                 'DFamilySize_0', 'DFamilySize_1', 'DFamilySize_2', 'LogTotalIncome',
                 'Dependents_0', 'Dependents_1', 'Dependents_2', 'LogLoan_by_FamilyIncome',
                 'LogFamilyIncome', 'Gender_0', 'Gender_1', 'LogLoanPerMonth',
                 'LogLoanAmount', 'Self_Employed_0', 'Self_Employed_1'] ]




#####      Decision Trees

# A decision tree is built top-down from a root node and involves partitioning the data into subsets 
# that contain instances with similar values (homogenous).


from sklearn.tree import DecisionTreeClassifier


# We will tune the decision tree model to find the optimal parameter values.

# Parameters that will be tuned    
# max depth : indicates how deep the tree can be. The deeper the tree, the more splits 
# it has, which means that it captures more information about the data.
max_depth = np.arange(1,10)    
# min_samples_split: represents the minimum number of samples required to split an internal node.
# vary the parameter from 10% to 100% of the samples
min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True) 
# min_samples_leaf: represents the minimum number of samples required to be at a leaf node. 
min_samples_leaf = np.linspace(0.1, 0.5, 5, endpoint=True)
# max_features: represents the number of features to consider when looking for the best split.


decision_tree = DecisionTreeClassifier(random_state=1)


for idx,predictor in enumerate(predictors):   

    x_train = train_set_new.loc[:,predictor]
    y_train = train_set_new.loc[:,'Loan_Status']
    x_test = test_set_new.loc[:,predictor]
    
    max_features = list(range(1,x_train.shape[1])) 
    param_grid = {'max_depth' : max_depth, 'min_samples_split': min_samples_splits, 
              'min_samples_leaf': min_samples_leaf, 'max_features': max_features}
    
    decision_tree = DecisionTreeClassifier(random_state=1)
    
    decision_tree_cv = GridSearchCV(decision_tree, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs = n_jobs, verbose=150)
    # Set the verbose parameter in GridSearchCV to a positive number 
    # (the greater the number the more detail you will get) to print out progress
    decision_tree_cv.fit(x_train, y_train)
   
    print("Best parameters found on training set:")
    print(decision_tree_cv.best_params_)
    
    decision_tree_optimised = decision_tree_cv.best_estimator_
    decision_tree_optimised.fit(x_train, y_train)

    # Performance on the training set
    
    scores = cross_val_score(decision_tree_optimised, x_train, y_train, cv=shuffle_validator, scoring='accuracy')
    # Take the mean of the scores
    print("Model %d - Accuracy on Training Set: %0.4f (+/- %0.2f)" % (idx+1, scores.mean(), scores.std()))
    
    # Performance on the testing set
    y_pred = decision_tree_optimised.predict(x_test)
    
    # Save the solution to a dataframe with two columns: Loan_ID and Loan_Status (prediction)
    filename="dec_tree%d.csv" % (idx+1)
    my_solution  = pd.DataFrame({
            "Loan_ID": test_set_new.loc[:,'Loan_ID'],
            "Loan_Status": y_pred
            })
    my_solution.to_csv(filename, index=False)



# Summary for Decision Trees - Accuracy  
    
#                                                    
# 1. Predictors: Credit_History, Property_Area, Coapplicant,
#    Loan_by_TotalIncome, Married, Education
#    Best parameters found on training set:
#    {'max_depth': 1, 'max_features': 7, 'min_samples_leaf': 0.1, 'min_samples_split': 0.1}
#    Model 1 - Accuracy on Training Set: 0.8053 (+/- 0.02)
#    Accuracy on the public leaderboard: 0.7777777777777778
#
# 2. Predictors: Credit_History, Property_Area, Coapplicant,
#    Loan_by_TotalIncome, Married, Education, Loan_Amount_Term,
#    DFamilySize, LogTotalIncome, Dependents
#    Best parameters found on training set:    
#    {'max_depth': 1, 'max_features': 16, 'min_samples_leaf': 0.1, 'min_samples_split': 0.1}
#    Model 2 - Accuracy on Training Set: 0.8053 (+/- 0.02)
#    Accuracy on the public leaderboard: 0.7777777777777778
#  
# 3. Predictors: Credit_History, Property_Area, Coapplicant,
#    Loan_by_TotalIncome, Married, Education, Loan_Amount_Term,
#    DFamilySize, LogTotalIncome, Dependents, LogLoan_by_FamilyIncome,
#    LogFamilyIncome, Gender, LogLoanPerMonth, LogLoanAmount,Self_Employed
#    Best parameters found on training set:    
#    {'max_depth': 1, 'max_features': 8, 'min_samples_leaf': 0.1, 'min_samples_split': 0.1}
#    Model 3 - Accuracy on Training Set: 0.8053 (+/- 0.02)
#    Accuracy on the public leaderboard: 0.7777777777777778

 
    
    
    
    
#####      Random Forest


# Random Forest fits multiple classification trees to random subsets of the input data and
# averages the predictions to return whichever prediction was returned by the most trees.
# This helps to avoid overfitting, a problem that occurs when a model is so tightly fitted
# to arbitrary correlations in the training set that it performs poorly on the testing set.


from sklearn.ensemble import RandomForestClassifier


# We will tune the Random Forest model to find the optimal parameter values.

# Parameters that will be tuned    
# n_estimators: represents the number of trees in the forest. Usually the higher 
# the number of trees the better to learn the data. However, by adding a lot of trees, 
# the training process can be slowed down considerably.
n_estimators = [5, 10, 20, 35, 50, 75, 100, 150]
# max_depth: represents the depth of each tree in the forest. The deeper the tree, 
# the more splits it has and it captures more information about the data. 
max_depth = [2, 6, 10, 15, 20, 25, 30]
# min_samples_split: represents the minimum number of samples required to split an internal node. 
# This number can vary between considering at least one sample at each node to considering all 
# of the samples at each node. When we increase this parameter, each tree in the forest becomes 
# more constrained as it has to consider more samples at each node.
# vary the parameter from 10% to 100% of the samples
min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True) 
# min_samples_leaf: represents the minimum number of samples required to be at a leaf node. 
min_samples_leaf = np.linspace(0.1, 0.5, 5, endpoint=True)
# max_features: represents the number of features to consider when looking for the best split.
max_features = ['sqrt','log2',0.2,0.5,0.8]
# criterion: Gini is intended for continuous attributes and Entropy for attributes that occur 
# in classes. Gini tends to find the largest class, while Entropy tends to find groups of classes 
# that make up ~50% of the data. Gini minimizes misclassification. 
# Entropy may be a little slower to compute because it makes use of the logarithm.
criterion = ['gini', 'entropy']


random_forest = RandomForestClassifier(random_state=1)

for idx,predictor in enumerate(predictors):   

    x_train = train_set_new.loc[:,predictor]
    y_train = train_set_new.loc[:,'Loan_Status']
    x_test = test_set_new.loc[:,predictor]
    
    
    param_grid = {'n_estimators': n_estimators, 'max_depth': max_depth,  
              'min_samples_split': min_samples_splits, 'min_samples_leaf': min_samples_leaf, 
              'max_features': max_features, 'criterion': criterion}

    random_forest_cv = GridSearchCV(random_forest, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs = n_jobs, verbose=150)
    # Set the verbose parameter in GridSearchCV to a positive number 
    # (the greater the number the more detail you will get) to print out progress
    random_forest_cv.fit(x_train, y_train)

    print("Best parameters found on training set:")
    print(random_forest_cv.best_params_)
    
    random_forest_optimised = random_forest_cv.best_estimator_
    random_forest_optimised.fit(x_train, y_train)
    
    
    # Plot the relative variable importance

    features = pd.DataFrame()
    features['Feature'] = x_train.columns
    features['Importance'] = random_forest_optimised.feature_importances_
    features.sort_values(by=['Importance'], ascending=True, inplace=True)
    features.set_index('Feature', inplace=True)
    features.plot(kind='barh', figsize=(10, 10), title = 'Feature Importance')
    plt.legend(loc='upper right', bbox_to_anchor=(0.9, 0.85),
           bbox_transform=plt.gcf().transFigure)
    
    
    # Performance on the training set
    
    scores = cross_val_score(random_forest_optimised, x_train, y_train, cv=shuffle_validator, scoring='accuracy')
    # Take the mean of the scores
    print("Model %d - Accuracy on Training Set: %0.4f (+/- %0.2f)" % (idx+1, scores.mean(), scores.std()))
    
    # Performance on the testing set
    y_pred = random_forest_optimised.predict(x_test)
    
    # Save the solution to a dataframe with two columns: Loan_ID and Loan_Status (prediction)
    filename="r_forest%d.csv" % (idx+1)
    my_solution  = pd.DataFrame({
            "Loan_ID": test_set_new.loc[:,'Loan_ID'],
            "Loan_Status": y_pred
            })
    my_solution.to_csv(filename, index=False)
 
    
    
    
    
# Summary for Random Forest - Accuracy  
    
#                                                    
# 1. Predictors: Credit_History, Property_Area, Coapplicant,
#    Loan_by_TotalIncome, Married, Education
#    Best parameters found on training set:
#    {'criterion': 'gini', 'max_depth': 2, 'max_features': 0.5, 'min_samples_leaf': 0.1, 
#    'min_samples_split': 0.6, 'n_estimators': 50}
#    Model 1 - Accuracy on Training Set: 0.7720 (+/- 0.05)
#    Accuracy on the public leaderboard: 0.7916666666666666
#
# 2. Predictors: Credit_History, Property_Area, Coapplicant,
#    Loan_by_TotalIncome, Married, Education, Loan_Amount_Term,
#    DFamilySize, LogTotalIncome, Dependents
#    Best parameters found on training set:    
#    {'criterion': 'gini', 'max_depth': 2, 'max_features': 0.5, 'min_samples_leaf': 0.1, 
#    'min_samples_split': 0.6, 'n_estimators': 50}
#    Model 2 - Accuracy on Training Set: 0.7557 (+/- 0.06)
#    Accuracy on the public leaderboard: 0.7777777777777778
#  
# 3. Predictors: Credit_History, Property_Area, Coapplicant,
#    Loan_by_TotalIncome, Married, Education, Loan_Amount_Term,
#    DFamilySize, LogTotalIncome, Dependents, LogLoan_by_FamilyIncome,
#    LogFamilyIncome, Gender, LogLoanPerMonth, LogLoanAmount,Self_Employed
#    Best parameters found on training set:    
#    {'criterion': 'gini', 'max_depth': 2, 'max_features': 0.5, 
#    'min_samples_leaf': 0.1, 'min_samples_split': 0.6, 'n_estimators': 50}
#    Model 3 - Accuracy on Training Set: 0.7785 (+/- 0.03)
#    Accuracy on the public leaderboard: 0.7847222222222222
    
    




# Extra-Tree method (Extremely Randomized trees)

# Extremely Randomized trees are very similar to Random Forests. Their main differences are that
# Extremely Randomized trees do not resample observations when building a tree (they do not perform bagging)
# and do not use the "best split". The resulting "forest" contains trees that are more variable, 
# but less correlated than the trees in a Random Forest, which means that Extremely Randomized trees 
# are better than Random Forest in term of variance. 


from sklearn.ensemble import ExtraTreesClassifier
    

 
# We will tune the Extra-Tree model to find the optimal parameter values.

# Parameters that will be tuned   
# n_estimators: represents the number of trees in the forest. Usually the higher 
# the number of trees the better to learn the data. However, by adding a lot of trees, 
# the training process can be slowed down considerably.
n_estimators = [5, 10, 20, 35, 50, 75, 100, 150]
# max_depth: represents the depth of each tree in the forest. The deeper the tree, 
# the more splits it has and it captures more information about the data. 
max_depth = [2, 6, 10, 15, 20, 25, 30]
# min_samples_split: represents the minimum number of samples required to split an internal node. 
# This number can vary between considering at least one sample at each node to considering all 
# of the samples at each node. When we increase this parameter, each tree in the forest becomes 
# more constrained as it has to consider more samples at each node.
# vary the parameter from 10% to 100% of the samples
min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True) 
# min_samples_leaf: represents the minimum number of samples required to be at a leaf node. 
min_samples_leaf = np.linspace(0.1, 0.5, 5, endpoint=True)
# max_features: represents the number of features to consider when looking for the best split.
max_features = ['sqrt','log2',0.2,0.5,0.8]
# criterion: Gini is intended for continuous attributes and Entropy for attributes that occur 
# in classes. Gini tends to find the largest class, while Entropy tends to find groups of classes 
# that make up ~50% of the data. Gini minimizes misclassification. 
# Entropy may be a little slower to compute because it makes use of the logarithm.
criterion = ['gini', 'entropy']


    
extra_tree = ExtraTreesClassifier(random_state=1)



for idx,predictor in enumerate(predictors):   

    x_train = train_set_new.loc[:,predictor]
    y_train = train_set_new.loc[:,'Loan_Status']
    x_test = test_set_new.loc[:,predictor]
    
    
    param_grid = {'n_estimators': n_estimators, 'max_depth': max_depth,  
              'min_samples_split': min_samples_splits, 'min_samples_leaf': min_samples_leaf, 
              'max_features': max_features, 'criterion': criterion}
    
    extra_tree_cv = GridSearchCV(extra_tree, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs = n_jobs, verbose=150)
    # Set the verbose parameter in GridSearchCV to a positive number 
    # (the greater the number the more detail you will get) to print out progress
    extra_tree_cv.fit(x_train, y_train)

    print("Best parameters found on training set:")
    print(extra_tree_cv.best_params_)
    
    extra_tree_optimised = extra_tree_cv.best_estimator_
    extra_tree_optimised.fit(x_train, y_train)
    
    
    # Plot the relative variable importance

    features = pd.DataFrame()
    features['Feature'] = x_train.columns
    features['Importance'] = extra_tree_optimised.feature_importances_
    features.sort_values(by=['Importance'], ascending=True, inplace=True)
    features.set_index('Feature', inplace=True)
    features.plot(kind='barh', figsize=(10, 10), title = 'Feature Importance')
    plt.legend(loc='upper right', bbox_to_anchor=(0.9, 0.85),
           bbox_transform=plt.gcf().transFigure)
    
    
    # Performance on the training set
    
    scores = cross_val_score(extra_tree_optimised, x_train, y_train, cv=shuffle_validator, scoring='accuracy')
    # Take the mean of the scores
    print("Model %d - Accuracy on Training Set: %0.4f (+/- %0.2f)" % (idx+1, scores.mean(), scores.std()))
    
    # Performance on the testing set
    y_pred = extra_tree_optimised.predict(x_test)
    
    # Save the solution to a dataframe with two columns: Loan_ID and Loan_Status (prediction)
    filename="extra_tree%d.csv" % (idx+1)
    my_solution  = pd.DataFrame({
            "Loan_ID": test_set_new.loc[:,'Loan_ID'],
            "Loan_Status": y_pred
            })
    my_solution.to_csv(filename, index=False)


# Summary for Extra-Tree - Accuracy  
    
#                                                    
# 1. Predictors: Credit_History, Property_Area, Coapplicant,
#    Loan_by_TotalIncome, Married, Education
#    Best parameters found on training set:
#    {'criterion': 'gini', 'max_depth': 2, 'max_features': 'sqrt', 'min_samples_leaf': 0.1, 
#     'min_samples_split': 0.1, 'n_estimators': 10}
#    Model 1 - Accuracy on Training Set: 0.8049 (+/- 0.02)
#    Accuracy on the public leaderboard: 0.7777777777777778
#
# 2. Predictors: Credit_History, Property_Area, Coapplicant,
#    Loan_by_TotalIncome, Married, Education, Loan_Amount_Term,
#    DFamilySize, LogTotalIncome, Dependents
#    Best parameters found on training set:    
#    {'criterion': 'gini', 'max_depth': 2, 'max_features': 'sqrt', 
#     'min_samples_leaf': 0.1, 'min_samples_split': 0.4, 'n_estimators': 150}
#    Model 2 - Accuracy on Training Set: 0.8016 (+/- 0.03)
#    Accuracy on the public leaderboard: 0.7777777777777778
#  
# 3. Predictors: Credit_History, Property_Area, Coapplicant,
#    Loan_by_TotalIncome, Married, Education, Loan_Amount_Term,
#    DFamilySize, LogTotalIncome, Dependents, LogLoan_by_FamilyIncome,
#    LogFamilyIncome, Gender, LogLoanPerMonth, LogLoanAmount,Self_Employed
#    Best parameters found on training set:    
#    {'criterion': 'gini', 'max_depth': 2, 'max_features': 'sqrt', 'min_samples_leaf': 0.1, 
#    'min_samples_split': 0.1, 'n_estimators': 10}
#    Model 3 - Accuracy on Training Set: 0.8053 (+/- 0.02)
#    Accuracy on the public leaderboard: 0.7777777777777778
    
    
    
 
#####      Gradient Boosting Machine (GBM)


# The Gradient Boosting classifier generates many weak prediction trees and combines or "boost" them 
# into a stronger model. 

from sklearn.ensemble import GradientBoostingClassifier



# We will tune the Gradient Boosting Machine to find the optimal parameter values.


# Parameters that will be tuned  
# learning_rate: learning rate shrinks the contribution of each tree by learning_rate.
# There is a trade-off between learning_rate and n_estimators.
learning_rate = [1, 0.5, 0.25, 0.1, 0.05, 0.01]
# n_estimators: represents the number of trees in the forest. Usually the higher 
# the number of trees the better to learn the data. However, by adding a lot of trees, 
# the training process can be slowed down considerably.
# Gradient boosting is fairly robust to over-fitting so a large number usually
# results in better performance.
n_estimators = [5, 10, 20, 35, 50, 75, 100, 150]
# loss: loss function to be optimized. 'deviance' refers to deviance (= logistic regression) for 
# classification with probabilistic outputs. For loss 'exponential' gradient boosting
# recovers the AdaBoost algorithm.
loss = ['deviance','exponential']
# max_depth: represents the depth of each tree in the forest. The deeper the tree, 
# the more splits it has and it captures more information about the data. 
max_depth = [2, 6, 10, 15, 20, 25, 30]
# min_samples_split: represents the minimum number of samples required to split an internal node. 
# This number can vary between considering at least one sample at each node to considering all 
# of the samples at each node. When we increase this parameter, each tree in the forest becomes 
# more constrained as it has to consider more samples at each node.
# vary the parameter from 10% to 100% of the samples
min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True) 
# min_samples_leaf: represents the minimum number of samples required to be at a leaf node. 
min_samples_leaf = np.linspace(0.1, 0.5, 5, endpoint=True)
# max_features: represents the number of features to consider when looking for the best split.
max_features = ['sqrt','log2',0.2,0.5,0.8]



gbm = GradientBoostingClassifier(random_state=1)



for idx,predictor in enumerate(predictors):   

    x_train = train_set_new.loc[:,predictor]
    y_train = train_set_new.loc[:,'Loan_Status']
    x_test = test_set_new.loc[:,predictor]
    
    
    param_grid = {'learning_rate': learning_rate, 'n_estimators': n_estimators, 'loss': loss, 
              'max_depth': max_depth, 'min_samples_split': min_samples_splits, 
              'min_samples_leaf': min_samples_leaf, 'max_features': max_features}
    
    gbm_cv = GridSearchCV(gbm, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs = n_jobs, verbose=150)
    # Set the verbose parameter in GridSearchCV to a positive number 
    # (the greater the number the more detail you will get) to print out progress
    gbm_cv.fit(x_train, y_train)

    print("Best parameters found on training set:")
    print(gbm_cv.best_params_)
    
    gbm_optimised = gbm_cv.best_estimator_
    gbm_optimised.fit(x_train, y_train)
    
    
    # Performance on the training set
    scores = cross_val_score(gbm_optimised, x_train, y_train, cv=shuffle_validator, scoring='accuracy')
    # Take the mean of the scores
    print("Model %d - Accuracy on Training Set: %0.4f (+/- %0.2f)" % (idx+1, scores.mean(), scores.std()))
    
    # Performance on the testing set
    y_pred = gbm_optimised.predict(x_test)
    
    # Save the solution to a dataframe with two columns: Loan_ID and Loan_Status (prediction)
    filename="gbm%d.csv" % (idx+1)
    my_solution  = pd.DataFrame({
            "Loan_ID": test_set_new.loc[:,'Loan_ID'],
            "Loan_Status": y_pred
            })
    my_solution.to_csv(filename, index=False)




# Summary for Gradient Boosting Classifier - Accuracy  
    
#                                                    
# 1. Predictors: Credit_History, Property_Area, Coapplicant,
#    Loan_by_TotalIncome, Married, Education
#    Best parameters found on training set:
#    {'learning_rate': 1, 'loss': 'exponential', 'max_depth': 6, 'max_features': 'sqrt', 
#    'min_samples_leaf': 0.1, 'min_samples_split': 0.7000000000000001, 'n_estimators': 20}
#    Model 1 - Accuracy on Training Set: 0.8037 (+/- 0.02)
#    Accuracy on the public leaderboard: 0.7777777777777778
#
# 2. Predictors: Credit_History, Property_Area, Coapplicant,
#    Loan_by_TotalIncome, Married, Education, Loan_Amount_Term,
#    DFamilySize, LogTotalIncome, Dependents
#    Best parameters found on training set:    
#    {'learning_rate': 0.5, 'loss': 'deviance', 'max_depth': 6, 
#    'max_features': 'log2', 'min_samples_leaf': 0.1, 'min_samples_split': 0.5, 'n_estimators': 10}
#    Model 2 - Accuracy on Training Set: 0.7988 (+/- 0.02)
#    Accuracy on the public leaderboard: 0.7777777777777778
#  
# 3. Predictors: Credit_History, Property_Area, Coapplicant,
#    Loan_by_TotalIncome, Married, Education, Loan_Amount_Term,
#    DFamilySize, LogTotalIncome, Dependents, LogLoan_by_FamilyIncome,
#    LogFamilyIncome, Gender, LogLoanPerMonth, LogLoanAmount,Self_Employed
#    Best parameters found on training set: 
#    {'learning_rate': 0.5, 'loss': 'deviance', 'max_depth': 6, 'max_features': 'sqrt', 
#    'min_samples_leaf': 0.1, 'min_samples_split': 0.1, 'n_estimators': 5}
#    Model 3 - Accuracy on Training Set: 0.7988 (+/- 0.02)
#    Accuracy on the public leaderboard: 0.7777777777777778
    
      


########   XGBoost (or eXtreme Gradient Boosting)  

# XGBoost is one of the implementations of Gradient Boosting concept.Its difference
# with the classical Gradient Boosting is that it uses a more regularized model 
# formalization to control over-fitting, which gives it better performance.


from xgboost import XGBClassifier

# We will tune the XGBoost to find the optimal parameter values.
# Parameters that will be tuned  
# learning_rate: learning rate shrinks the contribution of each tree by learning_rate.
# There is a trade-off between learning_rate and n_estimators.
learning_rate = [0.01, 0.02, 0.05, 0.1, 0.2]
# n_estimators: represents the number of trees in the forest. Usually the higher 
# the number of trees the better to learn the data. However, by adding a lot of trees, 
# the training process can be slowed down considerably.
# XGBoost is fairly robust to over-fitting so a large number usually
# results in better performance.
n_estimators = [100, 200, 300, 500, 1000]
# max_depth: represents the depth of each tree in the forest. The deeper the tree, 
# the more splits it has and it captures more information about the data. 
max_depth = [3,7,10]
# gamma: represents the minimum loss reduction required to make a further
# partition on a leaf node of the tree.
gamma = [0, 0.1, 0.2, 0.5, 1]
# reg_alpha : (xgb's alpha) L1 regularization term on weights
reg_alpha = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 1]
# min_child_weight : Minimum sum of instance weight (hessian) needed in a child.
min_child_weight = [1,3,5]


xgb = XGBClassifier(random_state=1)


for idx,predictor in enumerate(predictors):   

    x_train = train_set_new.loc[:,predictor]
    y_train = train_set_new.loc[:,'Loan_Status']
    x_test = test_set_new.loc[:,predictor]
    
    param_grid = {'learning_rate': learning_rate, 'n_estimators': n_estimators, 
              'max_depth': max_depth, 'gamma': gamma, 'reg_alpha': reg_alpha, 
              'min_child_weight': min_child_weight}
    
    xgb_cv = GridSearchCV(xgb, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs = n_jobs, verbose=150)
    # Set the verbose parameter in GridSearchCV to a positive number 
    # (the greater the number the more detail you will get) to print out progress
    xgb_cv.fit(x_train, y_train)

    print("Best parameters found on training set:")
    print(xgb_cv.best_params_)
    
    xgb_optimised = xgb_cv.best_estimator_
    xgb_optimised.fit(x_train, y_train)
    
    
    # Plot the relative variable importance

    features = pd.DataFrame()
    features['Feature'] = x_train.columns
    features['Importance'] = xgb_optimised.feature_importances_
    features.sort_values(by=['Importance'], ascending=True, inplace=True)
    features.set_index('Feature', inplace=True)
    features.plot(kind='barh', figsize=(10, 10), title = 'Feature Importance')
    plt.legend(loc='upper right', bbox_to_anchor=(0.9, 0.85),
           bbox_transform=plt.gcf().transFigure)
    
    
    # Performance on the training set
    scores = cross_val_score(xgb_optimised, x_train, y_train, cv=shuffle_validator, scoring='accuracy')
    # Take the mean of the scores
    print("Model %d - Accuracy on Training Set: %0.4f (+/- %0.2f)" % (idx+1, scores.mean(), scores.std()))
    
    # Performance on the testing set
    y_pred = xgb_optimised.predict(x_test)
    
    # Save the solution to a dataframe with two columns: Loan_ID and Loan_Status (prediction)
    filename="xgb%d.csv" % (idx+1)
    my_solution  = pd.DataFrame({
            "Loan_ID": test_set_new.loc[:,'Loan_ID'],
            "Loan_Status": y_pred
            })
    my_solution.to_csv(filename, index=False)






# Summary for XGBoost - Accuracy  
    
#                                                    
# 1. Predictors: Credit_History, Property_Area, Coapplicant,
#    Loan_by_TotalIncome, Married, Education
#    Best parameters found on training set:
#    {'gamma': 0, 'learning_rate': 0.01, 'max_depth': 7, 'min_child_weight': 3, 
#    'n_estimators': 100, 'reg_alpha': 1}
#    Model 1 - Accuracy on Training Set: 0.8159 (+/- 0.03)
#    Accuracy on the public leaderboard: 0.7847222222222222
#
# 2. Predictors: Credit_History, Property_Area, Coapplicant,
#    Loan_by_TotalIncome, Married, Education, Loan_Amount_Term,
#    DFamilySize, LogTotalIncome, Dependents
#    Best parameters found on training set:    
#    {'gamma': 0, 'learning_rate': 0.01, 'max_depth': 3, 'min_child_weight': 3, 
#    'n_estimators': 100, 'reg_alpha': 0}
#    Model 2 - Accuracy on Training Set: 0.8110 (+/- 0.03)
#    Accuracy on the public leaderboard: 0.7847222222222222
#  
# 3. Predictors: Credit_History, Property_Area, Coapplicant,
#    Loan_by_TotalIncome, Married, Education, Loan_Amount_Term,
#    DFamilySize, LogTotalIncome, Dependents, LogLoan_by_FamilyIncome,
#    LogFamilyIncome, Gender, LogLoanPerMonth, LogLoanAmount,Self_Employed
#    Best parameters found on training set: 
#    {'gamma': 0.2, 'learning_rate': 0.01, 'max_depth': 7, 'min_child_weight': 3, 
#    'n_estimators': 100, 'reg_alpha': 1}
#    Model 3 - Accuracy on Training Set: 0.8093 (+/- 0.03)
#    Accuracy on the public leaderboard: 0.7847222222222222
   
    
    
    

#####      Logistic Regression


# Logistic Regression is part of a larger class of algorithms known as Generalized Linear Model (GLM).
# Logistic regression is a classification algorithm used to assign observations to a discrete set of 
# classes. Unlike linear regression which outputs continuous number values, logistic regression 
# transforms its output using the logistic sigmoid function to return a probability value which can 
# then be mapped to two or more discrete classes.

from sklearn.linear_model import LogisticRegression    
    
    

# We will tune the Logistic Regression model to find the optimal parameter values.

# Parameters that will be tuned   
# penalty: Regularization is a way to avoid overfitting by penalizing high-valued regression coefficients.
# It can be used to train models that generalize better on unseen data, by preventing the algorithm 
# from overfitting the training dataset. 
# A regression model that uses L1 regularization technique is called Lasso Regression and a model 
# which uses L2 is called Ridge Regression.
# The key difference between these techniques is that Lasso shrinks the less important feature’s 
# coefficient to zero, thus, removing some feature completely. So, this works well for feature selection 
# in case we have a huge number of features.
# We will use Solver 'lbfgs' which supports only l2 penalties. 
# So we can't try to tune the penalty parameter here.
# penalty = ['l1','l2']
# C: C = 1/λ
# Lambda (λ) controls the trade-off between allowing the model to increase its complexity as much as 
# it wants while trying to keep it simple. For example, if λ is very low or 0, the model will have 
# enough power to increase its complexity (overfit) by assigning big values to the weights for 
# each parameter. If, in the other hand, we increase the value of λ, the model will tend to underfit, 
# as the model will become too simple.
# Parameter C will work the other way around. For small values of C, we increase the regularization 
# strength which will create simple models and thus underfit the data. For big values of C, 
# we reduce the power of regularization, which imples the model is allowed to increase its complexity, 
# and therefore, overfit the data.
C = [0.0001, 0.001, 0.01, 0.05, 0.09, 1, 2, 3, 4, 5, 10, 100]  


log_reg = LogisticRegression(solver ='lbfgs', random_state=1)


for idx,predictor in enumerate(predictors):   

    x_train = train_set_new.loc[:,predictor]
    y_train = train_set_new.loc[:,'Loan_Status']
    x_test = test_set_new.loc[:,predictor]
    
    
    param_grid = {'C': C}
    
    log_reg_cv = GridSearchCV(log_reg, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs = n_jobs, verbose=150)
    # Set the verbose parameter in GridSearchCV to a positive number 
    # (the greater the number the more detail you will get) to print out progress
    log_reg_cv.fit(x_train, y_train)

    print("Best parameters found on training set:")
    print(log_reg_cv.best_params_)
    
    log_reg_optimised = log_reg_cv.best_estimator_
    log_reg_optimised.fit(x_train, y_train)
    
    
    # Performance on the training set
    scores = cross_val_score(log_reg_optimised, x_train, y_train, cv=shuffle_validator, scoring='accuracy')
    # Take the mean of the scores
    print("Model %d - Accuracy on Training Set: %0.4f (+/- %0.2f)" % (idx+1, scores.mean(), scores.std()))
    
    # Performance on the testing set
    y_pred = log_reg_optimised.predict(x_test)
    
    # Save the solution to a dataframe with two columns: Loan_ID and Loan_Status (prediction)
    filename="log_reg%d.csv" % (idx+1)
    my_solution  = pd.DataFrame({
            "Loan_ID": test_set_new.loc[:,'Loan_ID'],
            "Loan_Status": y_pred
            })
    my_solution.to_csv(filename, index=False)





# Summary for Logistic Regression - Accuracy  
    
#                                                    
# 1. Predictors: Credit_History, Property_Area, Coapplicant,
#    Loan_by_TotalIncome, Married, Education
#    Best parameters found on training set:
#    {'C': 5}
#    Model 1 - Accuracy on Training Set: 0.8045 (+/- 0.02)
#    Accuracy on the public leaderboard: 0.7777777777777778
#
# 2. Predictors: Credit_History, Property_Area, Coapplicant,
#    Loan_by_TotalIncome, Married, Education, Loan_Amount_Term,
#    DFamilySize, LogTotalIncome, Dependents
#    Best parameters found on training set:    
#    {'C': 2}
#    Model 2 - Accuracy on Training Set: 0.7988 (+/- 0.02)
#    Accuracy on the public leaderboard: 0.7847222222222222
#  
# 3. Predictors: Credit_History, Property_Area, Coapplicant,
#    Loan_by_TotalIncome, Married, Education, Loan_Amount_Term,
#    DFamilySize, LogTotalIncome, Dependents, LogLoan_by_FamilyIncome,
#    LogFamilyIncome, Gender, LogLoanPerMonth, LogLoanAmount,Self_Employed
#    Best parameters found on training set:    
#    {'C': 2}
#    Model 3 - Accuracy on Training Set: 0.7976 (+/- 0.02)
#    Accuracy on the public leaderboard: 0.7777777777777778
    
    




#####      Support Vector Machines (SVM)

# The SVM classifier maps the input space into a new space by a kernel transformation, and then 
# finds the optimal separating hyperplane and the margin of separations in that space. In other words, 
# given labeled training data (supervised learning), the algorithm outputs an optimal hyperplane which 
# categorizes new observations


from sklearn.svm import SVC


# Since SVM tuning can be very slow, we will try using some parameters manually. 

# We will tune the SVM to find the optimal parameter values.
# Parameters that will be tuned  
# kernel: represents the type of hyperplane used to separate the data. 
# 'rbf' and 'poly' uses a non linear hyper-plane
#kernel = ['rbf', 'poly']
# gamma: is a parameter for non linear hyperplanes. The higher the gamma value it tries to exactly 
# fit the training data set
#gamma = [0.01, 0.1, 1, 10, 100, 'scale']
#C: is the penalty parameter of the error term. It controls the trade off between 
# smooth decision boundary and classifying the training points correctly.
#C = [0.01, 0.1, 1, 10, 100, 1000]
#degree: Degree of the polynomial kernel function ('poly').Ignored by all other kernels.
#degree = [3,4,10]



# Scaling numerical variables
categorical_feature_mask = train_set_new.dtypes == object
# Get list of non-categorical column names
non_categorical_columns = train_set_new.columns[~categorical_feature_mask].tolist()


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_set_new1=train_set_new.copy()
scale = scaler.fit(train_set_new1.loc[:,non_categorical_columns].copy())
train_set_new1.loc[:,non_categorical_columns] = scale.transform(train_set_new1.loc[:,non_categorical_columns])

test_set_new1=test_set_new.copy()
scale = scaler.fit(test_set_new1.loc[:,non_categorical_columns].copy())
test_set_new1.loc[:,non_categorical_columns] = scale.transform(test_set_new1.loc[:,non_categorical_columns])


#######     Model 1


predictors =  ['Credit_History_0', 'Credit_History_1', 'Credit_History_2',
'Property_Area_0', 'Property_Area_1', 'Property_Area_2',   
'Coapplicant_0', 'Coapplicant_1', 'Loan_by_TotalIncome', 
'Married_0', 'Married_1', 'Education_0', 'Education_1']

## Kernel tuning

svm = SVC(random_state=1)

x_train = train_set_new1.loc[:,predictors]
y_train = train_set_new1.loc[:,'Loan_Status']
x_test = test_set_new1.loc[:,predictors]
    
kernel = ['rbf', 'poly']
param_grid = {'kernel': kernel}
    
svm_cv = GridSearchCV(svm, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs = n_jobs, verbose=150)
# Set the verbose parameter in GridSearchCV to a positive number 
# (the greater the number the more detail you will get) to print out progress
svm_cv.fit(x_train, y_train)

print("Best parameters found on training set:")
print(svm_cv.best_params_)
# {'kernel': 'rbf'}
    
svm_optimised = svm_cv.best_estimator_
svm_optimised.fit(x_train, y_train)
    
    
# Performance on the training set
scores = cross_val_score(svm_optimised, x_train, y_train, cv=shuffle_validator, scoring='accuracy')
# Take the mean of the scores
print("Model %d - Accuracy on Training Set: %0.4f (+/- %0.2f)" % (1, scores.mean(), scores.std()))
# Model 1 - Accuracy on Training Set: 0.7996 (+/- 0.03)    



## gamma tuning

svm = SVC(kernel = 'rbf',random_state=1)

x_train = train_set_new1.loc[:,predictors]
y_train = train_set_new1.loc[:,'Loan_Status']
x_test = test_set_new1.loc[:,predictors]
    
gamma = [0.01, 0.1, 1, 10, 100, 'scale']
param_grid = {'gamma': gamma}
    
svm_cv = GridSearchCV(svm, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs = n_jobs, verbose=150)
# Set the verbose parameter in GridSearchCV to a positive number 
# (the greater the number the more detail you will get) to print out progress
svm_cv.fit(x_train, y_train)

print("Best parameters found on training set:")
print(svm_cv.best_params_)
# {'gamma': 0.01}
    
svm_optimised = svm_cv.best_estimator_
svm_optimised.fit(x_train, y_train)
    
    
# Performance on the training set
scores = cross_val_score(svm_optimised, x_train, y_train, cv=shuffle_validator, scoring='accuracy')
# Take the mean of the scores
print("Model %d - Accuracy on Training Set: %0.4f (+/- %0.2f)" % (1, scores.mean(), scores.std()))
# Model 1 - Accuracy on Training Set: 0.8053 (+/- 0.02)



## C tuning

svm = SVC(kernel = 'rbf', gamma= 0.01, random_state=1)

x_train = train_set_new1.loc[:,predictors]
y_train = train_set_new1.loc[:,'Loan_Status']
x_test = test_set_new1.loc[:,predictors]
    
C = [0.01, 0.1, 1, 10, 100, 1000]
param_grid = {'C': C}
    
svm_cv = GridSearchCV(svm, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs = n_jobs, verbose=150)
# Set the verbose parameter in GridSearchCV to a positive number 
# (the greater the number the more detail you will get) to print out progress
svm_cv.fit(x_train, y_train)

print("Best parameters found on training set:")
print(svm_cv.best_params_)
# {'C': 1}
    
svm_optimised = svm_cv.best_estimator_
svm_optimised.fit(x_train, y_train)
    
    
# Performance on the training set
scores = cross_val_score(svm_optimised, x_train, y_train, cv=shuffle_validator, scoring='accuracy')
# Take the mean of the scores
print("Model %d - Accuracy on Training Set: %0.4f (+/- %0.2f)" % (1, scores.mean(), scores.std()))
# Model 1 - Accuracy on Training Set: 0.8053 (+/- 0.02)


# Performance on the testing set
y_pred = svm_optimised.predict(x_test)
    
# Save the solution to a dataframe with two columns: Loan_ID and Loan_Status (prediction)
filename="svm%d.csv" % (1)
my_solution  = pd.DataFrame({
        "Loan_ID": test_set_new.loc[:,'Loan_ID'],
        "Loan_Status": y_pred
        })
my_solution.to_csv(filename, index=False)





#######     Model 2


predictors =  ['Credit_History_0', 'Credit_History_1', 'Credit_History_2',
'Property_Area_0', 'Property_Area_1', 'Property_Area_2',   
'Coapplicant_0', 'Coapplicant_1', 'Loan_by_TotalIncome', 
'Married_0', 'Married_1', 'Education_0', 'Education_1','Loan_Amount_Term_0',
'Loan_Amount_Term_1', 'Loan_Amount_Term_2', 'Loan_Amount_Term_3', 'Loan_Amount_Term_4',
'DFamilySize_0', 'DFamilySize_1', 'DFamilySize_2', 'LogTotalIncome',
'Dependents_0', 'Dependents_1', 'Dependents_2']

## Kernel tuning

svm = SVC(random_state=1)

x_train = train_set_new1.loc[:,predictors]
y_train = train_set_new1.loc[:,'Loan_Status']
x_test = test_set_new1.loc[:,predictors]
    
kernel = ['rbf', 'poly']
param_grid = {'kernel': kernel}
    
svm_cv = GridSearchCV(svm, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs = n_jobs, verbose=150)
# Set the verbose parameter in GridSearchCV to a positive number 
# (the greater the number the more detail you will get) to print out progress
svm_cv.fit(x_train, y_train)

print("Best parameters found on training set:")
print(svm_cv.best_params_)
# {'kernel': 'rbf'}
    
svm_optimised = svm_cv.best_estimator_
svm_optimised.fit(x_train, y_train)
    
    
# Performance on the training set
scores = cross_val_score(svm_optimised, x_train, y_train, cv=shuffle_validator, scoring='accuracy')
# Take the mean of the scores
print("Model %d - Accuracy on Training Set: %0.4f (+/- %0.2f)" % (2, scores.mean(), scores.std()))
# Model 2 - Accuracy on Training Set: 0.7976 (+/- 0.03)



## gamma tuning

svm = SVC(kernel = 'rbf',random_state=1)

x_train = train_set_new1.loc[:,predictors]
y_train = train_set_new1.loc[:,'Loan_Status']
x_test = test_set_new1.loc[:,predictors]
    
gamma = [0.01, 0.1, 1, 10, 100, 'scale']
param_grid = {'gamma': gamma}
    
svm_cv = GridSearchCV(svm, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs = n_jobs, verbose=150)
# Set the verbose parameter in GridSearchCV to a positive number 
# (the greater the number the more detail you will get) to print out progress
svm_cv.fit(x_train, y_train)

print("Best parameters found on training set:")
print(svm_cv.best_params_)
# {'gamma': 0.01}
    
svm_optimised = svm_cv.best_estimator_
svm_optimised.fit(x_train, y_train)
    
    
# Performance on the training set
scores = cross_val_score(svm_optimised, x_train, y_train, cv=shuffle_validator, scoring='accuracy')
# Take the mean of the scores
print("Model %d - Accuracy on Training Set: %0.4f (+/- %0.2f)" % (2, scores.mean(), scores.std()))
# Model 2 - Accuracy on Training Set: 0.8045 (+/- 0.02)



## C tuning

svm = SVC(kernel = 'rbf', gamma= 0.01, random_state=1)

x_train = train_set_new1.loc[:,predictors]
y_train = train_set_new1.loc[:,'Loan_Status']
x_test = test_set_new1.loc[:,predictors]
    
C = [0.01, 0.1, 1, 10, 100, 1000]
param_grid = {'C': C}
    
svm_cv = GridSearchCV(svm, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs = n_jobs, verbose=150)
# Set the verbose parameter in GridSearchCV to a positive number 
# (the greater the number the more detail you will get) to print out progress
svm_cv.fit(x_train, y_train)

print("Best parameters found on training set:")
print(svm_cv.best_params_)
# {'C': 1}
    
svm_optimised = svm_cv.best_estimator_
svm_optimised.fit(x_train, y_train)
    
    
# Performance on the training set
scores = cross_val_score(svm_optimised, x_train, y_train, cv=shuffle_validator, scoring='accuracy')
# Take the mean of the scores
print("Model %d - Accuracy on Training Set: %0.4f (+/- %0.2f)" % (1, scores.mean(), scores.std()))
# Model 2 - Accuracy on Training Set: 0.8045 (+/- 0.02)


# Performance on the testing set
y_pred = svm_optimised.predict(x_test)
    
# Save the solution to a dataframe with two columns: Loan_ID and Loan_Status (prediction)
filename="svm%d.csv" % (2)
my_solution  = pd.DataFrame({
        "Loan_ID": test_set_new.loc[:,'Loan_ID'],
        "Loan_Status": y_pred
        })
my_solution.to_csv(filename, index=False)





#######     Model 3


predictors =  ['Credit_History_0', 'Credit_History_1', 'Credit_History_2',
'Property_Area_0', 'Property_Area_1', 'Property_Area_2',   
'Coapplicant_0', 'Coapplicant_1', 'Loan_by_TotalIncome', 
'Married_0', 'Married_1', 'Education_0', 'Education_1','Loan_Amount_Term_0',
'Loan_Amount_Term_1', 'Loan_Amount_Term_2', 'Loan_Amount_Term_3', 'Loan_Amount_Term_4',
'DFamilySize_0', 'DFamilySize_1', 'DFamilySize_2', 'LogTotalIncome',
'Dependents_0', 'Dependents_1', 'Dependents_2', 'LogLoan_by_FamilyIncome',
'LogFamilyIncome', 'Gender_0', 'Gender_1', 'LogLoanPerMonth',
'LogLoanAmount',  'Self_Employed_0', 'Self_Employed_1']

## Kernel tuning

svm = SVC(random_state=1)

x_train = train_set_new1.loc[:,predictors]
y_train = train_set_new1.loc[:,'Loan_Status']
x_test = test_set_new1.loc[:,predictors]
    
kernel = ['rbf', 'poly']
param_grid = {'kernel': kernel}
    
svm_cv = GridSearchCV(svm, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs = n_jobs, verbose=150)
# Set the verbose parameter in GridSearchCV to a positive number 
# (the greater the number the more detail you will get) to print out progress
svm_cv.fit(x_train, y_train)

print("Best parameters found on training set:")
print(svm_cv.best_params_)
# {'kernel': 'rbf'}
    
svm_optimised = svm_cv.best_estimator_
svm_optimised.fit(x_train, y_train)
    
    
# Performance on the training set
scores = cross_val_score(svm_optimised, x_train, y_train, cv=shuffle_validator, scoring='accuracy')
# Take the mean of the scores
print("Model %d - Accuracy on Training Set: %0.4f (+/- %0.2f)" % (3, scores.mean(), scores.std()))
# Model 3 - Accuracy on Training Set: 0.8012 (+/- 0.02)



## gamma tuning

svm = SVC(kernel = 'rbf',random_state=1)

x_train = train_set_new1.loc[:,predictors]
y_train = train_set_new1.loc[:,'Loan_Status']
x_test = test_set_new1.loc[:,predictors]
    
gamma = [0.01, 0.1, 1, 10, 100, 'scale']
param_grid = {'gamma': gamma}
    
svm_cv = GridSearchCV(svm, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs = n_jobs, verbose=150)
# Set the verbose parameter in GridSearchCV to a positive number 
# (the greater the number the more detail you will get) to print out progress
svm_cv.fit(x_train, y_train)

print("Best parameters found on training set:")
print(svm_cv.best_params_)
# {'gamma': 0.01}
    
svm_optimised = svm_cv.best_estimator_
svm_optimised.fit(x_train, y_train)
    
    
# Performance on the training set
scores = cross_val_score(svm_optimised, x_train, y_train, cv=shuffle_validator, scoring='accuracy')
# Take the mean of the scores
print("Model %d - Accuracy on Training Set: %0.4f (+/- %0.2f)" % (3, scores.mean(), scores.std()))
# Model 3 - Accuracy on Training Set: 0.8045 (+/- 0.02)



## C tuning

svm = SVC(kernel = 'rbf', gamma= 0.01, random_state=1)

x_train = train_set_new1.loc[:,predictors]
y_train = train_set_new1.loc[:,'Loan_Status']
x_test = test_set_new1.loc[:,predictors]
    
C = [0.01, 0.1, 1, 10, 100, 1000]
param_grid = {'C': C}
    
svm_cv = GridSearchCV(svm, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs = n_jobs, verbose=150)
# Set the verbose parameter in GridSearchCV to a positive number 
# (the greater the number the more detail you will get) to print out progress
svm_cv.fit(x_train, y_train)

print("Best parameters found on training set:")
print(svm_cv.best_params_)
# {'C': 1}
    
svm_optimised = svm_cv.best_estimator_
svm_optimised.fit(x_train, y_train)
    
    
# Performance on the training set
scores = cross_val_score(svm_optimised, x_train, y_train, cv=shuffle_validator, scoring='accuracy')
# Take the mean of the scores
print("Model %d - Accuracy on Training Set: %0.4f (+/- %0.2f)" % (3, scores.mean(), scores.std()))
# Model 3 - Accuracy on Training Set: 0.8045 (+/- 0.02)


# Performance on the testing set
y_pred = svm_optimised.predict(x_test)
    
# Save the solution to a dataframe with two columns: Loan_ID and Loan_Status (prediction)
filename="svm%d.csv" % (3)
my_solution  = pd.DataFrame({
        "Loan_ID": test_set_new.loc[:,'Loan_ID'],
        "Loan_Status": y_pred
        })
my_solution.to_csv(filename, index=False)




# Summary for SVM Classifier - Accuracy  
    
#                                                    
# 1. Predictors: Credit_History, Property_Area, Coapplicant,
#    Loan_by_TotalIncome, Married, Education
#    Best parameters found on training set:
#    {'kernel': 'rbf', 'gamma': 0.01, 'C': 1}
#    Model 1 - Accuracy on Training Set: 0.8053 (+/- 0.02)
#    Accuracy on the public leaderboard: 0.7777777777777778
#
# 2. Predictors: Credit_History, Property_Area, Coapplicant,
#    Loan_by_TotalIncome, Married, Education, Loan_Amount_Term,
#    DFamilySize, LogTotalIncome, Dependents
#    Best parameters found on training set:    
#    {'kernel': 'rbf', 'gamma': 0.01, 'C': 1}
#    Model 2 - Accuracy on Training Set: 0.8045 (+/- 0.02)
#    Accuracy on the public leaderboard: 0.7777777777777778
#  
# 3. Predictors: Credit_History, Property_Area, Coapplicant,
#    Loan_by_TotalIncome, Married, Education, Loan_Amount_Term,
#    DFamilySize, LogTotalIncome, Dependents, LogLoan_by_FamilyIncome,
#    LogFamilyIncome, Gender, LogLoanPerMonth, LogLoanAmount,Self_Employed
#    Best parameters found on training set: 
#    {'kernel': 'rbf', 'gamma': 0.01, 'C': 1}
#    Model 3 - Accuracy on Training Set: 0.8045 (+/- 0.02)
#    Accuracy on the public leaderboard: 0.7777777777777778
    
    




#####      kNN (k-Nearest Neighbors)

# In the kth-Nearest Neighbor (kNN) algorithm, an observation is classified by a majority vote of its 
# k closest points (neighbors), determined by a predefined distance function. The observation is then 
# assigned to the class most common amongst its k nearest neighbors. The number “k” is the nearest 
# neighbors we wish to take vote from and is typically a small positive integer.

from sklearn.neighbors import KNeighborsClassifier

# We will tune the kNN to find the optimal parameter values.
# Parameters that will be tuned 
# n_neighbors : Number of nearest neighbors k
n_neighbors = np.arange(1, 31, 2)
# weights: weight function used in prediction. When using 'uniform', all points in each neighborhood
# are weighted equally.  When using 'distance', points are weighted by the inverse of their distance.
# In this case, closer neighbors of a query point will have a greater influence than neighbors 
# which are further away.
weights = ['uniform', 'distance']
# algorithm:  Algorithm used to compute the nearest neighbors:
algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
# metric: the distance metric to use for the tree
metric = ["minkowski", "euclidean", "cityblock"]




knn = KNeighborsClassifier()


for idx,predictor in enumerate(predictors):   

    x_train = train_set_new.loc[:,predictor]
    y_train = train_set_new.loc[:,'Loan_Status']
    x_test = test_set_new.loc[:,predictor]
    
    
    param_grid = {'n_neighbors': n_neighbors, 'weights': weights, 'algorithm': algorithm, 
              'metric': metric}
    
    knn_cv = GridSearchCV(knn, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs = n_jobs, verbose=150)
    # Set the verbose parameter in GridSearchCV to a positive number 
    # (the greater the number the more detail you will get) to print out progress
    knn_cv.fit(x_train, y_train)

    print("Best parameters found on training set:")
    print(knn_cv.best_params_)
    
    knn_optimised = knn_cv.best_estimator_
    knn_optimised.fit(x_train, y_train)
    
    
    # Performance on the training set
    scores = cross_val_score(knn_optimised, x_train, y_train, cv=shuffle_validator, scoring='accuracy')
    # Take the mean of the scores
    print("Model %d - Accuracy on Training Set: %0.4f (+/- %0.2f)" % (idx+1, scores.mean(), scores.std()))
    
    # Performance on the testing set
    y_pred = knn_optimised.predict(x_test)
    
    # Save the solution to a dataframe with two columns: Loan_ID and Loan_Status (prediction)
    filename="knn%d.csv" % (idx+1)
    my_solution  = pd.DataFrame({
            "Loan_ID": test_set_new.loc[:,'Loan_ID'],
            "Loan_Status": y_pred
            })
    my_solution.to_csv(filename, index=False)






# Summary for kNN - Accuracy  
    
#                                                    
# 1. Predictors: Credit_History, Property_Area, Coapplicant,
#    Loan_by_TotalIncome, Married, Education
#    Best parameters found on training set:
#    {'algorithm': 'auto', 'metric': 'minkowski', 'n_neighbors': 7, 'weights': 'uniform'}
#    Model 1 - Accuracy on Training Set: 0.7898 (+/- 0.03)
#    Accuracy on the public leaderboard: 0.7777777777777778
#
# 2. Predictors: Credit_History, Property_Area, Coapplicant,
#    Loan_by_TotalIncome, Married, Education, Loan_Amount_Term,
#    DFamilySize, LogTotalIncome, Dependents
#    Best parameters found on training set:    
#    {'algorithm': 'auto', 'metric': 'minkowski', 'n_neighbors': 15, 'weights': 'uniform'}
#    Model 2 - Accuracy on Training Set: 0.7720 (+/- 0.03)
#    Accuracy on the public leaderboard: 0.7638888888888888
#  
# 3. Predictors: Credit_History, Property_Area, Coapplicant,
#    Loan_by_TotalIncome, Married, Education, Loan_Amount_Term,
#    DFamilySize, LogTotalIncome, Dependents, LogLoan_by_FamilyIncome,
#    LogFamilyIncome, Gender, LogLoanPerMonth, LogLoanAmount,Self_Employed
#    Best parameters found on training set: 
#    {'algorithm': 'auto', 'metric': 'cityblock', 'n_neighbors': 13, 'weights': 'distance'}
#    Model 3 - Accuracy on Training Set: 0.7589 (+/- 0.03)
#    Accuracy on the public leaderboard: 0.7361111111111112
    
    
    


  



#####      Neural Network (NN)

# A Neural Network (NN) is a computational model inspired by the way biological neural networks in 
# the human brain process information. In other words, we can say that a neural network is a system 
# composed of many simple processing elements that are interconnected via weights and can acquire, 
# store, and utilize experiential knowledge.
# In more detail, it consists of units (neurons), arranged in layers, which convert an input into some 
# output. Each unit takes an input, applies a (often nonlinear) function to it and then passes the 
# output on to the next layer.

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from tensorflow import set_random_seed



# Create a NN with one input layer with x_train.shape[1] nodes which feeds into a hidden layer with 8 nodes 
# and an output layer which is used to predict a passenger's survival.
# The output layer has a sigmoid activation function, which is used to 'squash' all the outputs 
# to be between 0 and 1.

## Function to create model, required for KerasClassifier
def create_model(hidden_layer_sizes=[8], act='linear', opt='Adam', dr=0.0):
    
    # set random seed for reproducibility
    set_random_seed(42)
    
    # Initialising the NN
    model = Sequential()
    
    # create first hidden layer
    model.add(Dense(hidden_layer_sizes[0], input_dim=x_train.shape[1], activation=act)) 
    
    # create additional hidden layers
    for i in range(1,len(hidden_layer_sizes)):
        model.add(Dense(hidden_layer_sizes[i], activation=act))
    
    # add dropout (default is none)
    model.add(Dropout(dr))
    
    # create output layer
    model.add(Dense(1, activation='sigmoid'))  # output layer
    
    # Compiling the NN
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model

#model = create_model()
#print(model.summary())
   
    
#
## We will tune the NN to find the optimal parameter values.
## Parameters that will be tuned 
#
## define the grid search parameters
##size of data to grab at a time
#batch_size = [16, 32, 64]
##loops on dataset
#epochs = [50, 100]
## optimizer
#optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Nadam']
## hidden neurons
## # of neurons
## neurons = [[8],[10],[12],[18]]
#hidden_layer_sizes = [[8],[10],[10,5],[12,6],[12,8,4]]
## dropout
#dr = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
#
#
#
#from keras.wrappers.scikit_learn import KerasClassifier
#
#
## Since running GridSearch may be quite time/calculation consuming  
# # we will use GPU based tensorflow in order to speed-up calculation  
#import tensorflow as tf  
#from keras.backend.tensorflow_backend import set_session  
#
#config = tf.ConfigProto(log_device_placement=True)  
## Defining GPU usage limit to 75%  
#config.gpu_options.per_process_gpu_memory_fraction = 0.75  
## Inserting session configuration   
#ses=set_session(tf.Session(config=config))  
#
#
## create model
#model = KerasClassifier(build_fn=create_model, verbose=0)
#
#
#
#
#for idx,predictor in enumerate(predictors):   
#
#    x_train = train_set_new.loc[:,predictor]
#    y_train = train_set_new.loc[:,'Loan_Status']
#    x_test = test_set_new.loc[:,predictor]
#    
#
#    param_grid = {'batch_size': batch_size, 'epochs': epochs, 'opt': optimizer, 
#              'hidden_layer_sizes': hidden_layer_sizes, 'dr': dr}
#    
#    nn_cv = GridSearchCV(estimator=model,  param_grid=param_grid, cv=3, scoring='accuracy', error_score='raise', verbose=150)
#    # Set the verbose parameter in GridSearchCV to a positive number 
#    # (the greater the number the more detail you will get) to print out progress
#    nn_cv.fit(x_train, y_train)
#
#    print("Best parameters found on training set:")
#    print(nn_cv.best_params_)
#    
#    nn_optimised = nn_cv.best_estimator_
#    nn_optimised.fit(x_train, y_train)
#    
#    
#    # Performance on the training set
#    score = nn_optimised.evaluate(x_train, y_train, verbose=0)
#    print('Optimised Model %d - Training loss: %0.4f'  % (idx+1, score[0]))
#    print('Optimised Model %d - Training accuracy: %0.4f' % (idx+1, score[1]))
#    
#
#    # Performance on the testing set
#    y_pred1 = nn_optimised.predict(x_test)
#    y_pred = (y_pred1 > 0.5).astype(int).reshape(x_test.shape[0])
#    
#    
#    # Save the solution to a dataframe with two columns: Loan_ID and Loan_Status (prediction)
#    filename="nn%d.csv" % (idx+1)
#    my_solution  = pd.DataFrame({
#            "Loan_ID": test_set_new.loc[:,'Loan_ID'],
#            "Loan_Status": y_pred
#            })
#    my_solution.to_csv(filename, index=False)
#



# Since NN tuning can be very slow, we will try using some parameters manually. 
# We will tune the NN to find the optimal parameter values.

from keras.wrappers.scikit_learn import KerasClassifier


# Since running GridSearch may be quite time/calculation consuming  
 # we will use GPU based tensorflow in order to speed-up calculation  
import tensorflow as tf  
from keras.backend.tensorflow_backend import set_session  

config = tf.ConfigProto(log_device_placement=True)  
# Defining GPU usage limit to 75%  
config.gpu_options.per_process_gpu_memory_fraction = 0.75  
# Inserting session configuration   
ses=set_session(tf.Session(config=config))  




#######     Model 1


predictors =  ['Credit_History_0', 'Credit_History_1', 'Credit_History_2',
'Property_Area_0', 'Property_Area_1', 'Property_Area_2',   
'Coapplicant_0', 'Coapplicant_1', 'Loan_by_TotalIncome', 
'Married_0', 'Married_1', 'Education_0', 'Education_1']


# batch_size tuning

#size of data to grab at a time
batch_size = [16, 32, 64]


x_train = train_set_new.loc[:,predictors]
y_train = train_set_new.loc[:,'Loan_Status']
x_test = test_set_new.loc[:,predictors]
    

param_grid = {'batch_size': batch_size}

#create model
model = KerasClassifier(build_fn=create_model, verbose=0)
    
nn_cv = GridSearchCV(estimator=model,  param_grid=param_grid, cv=3, scoring='accuracy', error_score='raise', verbose=150)
# Set the verbose parameter in GridSearchCV to a positive number 
# (the greater the number the more detail you will get) to print out progress
nn_cv.fit(x_train, y_train)

print("Best parameters found on training set:")
print(nn_cv.best_params_)
# {'batch_size': 16}
  

# For the evaluation, we need to convert 'Y', 'N' of Loan_Status to 1,0, respectively
y_train2 = y_train.copy()
y_train2 = pd.Series(np.where(y_train2.values=='Y',1,0), y_train2.index)

# create model
model = create_model()

# Next we “fit” this classifier to our training set, enabling it to learn about
# how different factors affect the survivability of a passenger
model.fit(x_train, y_train2, batch_size=16, verbose=0)

# Performance on the training set

score = model.evaluate(x_train, y_train2, verbose=0)

print('Optimised Model %d - Training loss: %0.4f'  % (1, score[0]))
print('Optimised Model %d - Training accuracy: %0.4f' % (1, score[1]))

#Optimised Model 1 - Training loss: 0.4649
#Optimised Model 1 - Training accuracy: 0.8062



# epochs tuning

#loops on dataset
epochs = [50, 100]


param_grid = {'epochs': epochs}

#create model
model = KerasClassifier(build_fn=create_model, verbose=0)

nn_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', error_score='raise', verbose=150)
# Set the verbose parameter in GridSearchCV to a positive number 
# (the greater the number the more detail you will get) to print out progress
nn_cv.fit(x_train, y_train)

print("Best parameters found on training set:")
print(nn_cv.best_params_)
# {'epochs': 50}
   
# create model
model = create_model()
# Next we “fit” this classifier to our training set, enabling it to learn about
# how different factors affect the survivability of a passenger
model.fit(x_train, y_train2, batch_size=16, epochs=50, verbose=0)

# Performance on the training set

score = model.evaluate(x_train, y_train2, verbose=0)

print('Optimised Model %d - Training loss: %0.4f'  % (1, score[0]))
print('Optimised Model %d - Training accuracy: %0.4f' % (1, score[1]))

#Optimised Model 1 - Training loss: 0.4712
#Optimised Model 1 - Training accuracy: 0.7997


# optimizer tuning

# optimizer
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Nadam']


param_grid = {'opt': optimizer}

#create model
model = KerasClassifier(build_fn=create_model, verbose=0)

nn_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', error_score='raise', verbose=150)
# Set the verbose parameter in GridSearchCV to a positive number 
# (the greater the number the more detail you will get) to print out progress
nn_cv.fit(x_train, y_train)

print("Best parameters found on training set:")
print(nn_cv.best_params_)
# {'opt': 'Adadelta'}
   
# create model
model = create_model(opt ='Adadelta')
# Next we “fit” this classifier to our training set, enabling it to learn about
# how different factors affect the survivability of a passenger
model.fit(x_train, y_train2, batch_size=16, epochs=50, verbose=0)

# Performance on the training set

score = model.evaluate(x_train, y_train2, verbose=0)

print('Optimised Model %d - Training loss: %0.4f'  % (1, score[0]))
print('Optimised Model %d - Training accuracy: %0.4f' % (1, score[1]))

#Optimised Model 1 - Training loss: 0.4745
#Optimised Model 1 - Training accuracy: 0.7997


# hidden_layer_sizes tuning

# hidden neurons
# # of neurons
# neurons = [[8],[10],[12],[18]]
hidden_layer_sizes = [[8],[10],[10,5],[12,6],[12,8,4]]

param_grid = {'hidden_layer_sizes': hidden_layer_sizes}

#create model
model = KerasClassifier(build_fn=create_model, verbose=0)

nn_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', error_score='raise', verbose=150)
# Set the verbose parameter in GridSearchCV to a positive number 
# (the greater the number the more detail you will get) to print out progress
nn_cv.fit(x_train, y_train)

print("Best parameters found on training set:")
print(nn_cv.best_params_)
# {'hidden_layer_sizes': [12, 8, 4]}

# create model
model = create_model(opt ='Adadelta', hidden_layer_sizes = [12, 8, 4])
# Next we “fit” this classifier to our training set, enabling it to learn about
# how different factors affect the survivability of a passenger
model.fit(x_train, y_train2, batch_size=16, epochs=50, verbose=0)

# Performance on the training set

score = model.evaluate(x_train, y_train2, verbose=0)

print('Optimised Model %d - Training loss: %0.4f'  % (1, score[0]))
print('Optimised Model %d - Training accuracy: %0.4f' % (1, score[1]))

#Optimised Model 1 - Training loss: 0.4668
#Optimised Model 1 - Training accuracy: 0.8094


# dropout tuning
# dropout
dr = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]

param_grid = {'dr': dr}


#create model
model = KerasClassifier(build_fn=create_model, verbose=0)

nn_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', error_score='raise', verbose=150)
# Set the verbose parameter in GridSearchCV to a positive number 
# (the greater the number the more detail you will get) to print out progress
nn_cv.fit(x_train, y_train)

print("Best parameters found on training set:")
print(nn_cv.best_params_)
# {'dr': 0.01}


# create model
model = create_model(opt ='Adadelta', hidden_layer_sizes = [12, 8, 4], dr=0.01 )
# Next we “fit” this classifier to our training set, enabling it to learn about
# how different factors affect the survivability of a passenger
model.fit(x_train, y_train2, batch_size=16, epochs=50, verbose=0)

# Performance on the training set

score = model.evaluate(x_train, y_train2, verbose=0)

print('Optimised Model %d - Training loss: %0.4f'  % (1, score[0]))
print('Optimised Model %d - Training accuracy: %0.4f' % (1, score[1]))

# Optimised Model 1 - Training loss: 0.4620
# Optimised Model 1 - Training accuracy: 0.8094


### Now create the model using the best parameters

model = create_model(opt ='Adadelta', hidden_layer_sizes = [12, 8, 4], dr=0.01 )
print(model.summary())
#Layer (type)                 Output Shape              Param #   
#=================================================================
#dense_197 (Dense)            (None, 12)                168       
#_________________________________________________________________
#dense_198 (Dense)            (None, 8)                 104       
#_________________________________________________________________
#dense_199 (Dense)            (None, 4)                 36        
#_________________________________________________________________
#dropout_90 (Dropout)         (None, 4)                 0         
#_________________________________________________________________
#dense_200 (Dense)            (None, 1)                 5         
#=================================================================
#Total params: 313
#Trainable params: 313
#Non-trainable params: 0
#_________________________________________________________________
#None
    


# Next we “fit” this classifier to our training set, enabling it to learn about
# how different factors affect the survivability of a passenger
model.fit(x_train, y_train2, epochs=50, batch_size=16, validation_split=0.2, verbose=0)    
   

# Performance on the training set

score = model.evaluate(x_train, y_train2, verbose=0)
print('Training loss:', score[0])
print('Training accuracy:', score[1])
#Training loss: 0.4639054601860357
#Training accuracy: 0.8061889250814332


# Performance on the testing set

y_pred1 = model.predict(x_test)
y_pred = (y_pred1 > 0.5).astype(int).reshape(x_test.shape[0])
# For the evaluation, we need to convert 'Y', 'N' of Loan_Status to 1,0, respectively
y_pred2 = y_pred.copy()
y_pred2 =np.where(y_pred2==1,'Y','N')
# Save the solution to a dataframe with two columns: Loan_ID and Loan_Status (prediction)
my_solution  = pd.DataFrame({
        "Loan_ID": test_set_new.loc[:,'Loan_ID'],
        "Loan_Status": y_pred2
    })
my_solution.to_csv('nn1.csv', index=False)
# Kaggle score 0.7777777777777778





#######     Model 2


predictors =  ['Credit_History_0', 'Credit_History_1', 'Credit_History_2',
'Property_Area_0', 'Property_Area_1', 'Property_Area_2',   
'Coapplicant_0', 'Coapplicant_1', 'Loan_by_TotalIncome', 
'Married_0', 'Married_1', 'Education_0', 'Education_1','Loan_Amount_Term_0',
'Loan_Amount_Term_1', 'Loan_Amount_Term_2', 'Loan_Amount_Term_3', 'Loan_Amount_Term_4',
'DFamilySize_0', 'DFamilySize_1', 'DFamilySize_2', 'LogTotalIncome',
'Dependents_0', 'Dependents_1', 'Dependents_2']



# batch_size tuning

#size of data to grab at a time
batch_size = [16, 32, 64]


x_train = train_set_new.loc[:,predictors]
y_train = train_set_new.loc[:,'Loan_Status']
x_test = test_set_new.loc[:,predictors]
    

param_grid = {'batch_size': batch_size}

#create model
model = KerasClassifier(build_fn=create_model, verbose=0)
    
nn_cv = GridSearchCV(estimator=model,  param_grid=param_grid, cv=3, scoring='accuracy', error_score='raise', verbose=150)
# Set the verbose parameter in GridSearchCV to a positive number 
# (the greater the number the more detail you will get) to print out progress
nn_cv.fit(x_train, y_train)

print("Best parameters found on training set:")
print(nn_cv.best_params_)
# {'batch_size': 32}
  

# For the evaluation, we need to convert 'Y', 'N' of Loan_Status to 1,0, respectively
y_train2 = y_train.copy()
y_train2 = pd.Series(np.where(y_train2.values=='Y',1,0), y_train2.index)

# create model
model = create_model()

# Next we “fit” this classifier to our training set, enabling it to learn about
# how different factors affect the survivability of a passenger
model.fit(x_train, y_train2, batch_size=32, verbose=0)

# Performance on the training set

score = model.evaluate(x_train, y_train2, verbose=0)

print('Optimised Model %d - Training loss: %0.4f'  % (2, score[0]))
print('Optimised Model %d - Training accuracy: %0.4f' % (2, score[1]))

#Optimised Model 2 - Training loss: 0.6264
#Optimised Model 2 - Training accuracy: 0.6743



# epochs tuning

#loops on dataset
epochs = [50, 100]


param_grid = {'epochs': epochs}

#create model
model = KerasClassifier(build_fn=create_model, verbose=0)

nn_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', error_score='raise', verbose=150)
# Set the verbose parameter in GridSearchCV to a positive number 
# (the greater the number the more detail you will get) to print out progress
nn_cv.fit(x_train, y_train)

print("Best parameters found on training set:")
print(nn_cv.best_params_)
# {'epochs': 100}
   
# create model
model = create_model()
# Next we “fit” this classifier to our training set, enabling it to learn about
# how different factors affect the survivability of a passenger
model.fit(x_train, y_train2, batch_size=32, epochs=100, verbose=0)

# Performance on the training set

score = model.evaluate(x_train, y_train2, verbose=0)

print('Optimised Model %d - Training loss: %0.4f'  % (2, score[0]))
print('Optimised Model %d - Training accuracy: %0.4f' % (2, score[1]))

#Optimised Model 2 - Training loss: 0.4668
#Optimised Model 2 - Training accuracy: 0.7964


# optimizer tuning

# optimizer
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Nadam']


param_grid = {'opt': optimizer}

#create model
model = KerasClassifier(build_fn=create_model, verbose=0)

nn_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', error_score='raise', verbose=150)
# Set the verbose parameter in GridSearchCV to a positive number 
# (the greater the number the more detail you will get) to print out progress
nn_cv.fit(x_train, y_train)

print("Best parameters found on training set:")
print(nn_cv.best_params_)
# {'opt': 'Adagrad'}
   
# create model
model = create_model(opt ='Adagrad')
# Next we “fit” this classifier to our training set, enabling it to learn about
# how different factors affect the survivability of a passenger
model.fit(x_train, y_train2, batch_size=32, epochs=100, verbose=0)

# Performance on the training set

score = model.evaluate(x_train, y_train2, verbose=0)

print('Optimised Model %d - Training loss: %0.4f'  % (2, score[0]))
print('Optimised Model %d - Training accuracy: %0.4f' % (2, score[1]))

#Optimised Model 2 - Training loss: 0.4716
#Optimised Model 2 - Training accuracy: 0.8046


# hidden_layer_sizes tuning

# hidden neurons
# # of neurons
# neurons = [[8],[10],[12],[18]]
hidden_layer_sizes = [[8],[10],[10,5],[12,6],[12,8,4]]

param_grid = {'hidden_layer_sizes': hidden_layer_sizes}

#create model
model = KerasClassifier(build_fn=create_model, verbose=0)

nn_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', error_score='raise', verbose=150)
# Set the verbose parameter in GridSearchCV to a positive number 
# (the greater the number the more detail you will get) to print out progress
nn_cv.fit(x_train, y_train)

print("Best parameters found on training set:")
print(nn_cv.best_params_)
# {'hidden_layer_sizes': [12, 8, 4]}

# create model
model = create_model(opt ='Adagrad', hidden_layer_sizes = [12, 8, 4])
# Next we “fit” this classifier to our training set, enabling it to learn about
# how different factors affect the survivability of a passenger
model.fit(x_train, y_train2, batch_size=32, epochs=100, verbose=0)

# Performance on the training set

score = model.evaluate(x_train, y_train2, verbose=0)

print('Optimised Model %d - Training loss: %0.4f'  % (2, score[0]))
print('Optimised Model %d - Training accuracy: %0.4f' % (2, score[1]))

#Optimised Model 2 - Training loss: 0.4592
#Optimised Model 2 - Training accuracy: 0.8062


# dropout tuning
# dropout
dr = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]

param_grid = {'dr': dr}


#create model
model = KerasClassifier(build_fn=create_model, verbose=0)

nn_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', error_score='raise', verbose=150)
# Set the verbose parameter in GridSearchCV to a positive number 
# (the greater the number the more detail you will get) to print out progress
nn_cv.fit(x_train, y_train)

print("Best parameters found on training set:")
print(nn_cv.best_params_)
# {'dr': 0.01}


# create model
model = create_model(opt ='Adagrad', hidden_layer_sizes = [12, 8, 4], dr=0.01 )
# Next we “fit” this classifier to our training set, enabling it to learn about
# how different factors affect the survivability of a passenger
model.fit(x_train, y_train2, batch_size=32, epochs=100, verbose=0)

# Performance on the training set

score = model.evaluate(x_train, y_train2, verbose=0)

print('Optimised Model %d - Training loss: %0.4f'  % (2, score[0]))
print('Optimised Model %d - Training accuracy: %0.4f' % (2, score[1]))

# Optimised Model 2 - Training loss: 0.4732
# Optimised Model 2 - Training accuracy: 0.7915


### Now create the model using the best parameters

model = create_model(opt ='Adagrad', hidden_layer_sizes = [12, 8, 4], dr=0.01 )
print(model.summary())
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#dense_93 (Dense)             (None, 12)                312       
#_________________________________________________________________
#dense_94 (Dense)             (None, 8)                 104       
#_________________________________________________________________
#dense_95 (Dense)             (None, 4)                 36        
#_________________________________________________________________
#dropout_38 (Dropout)         (None, 4)                 0         
#_________________________________________________________________
#dense_96 (Dense)             (None, 1)                 5         
#=================================================================
#Total params: 457
#Trainable params: 457
#Non-trainable params: 0
#_________________________________________________________________
#None
    


# Next we “fit” this classifier to our training set, enabling it to learn about
# how different factors affect the survivability of a passenger
model.fit(x_train, y_train2, epochs=100, batch_size=32, validation_split=0.2, verbose=0)    
   

# Performance on the training set

score = model.evaluate(x_train, y_train2, verbose=0)
print('Training loss:', score[0])
print('Training accuracy:', score[1])
#Training loss: 0.47374772540132853
#Training accuracy: 0.7964169381107492


# Performance on the testing set

y_pred1 = model.predict(x_test)
y_pred = (y_pred1 > 0.5).astype(int).reshape(x_test.shape[0])
# For the evaluation, we need to convert 'Y', 'N' of Loan_Status to 1,0, respectively
y_pred2 = y_pred.copy()
y_pred2 =np.where(y_pred2==1,'Y','N')
# Save the solution to a dataframe with two columns: Loan_ID and Loan_Status (prediction)
my_solution  = pd.DataFrame({
        "Loan_ID": test_set_new.loc[:,'Loan_ID'],
        "Loan_Status": y_pred2
    })
my_solution.to_csv('nn2.csv', index=False)
# Kaggle score 0.7777777777777778





#######     Model 3


predictors =  ['Credit_History_0', 'Credit_History_1', 'Credit_History_2',
'Property_Area_0', 'Property_Area_1', 'Property_Area_2',   
'Coapplicant_0', 'Coapplicant_1', 'Loan_by_TotalIncome', 
'Married_0', 'Married_1', 'Education_0', 'Education_1','Loan_Amount_Term_0',
'Loan_Amount_Term_1', 'Loan_Amount_Term_2', 'Loan_Amount_Term_3', 'Loan_Amount_Term_4',
'DFamilySize_0', 'DFamilySize_1', 'DFamilySize_2', 'LogTotalIncome',
'Dependents_0', 'Dependents_1', 'Dependents_2', 'LogLoan_by_FamilyIncome',
'LogFamilyIncome', 'Gender_0', 'Gender_1', 'LogLoanPerMonth',
'LogLoanAmount',  'Self_Employed_0', 'Self_Employed_1']




# batch_size tuning

#size of data to grab at a time
batch_size = [16, 32, 64]


x_train = train_set_new.loc[:,predictors]
y_train = train_set_new.loc[:,'Loan_Status']
x_test = test_set_new.loc[:,predictors]
    

param_grid = {'batch_size': batch_size}

#create model
model = KerasClassifier(build_fn=create_model, verbose=0)
    
nn_cv = GridSearchCV(estimator=model,  param_grid=param_grid, cv=3, scoring='accuracy', error_score='raise', verbose=150)
# Set the verbose parameter in GridSearchCV to a positive number 
# (the greater the number the more detail you will get) to print out progress
nn_cv.fit(x_train, y_train)

print("Best parameters found on training set:")
print(nn_cv.best_params_)
# {'batch_size': 16}
  

# For the evaluation, we need to convert 'Y', 'N' of Loan_Status to 1,0, respectively
y_train2 = y_train.copy()
y_train2 = pd.Series(np.where(y_train2.values=='Y',1,0), y_train2.index)

# create model
model = create_model()

# Next we “fit” this classifier to our training set, enabling it to learn about
# how different factors affect the survivability of a passenger
model.fit(x_train, y_train2, batch_size=16, verbose=0)

# Performance on the training set

score = model.evaluate(x_train, y_train2, verbose=0)

print('Optimised Model %d - Training loss: %0.4f'  % (3, score[0]))
print('Optimised Model %d - Training accuracy: %0.4f' % (3, score[1]))

#Optimised Model 3 - Training loss: 0.6889
#Optimised Model 3 - Training accuracy: 0.6889



# epochs tuning

#loops on dataset
epochs = [50, 100]


param_grid = {'epochs': epochs}

#create model
model = KerasClassifier(build_fn=create_model, verbose=0)

nn_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', error_score='raise', verbose=150)
# Set the verbose parameter in GridSearchCV to a positive number 
# (the greater the number the more detail you will get) to print out progress
nn_cv.fit(x_train, y_train)

print("Best parameters found on training set:")
print(nn_cv.best_params_)
# {'epochs': 50}
   
# create model
model = create_model()
# Next we “fit” this classifier to our training set, enabling it to learn about
# how different factors affect the survivability of a passenger
model.fit(x_train, y_train2, batch_size=16, epochs=50, verbose=0)

# Performance on the training set

score = model.evaluate(x_train, y_train2, verbose=0)

print('Optimised Model %d - Training loss: %0.4f'  % (3, score[0]))
print('Optimised Model %d - Training accuracy: %0.4f' % (3, score[1]))

#Optimised Model 3 - Training loss: 0.4785
#Optimised Model 3 - Training accuracy: 0.7932


# optimizer tuning

# optimizer
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Nadam']


param_grid = {'opt': optimizer}

#create model
model = KerasClassifier(build_fn=create_model, verbose=0)

nn_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', error_score='raise', verbose=150)
# Set the verbose parameter in GridSearchCV to a positive number 
# (the greater the number the more detail you will get) to print out progress
nn_cv.fit(x_train, y_train)

print("Best parameters found on training set:")
print(nn_cv.best_params_)
# {'opt': 'RMSprop'}
   
# create model
model = create_model(opt ='RMSprop')
# Next we “fit” this classifier to our training set, enabling it to learn about
# how different factors affect the survivability of a passenger
model.fit(x_train, y_train2, batch_size=16, epochs=50, verbose=0)

# Performance on the training set

score = model.evaluate(x_train, y_train2, verbose=0)

print('Optimised Model %d - Training loss: %0.4f'  % (3, score[0]))
print('Optimised Model %d - Training accuracy: %0.4f' % (3, score[1]))

#Optimised Model 3 - Training loss: 0.4687
#Optimised Model 3 - Training accuracy: 0.8029


# hidden_layer_sizes tuning

# hidden neurons
# # of neurons
# neurons = [[8],[10],[12],[18]]
hidden_layer_sizes = [[8],[10],[10,5],[12,6],[12,8,4]]

param_grid = {'hidden_layer_sizes': hidden_layer_sizes}

#create model
model = KerasClassifier(build_fn=create_model, verbose=0)

nn_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', error_score='raise', verbose=150)
# Set the verbose parameter in GridSearchCV to a positive number 
# (the greater the number the more detail you will get) to print out progress
nn_cv.fit(x_train, y_train)

print("Best parameters found on training set:")
print(nn_cv.best_params_)
# {'hidden_layer_sizes': [10, 5]}

# create model
model = create_model(opt ='RMSprop', hidden_layer_sizes = [10, 5])
# Next we “fit” this classifier to our training set, enabling it to learn about
# how different factors affect the survivability of a passenger
model.fit(x_train, y_train2, batch_size=16, epochs=50, verbose=0)

# Performance on the training set

score = model.evaluate(x_train, y_train2, verbose=0)

print('Optimised Model %d - Training loss: %0.4f'  % (3, score[0]))
print('Optimised Model %d - Training accuracy: %0.4f' % (3, score[1]))

#Optimised Model 3 - Training loss: 0.4692
#Optimised Model 3 - Training accuracy: 0.8078


# dropout tuning
# dropout
dr = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]

param_grid = {'dr': dr}


#create model
model = KerasClassifier(build_fn=create_model, verbose=0)

nn_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', error_score='raise', verbose=150)
# Set the verbose parameter in GridSearchCV to a positive number 
# (the greater the number the more detail you will get) to print out progress
nn_cv.fit(x_train, y_train)

print("Best parameters found on training set:")
print(nn_cv.best_params_)
# {'dr': 0.5}


# create model
model = create_model(opt ='RMSprop', hidden_layer_sizes = [10, 5], dr=0.5 )
# Next we “fit” this classifier to our training set, enabling it to learn about
# how different factors affect the survivability of a passenger
model.fit(x_train, y_train2, batch_size=16, epochs=50, verbose=0)

# Performance on the training set

score = model.evaluate(x_train, y_train2, verbose=0)

print('Optimised Model %d - Training loss: %0.4f'  % (3, score[0]))
print('Optimised Model %d - Training accuracy: %0.4f' % (3, score[1]))

# Optimised Model 3 - Training loss: 0.4697
# Optimised Model 3 - Training accuracy: 0.8062


### Now create the model using the best parameters

model = create_model(opt ='RMSprop', hidden_layer_sizes = [10, 5], dr=0.5 )
print(model.summary())
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#dense_264 (Dense)            (None, 10)                340       
#_________________________________________________________________
#dense_265 (Dense)            (None, 5)                 55        
#_________________________________________________________________
#dropout_115 (Dropout)        (None, 5)                 0         
#_________________________________________________________________
#dense_266 (Dense)            (None, 1)                 6         
#=================================================================
#Total params: 401
#Trainable params: 401
#Non-trainable params: 0
#_________________________________________________________________
#None
    


# Next we “fit” this classifier to our training set, enabling it to learn about
# how different factors affect the survivability of a passenger
model.fit(x_train, y_train2, epochs=50, batch_size=16, validation_split=0.2, verbose=0)    
   

# Performance on the training set

score = model.evaluate(x_train, y_train2, verbose=0)
print('Training loss:', score[0])
print('Training accuracy:', score[1])
#Training loss: 0.4859440263591294
#Training accuracy: 0.7915309446254072


# Performance on the testing set

y_pred1 = model.predict(x_test)
y_pred = (y_pred1 > 0.5).astype(int).reshape(x_test.shape[0])
# For the evaluation, we need to convert 'Y', 'N' of Loan_Status to 1,0, respectively
y_pred2 = y_pred.copy()
y_pred2 =np.where(y_pred2==1,'Y','N')
# Save the solution to a dataframe with two columns: Loan_ID and Loan_Status (prediction)
my_solution  = pd.DataFrame({
        "Loan_ID": test_set_new.loc[:,'Loan_ID'],
        "Loan_Status": y_pred2
    })
my_solution.to_csv('nn3.csv', index=False)
# Kaggle score 0.7847222222222222



# Summary for NN - Accuracy  
    
#                                                    
# 1. Predictors: Credit_History, Property_Area, Coapplicant,
#    Loan_by_TotalIncome, Married, Education
#    Best parameters found on training set:
#    {'batch_size': 16, 'epochs': 50, 'opt': 'Adadelta', 'hidden_layer_sizes': [12, 8, 4], 'dr': 0.01}
#    Model 1 - Accuracy on Training Set: 0.8061889250814332
#    Accuracy on the public leaderboard: 0.7777777777777778
#
# 2. Predictors: Credit_History, Property_Area, Coapplicant,
#    Loan_by_TotalIncome, Married, Education, Loan_Amount_Term,
#    DFamilySize, LogTotalIncome, Dependents
#    Best parameters found on training set:    
#    {'batch_size': 32, 'epochs': 100, 'opt': 'Adagrad', 'hidden_layer_sizes': [12, 8, 4], 'dr': 0.01}
#    Model 2 - Accuracy on Training Set: 0.7964169381107492
#    Accuracy on the public leaderboard: 0.7777777777777778
#  
# 3. Predictors: Credit_History, Property_Area, Coapplicant,
#    Loan_by_TotalIncome, Married, Education, Loan_Amount_Term,
#    DFamilySize, LogTotalIncome, Dependents, LogLoan_by_FamilyIncome,
#    LogFamilyIncome, Gender, LogLoanPerMonth, LogLoanAmount,Self_Employed
#    Best parameters found on training set: 
#    {'batch_size': 16, 'epochs': 50, 'opt': 'RMSprop', 'hidden_layer_sizes': [10, 5], 'dr': 0.5}
#    Model 3 - Accuracy on Training Set: 0.8062
#    Accuracy on the public leaderboard: 0.7847222222222222




#####   Voting Classifier
    
# A "Voting" classifier can be used to apply multiple conceptually divergent classification models 
# to the same data set and will return the majority vote from all of the classifiers. 
    
    
import sklearn.ensemble as ske  


# We will use the predictors from model 1 with the Random Forest, XGBoost (or eXtreme Gradient Boosting)
# and Support Vector Machine (SVM) algorithms using their optimal paramaters
predictors =  ['Credit_History_0', 'Credit_History_1', 'Credit_History_2',
'Property_Area_0', 'Property_Area_1', 'Property_Area_2',   
'Coapplicant_0', 'Coapplicant_1', 'Loan_by_TotalIncome', 
'Married_0', 'Married_1', 'Education_0', 'Education_1']

x_train = train_set_new.loc[:,predictors]
y_train = train_set_new.loc[:,'Loan_Status']
x_test = test_set_new.loc[:,predictors]


# To create the Random Forest, we first initialize an instance of an untrained Random Forest classifier
random_forest = RandomForestClassifier(n_estimators=50, max_depth=2, max_features=0.5, 
                                       min_samples_split=0.6, min_samples_leaf=0.1, criterion='gini',
                                       random_state=1)

# To create the XGBoost Classifier, we first initialize an instance of an 
# untrained XGBoost Classifier

xgb = XGBClassifier(n_estimators=100, max_depth=7, gamma=0, learning_rate=0.01, min_child_weight=3,
                    reg_alpha=1, random_state=1) 


# To create the SVM Classifier, we first initialize an instance of an 
# untrained SVM Classifier
svm = SVC(C=1, gamma=0.01, kernel='rbf', random_state=1) 


voting = ske.VotingClassifier([('RandomForest', random_forest), ('XGB', xgb), ('SVM', svm)])



# Next we “fit” this classifier to our training set, enabling it to learn about
# how different factors affect the survivability of a passenger
voting.fit(x_train, y_train)


# Performance on the training set
scores = cross_val_score(voting, x_train, y_train, cv=shuffle_validator, scoring='accuracy')
# Take the mean of the scores
print("Accuracy on Training Set: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std()))
# Accuracy on Training Set: 0.8057 (+/- 0.02)


# Performance on the testing set

y_pred = voting.predict(x_test)
# Save the solution to a dataframe with two columns: Loan_ID and Loan_Status (prediction)
filename="voting.csv"
my_solution  = pd.DataFrame({
        "Loan_ID": test_set_new.loc[:,'Loan_ID'],
        "Loan_Status": y_pred
})
my_solution.to_csv(filename, index=False)
# Kaggle score 0.7777777777777778



    
# Best Score Per Algorithm:

# Random Forest 1: 0.7916666666666666    
# ---------------------------------------------
# Logistic Regression 2: 0.7847222222222222
# Neural Network (NN) 3: 0.7847222222222222
# XGBoost (or eXtreme Gradient Boosting) 1,2,3: 0.7847222222222222
# ---------------------------------------------
# Decision Trees 1,2,3: 0.7777777777777778
# Extra-Tree 1,2,3: 0.7777777777777778
# Gradient Boosting Classifier 1,2,3: 0.7777777777777778
# SVM 1,2,3: 0.7777777777777778
# kNN 1: 0.7777777777777778


# From the experimental section, we see that the most accurate model is the Random Forest
# when using the predictor variables 'Credit_History_0', 'Credit_History_1', 'Credit_History_2',
# 'Property_Area_0', 'Property_Area_1', 'Property_Area_2', 'Coapplicant_0', 'Coapplicant_1', 
# 'Loan_by_TotalIncome', 'Married_0', 'Married_1', 'Education_0' and 'Education_1'. 
# The accuracy of this model on the training set is 0.7720 (+/- 0.05) and on the 
# test set 0.7916666666666666. 



