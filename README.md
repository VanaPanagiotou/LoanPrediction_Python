# Loan Approval Prediction in R language


## Table of Contents
* [Project Description](#Project-Description)
* [Libraries Required](#Libraries-Required)
* [Data](#Data)
* [Descriptive Statistics and Exploratory Analysis](#Descriptive-Statistics-and-Exploratory-Analysis)
* [Prediction](#Prediction-section)
* [Conclusion](#Conclusion)


## <a name="Project-Description"></a> Project Description

Dream Housing Finance company deals in all home loans. A customer first applies for home loan and then the company validates the customer eligibility for loan. The company wants to automate the loan eligibility process based on customer detail provided while filling an online application form. These details are gender, marital status, education, number of dependents, income, loan amount, credit history and others.

The goal of this project is to predict whether the company should approve the loan based on the applicant profile or not. This means that we have a binary classification problem. For more information see [Loan Prediction](https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/) presented by [Analytics Vidhya](https://datahack.analyticsvidhya.com/).


## <a name="Libraries-Required"></a> Libraries Required

Here are the required libraries to run the code properly:

```
numpy
pandas
seaborn
matplotlib
os
scipy
statsmodels
fancyimpute
sklearn
collections
category_encoders
multiprocessing
xgboost
keras
tensorflow
```



## <a name="Data"></a> Data

As with most Analytics Vidhya competitions, the Loan prediction data consists of a training set and a test set, both of which are .csv files:
* The training set contains data for a subset of applicants including the outcomes or “ground truth” (i.e. the “Loan_Status” response variable). We see that the training set has 614 observations (rows) and 13 variables. This set will be used to build the machine learning models. 

* The test set contains data for a subset of applicants without providing the ground truth, since the project’s objective is to predict these outcomes. The test set has 367 observations and 12 variables, since the “Loan_Status” variable is missing. This set will be used to see how well the developed model performs on unseen data. 

A description of the variables that are encountered in the loan prediction dataset is given in the next Table.




| Variable Name | Variable Description | Possible Values | Categorical/Numerical |
| --- | --- | --- | --- |
| `Loan_ID` | Unique Loan ID | 1, 2, …, 981 | Categorical |
| `Gender` | Gender of applicant | Female, Male | Categorical |
| `Married` | Marital Status | No, Yes | Categorical |
| `Dependents` | No. of dependents | 0, 1, 2, 3+ | Categorical |
| `Education` | Education level | Graduate, Not Graduate | Categorical |
| `Self_Employed` | Self employment status | No, Yes | Categorical |
| `ApplicantIncome` | Applicant’s income | 0 – 81000 | Numerical |
| `CoapplicantIncome` | Co-applicant’s income | 0 – 41667 | Numerical |
| `LoanAmount` | Loan amount in thousands | 9.0 – 700.0 | Numerical |
| `Loan_Amount_Term` | Term of loan in months | 6.0 – 480.0 | Numerical |
| `Credit_History` | Credit history meets guidelines | 0, 1 | Numerical |
| `Property_Area` | Area of property | Rural, Semiurban, Urban | Categorical |
| `Loan_Status` | Status of the loan | N= No, Y= Yes | Categorical |




## <a name="Descriptive-Statistics-and-Exploratory-Analysis"></a> Descriptive Statistics and Exploratory Analysis 

First, we are going to take a look at the data and examine their relationships. In addition, we have to find how many missing values we have and in which variables and replace them with sensible values.
We will also use some visualizations in order to better understand the relationships between the variables. Furthermore, we will make several graphs and computations to determine which transformation of variables can be added as new features.




## <a name="Prediction-section"></a> Prediction

Now, we will build and fit some models to the training set and we will compute their accuracy on the test set. For this purpose, we will build seven different models: **Decision Tree (DT)**, **Random Forest (RF)**, **Extra-Tree**, **Gradient Boosting Machine (GBM)**, **XGBoost**, **Logistic Regression (LR)**, **Support Vector Machine (SVM)**, **kNN (k-Nearest Neighbors)** and **Neural Network (NN)**.  We reset the random number seed before each run to ensure that each algorithm will be evaluated using the same data partitions. This means that the results will be directly comparable.

More details can be found within the project.



## <a name="Conclusion"></a> Conclusion

From the experimental section, we see that the most accurate model is the Random Forest when using the predictor variables Credit_History, Property_Area, Married, Education, Loan_by_TotalIncome and Coapplicant. The accuracy of this model on the training set is 0.7720 (+/- 0.05) and on the test set 0.79167. At the time of writing, this model is on the top 5% of this Analytics Vidhya competition.
