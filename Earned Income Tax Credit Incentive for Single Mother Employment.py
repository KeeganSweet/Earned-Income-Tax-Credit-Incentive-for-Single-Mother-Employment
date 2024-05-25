# -*- coding: utf-8 -*-
"""
Created on Mon May 20 11:48:53 2024

@author: Keegan Sweet

References:

Eissa, N. & Liebman, J. (1996, May). "Labor Supply Response to the Earned Income Tax Credit". Quarterly Journal of Economics, 111(2), 605-637.

United States Congress. (1993). Omnibus Budget Reconciliation Act of 1993, Pub. L. No. 103-66, 107 Stat. 434 (1993).

Description:

This study examines the impact of government labor participation incentives via the earned income tax credit. An expansion was made to the tax code in 1993 to incentive work and reduce dependency on the welfare system for a specific demographic: single mothers. The tax credit constitutes a significant percentage of the income (outlined in 107 Stat. 433-434) if the mother is employed. This model examines regression effects when the x & y variables are binary; notably, the dependent variable: whether a single mother has a job or not (yes or no, 1 or 0). We will be using a logistic regression for this, since a linear regression does not fit this type of analysis. We'll first analyze the direct effects, then introduce omitted variables via the women's demographics, and finally, compare our results against a placebo test. Run each model sequentially to view results.

Last Updated: Last Updated: Sat, May 25 12:24EST, 2024

"""

import numpy as np
import pandas as pd
dataset=pd.read_stata("eitc.dta")

#Assigning dummy variables
#Creating dummy variable columns from the data with the needed parameters: single mothers after the 1993 policy expansion
dataset['post93']=np.where(dataset['year']>1993,1,0)
dataset['mom']=np.where(dataset['children']>0,1,0)
#Cross referencing to narrow down
dataset['mompost93']=dataset['mom']*dataset['post93']

#Isolate X & Y
Y=dataset.loc[:,'work'].values
X=dataset.loc[:,['post93','mom','mompost93']].values

#First logistic regression model
import statsmodels.api as sm

X=sm.add_constant(X)
model1=sm.Logit(Y,X).fit()
model1.summary(yname="work", 
               xname=("intercept", "post 93", "mom",
                      "mom post 93"),
               title= "Impact of Tax Credit on Employment")

#The logarithmic coefficient suggests a small positive relationship between employment levels and mothers after the policy implementation, with very acceptable significance. Additionally, the LLR p-value suggests our model is a good fit.


#Our second model adds 2 more independent variables to combat omitted variable bias.
X=dataset.loc[:,['post93','mom','mompost93',
                 'nonwhite','ed']].values

X=sm.add_constant(X)
model2=sm.Logit(Y,X).fit()
model2.summary(yname="work", 
               xname=("intercept", "post 93", "mom",
                      "mom post 93", "Hispanic or Black",
                      "Years of Education"),
               title= "Impact of Tax Credit on Employment")

#We see that the positive relationship for our target variable increases again, and still maintains a 99% significance level, lessening the probability that our first result was due to omitted variable bias.


#Placebo experiment
#Here we will analyze the year before the expansion. Note that the inequalities are exclusive.

#Prepare dummy variables for placebo test year before tax credit was implemented
dataset['post92']=np.where(dataset['year']>1992,1,0)
dataset['mompost92']=dataset['mom']*dataset['post92']

#Prepare placebo dataset
placebo_dataset=dataset[dataset['year']<1994]

#Isolate X & Y variables for placebo experiment
placebo_Y=placebo_dataset.loc[:,'work'].values
placebo_X=placebo_dataset.loc[:,['post92','mom','mompost92']].values

#Placebo logistic regression model
placebo_X=sm.add_constant(placebo_X)
placebo_model=sm.Logit(placebo_Y,placebo_X).fit()
placebo_model.summary(yname="work", 
               xname=("intercept", "post 92", "mom",
                      "mom post 92"),
               title= "Impact of Tax Credit on Employment - Placebo")

#The placebo p-value indicates an overwhelmingly insignificant relationship between the indepedent and dependent variables. Absent the policy change, we cannot meaningfully discern employment was affected by being a single mother or not.







