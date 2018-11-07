@author: yemi
"""
import numpy as np
import cvxpy as cv
import pandas as pd

prev_healthcare=pd.read_csv('raw_tables/preventive_healthcare.csv', sep=';', decimal=',')
prev_healthcare['id']=prev_healthcare.index
budget=10e6
expenses={'Cardio':1e6, 'Diabete':1e6, 'Cancer':1e6, 'Psychiatric':1e6,
       'Neurology':1e6, 'DHO':1e6, 'Orthopedics':1e6}

N=cv.Variable((1,5), integer=True)
obj=cv.Maximize(sum(
        sum(expenses[cat]*(1-prev_healthcare[cat].iloc[k])**N[0,k] for cat in expenses.keys()) 
        for k in range(prev_healthcare.shape[0])))
constraint=[sum(N[0,k]*prev_healthcare['Cost'].iloc[k] for k in 
                range(prev_healthcare.shape[0]))<=budget]
problem=cv.Problem(obj, constraint)
result=problem.solve()
print(N.value)
