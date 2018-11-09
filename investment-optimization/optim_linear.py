Created on Mon Nov  5 14:23:40 2018

@author: yemi
"""

import pandas as pd
import logging
from ortools.linear_solver import pywraplp

LOGGER = logging.getLogger(__file__)
result_status_to_str = {
    pywraplp.Solver.OPTIMAL: 'OPTIMAL',
    pywraplp.Solver.FEASIBLE: 'FEASIBLE',
    pywraplp.Solver.INFEASIBLE: 'INFEASIBLE',
    pywraplp.Solver.UNBOUNDED: 'UNBOUNDED',
    pywraplp.Solver.ABNORMAL: 'ABNORMAL',
    pywraplp.Solver.NOT_SOLVED: 'NOT_SOLVED'
}

prev_healthcare=pd.read_csv('raw_tables/preventive_healthcare.csv', sep=';', decimal=',')
prev_healthcare['id']=prev_healthcare.index
budget=10e6
costs={'Cardio':0, 'Diabete':0, 'Cancer':25000, 'Psychiatric':10000,
       'Neurology':0, 'DHO':0, 'Orthopedics':0}

class MIPSolver(object):
    """Class used to optimize spending in preventive healthcare

        Holds the data and the MILP model. The solution process consists of three phases:
        1. transformation to MILP model
        2. solution of MILP model
        3. transformation back into assignments

    The model is roughly:
        max     sum(number of actions * sum(expense reduction for each category) for each action)
        s.t.    sum(number of actions*cost for each action)
    """
    def __init__(self, costs:dict, prev_healthcare, budget:int):
        self.prev_healthcare=prev_healthcare.fillna(0)
        solver = pywraplp.Solver('SolveIntegerProblem', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
        infinity=solver.infinity()
        self.num_action={self.prev_healthcare.id[i]:solver.IntVar(0, infinity, 
                'nb#%i'%self.prev_healthcare.id[i]) for i in range (self.prev_healthcare.shape[0])}
        self.budget=budget
        self.costs=costs
        self.solver=solver
        self.impact_action={self.prev_healthcare.id[i]: sum(self.costs[cat]*(1-self.prev_healthcare.drop(['Action', 'Cost'], axis=1).iloc[i][cat]) 
        for cat in list(self.costs.keys())) for i in range (self.prev_healthcare.shape[0])}

        assert self.prev_healthcare.isnull().sum().sum()==0
        
        
    def _build_model(self):
        solver=self.solver
        # defining the cost and impact parameters
        cost_action={self.prev_healthcare.id[i]:self.prev_healthcare.Cost[i] for i in range (self.prev_healthcare.shape[0])}
        impact_action={self.prev_healthcare.id[i]: sum(self.costs[cat]*(1-self.prev_healthcare.drop(['Action', 'Cost'], axis=1).iloc[i][cat]) 
        for cat in list(self.costs.keys())) for i in range (self.prev_healthcare.shape[0])}
        
        #creating the constraint on healthcare budget
        constraint=solver.Constraint(0, self.budget)
        for action in self.num_action.keys():
            constraint.SetCoefficient(self.num_action[action], int(cost_action[action]))
        
        #defining the objective function
        obj = solver.Objective()
        for action in self.num_action.keys():
            obj.SetCoefficient(self.num_action[action], impact_action[action])
        obj.SetMaximization()
    
    def _solve_model(self):
        self.solver.SetTimeLimit(20000)
        self.solver.EnableOutput()
        result_status = self.solver.Solve()
        assert self.solver.VerifySolution(1e-7, True)
        if result_status not in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            raise SchedulingError('could not solve model: status=%i' % result_status)
        print('Number of variables =', self.solver.NumVariables())
        print('Number of constraints =', self.solver.NumConstraints())

        # The objective value of the solution.
        print('Optimal objective value = %d' % self.solver.Objective().Value())
        obj = self.solver.Objective()
        LOGGER.info('solver finished: status=%s objective=%f bound=%f time=%.3fms',
                    result_status_to_str[result_status], obj.Value(), obj.BestBound(), self.solver.wall_time())

    
    def _return_results(self):
        for i in self.num_action:
            print('%s = %d' % (self.num_action[i].name(), self.num_action[i].solution_value()))
    def run_model(self):
        self._build_model()
        self._solve_model()
        self._return_results()

optim=MIPSolver(costs, prev_healthcare, budget)     
optim.run_model()
