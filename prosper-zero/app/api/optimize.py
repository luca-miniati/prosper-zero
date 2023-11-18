from flask import Flask
from pulp import LpProblem, LpVariable, lpSum, LpMaximize
from pulp import GLPK_CMD, PYGLPK, CPLEX_CMD, CPLEX_PY, GUROBI, GUROBI_CMD, MOSEK, XPRESS, XPRESS_PY, PULP_CBC_CMD, COIN_CMD, COINMP_DLL, CHOCO_CMD, MIPCL_CMD, SCIP_CMD, HiGHS_CMD
import pulp as pl
import math

app = Flask(__name__)

def get_risk_buckets():
    return { 
        "AA" : 0.059844,
        "A" :  0.106875,
        "B" :  0.158608,
        "C" :  0.230475,
        "D" :  0.285390,
        "E" :  0.286738,
        "HR" : 0.341582,
    }

optimization_solvers = {
    'GLPK_CMD' : GLPK_CMD, 
    'PYGLPK' : PYGLPK, 
    'CPLEX_CMD' : CPLEX_CMD, 
    'CPLEX_PY' : CPLEX_PY, 
    'GUROBI' : GUROBI, 
    'GUROBI_CMD' : GUROBI_CMD, 
    'MOSEK' : MOSEK, 
    'XPRESS' : XPRESS, 
    'XPRESS_PY' : XPRESS_PY, 
    'PULP_CBC_CMD' : PULP_CBC_CMD, 
    'COIN_CMD' : COIN_CMD, 
    'COINMP_DLL' : COINMP_DLL, 
    'CHOCO_CMD' : CHOCO_CMD, 
    'MIPCL_CMD' : MIPCL_CMD, 
    'SCIP_CMD' : SCIP_CMD, 
    'HiGHS_CMD' : HiGHS_CMD,
}

def get_expected_portfolio_return(portfolio):
    num_loans = len(portfolio)
    weights = [1 / num_loans] * num_loans

    portfolio_return = sum(weight * loan['expected_return'] for weight, loan in zip(weights, portfolio))
    return portfolio_return

def get_sharpe_ratio(portfolio, risk_free_rate):
    risk_buckets = get_risk_buckets()
    
    size = len(portfolio)

    mean_return = (sum(loan['expected_return'] for loan in portfolio)) / size
    risk_free_return = (1 - risk_free_rate) * mean_return

    standard_dev = math.sqrt((sum((loan['expected_return'] - mean_return)**2 for loan in portfolio)) / size)

    expected_value = sum(loan['expected_return'] * risk_buckets[loan['prosper_rating']] for loan in portfolio)

    sharpe_ratio = (expected_value - risk_free_return) / standard_dev
    
    return sharpe_ratio

def optimize_portfolio(max_loans, listings, risk_free_rate, risk_weight=1, optimization_solver='PULP_CBC_CMD', portfolio=None):
    """
    Optimize a portfolio of loans using linear programming.

    Parameters:
    - max_loans (int): Maximum number of loans to select.
    - listings (list): List of loan dictionaries.
    - risk_free_rate (float): Risk-free rate of return.
    - risk_weight (float): Tunes how much you want to prioritize risk in your portfolio
    - optimization_solver (str): Solver for linear programming.
    - portfolio (list, optional): List of dictionaries representing an existing portfolio.

    Returns:
    - selected_loans (list): List of selected loan IDs.
    """
    risk_buckets = get_risk_buckets()

    model = LpProblem(name="Portfolio_Optimization", sense=LpMaximize)


    risk_adjusted_returns = [(loan["expected_return"] - (risk_weight * loan["expected_return"] * risk_buckets[loan["prosper_rating"]])) for loan in listings]
    
    loans = range(len(listings))
    x = LpVariable.dicts("loan", loans, cat="Binary")
    if not portfolio:
        model += lpSum(risk_adjusted_returns[i] * x[i] for i in loans)
        model += lpSum(x[i] for i in loans) <= max_loans
        model.solve(optimization_solvers[optimization_solver](msg=False))
    else:
        sharpe_ratio = get_sharpe_ratio(portfolio=portfolio, risk_free_rate=risk_free_rate)
        model += lpSum(risk_adjusted_returns[i] * x[i] for i in loans) + sharpe_ratio
        model += lpSum(x[i] for i in loans) <= max_loans
        model.solve(optimization_solvers[optimization_solver](msg=False))


    selected_loans = [listings[i]['id'] for i in loans if x[i].value() == 1]

    return selected_loans


@app.route('/optimize_portfolio_route', methods=['GET', 'POST'])
def optimize_portfolio_route(portfolio, max_loans, listings, optimization_solver, risk_free_rate):
    return optimize_portfolio(portfolio=portfolio, max_loans=max_loans, listings=listings, optimization_solver=optimization_solver, risk_free_rate=risk_free_rate)

# if __name__ == '__main__':
#     app.run()
'''
listings = [
    {'id' : "loan1", 'prosper_rating' : "AA", 'expected_return' : 0.05},
    {'id' : "loan2", 'prosper_rating' : "A", 'expected_return' : 0.06},
    {'id' : "loan3", 'prosper_rating' : "B", 'expected_return' : 0.07},
    {'id' : "loan4", 'prosper_rating' : "C", 'expected_return' : 0.08},
    {'id' : "loan5", 'prosper_rating' : "B", 'expected_return' : 0.065},
]

portfolio = [
    {'id' : "loan6", 'prosper_rating' : "AA", 'expected_return' : 0.02},
    {'id' : "loan7", 'prosper_rating' : "A", 'expected_return' : 0.07},
    {'id' : "loan8", 'prosper_rating' : "B", 'expected_return' : 0.075},
    {'id' : "loan9", 'prosper_rating' : "C", 'expected_return' : 0.06},
    {'id' : "loan10", 'prosper_rating' : "B", 'expected_return' : 0.055},
]
'''
listings = [
    {'id': 'loan1', 'prosper_rating': 'AA', 'expected_return': 0.05},
    {'id': 'loan2', 'prosper_rating': 'A', 'expected_return': 0.06},
    {'id': 'loan3', 'prosper_rating': 'B', 'expected_return': 0.07},
    {'id': 'loan4', 'prosper_rating': 'C', 'expected_return': 0.08},
    {'id': 'loan5', 'prosper_rating': 'B', 'expected_return': 0.065},
    {'id': 'loan6', 'prosper_rating': 'AA', 'expected_return': 0.055},
    {'id': 'loan7', 'prosper_rating': 'A', 'expected_return': 0.062},
    {'id': 'loan8', 'prosper_rating': 'B', 'expected_return': 0.071},
    {'id': 'loan9', 'prosper_rating': 'C', 'expected_return': 0.078},
    {'id': 'loan10', 'prosper_rating': 'B', 'expected_return': 0.068},
    {'id': 'loan11', 'prosper_rating': 'AA', 'expected_return': 0.049},
    {'id': 'loan12', 'prosper_rating': 'A', 'expected_return': 0.058},
    {'id': 'loan13', 'prosper_rating': 'B', 'expected_return': 0.072},
    {'id': 'loan14', 'prosper_rating': 'C', 'expected_return': 0.081},
    {'id': 'loan15', 'prosper_rating': 'B', 'expected_return': 0.063},
    {'id': 'loan16', 'prosper_rating': 'AA', 'expected_return': 0.055},
    {'id': 'loan17', 'prosper_rating': 'A', 'expected_return': 0.065},
    {'id': 'loan18', 'prosper_rating': 'B', 'expected_return': 0.07},
    {'id': 'loan19', 'prosper_rating': 'C', 'expected_return': 0.075},
    {'id': 'loan20', 'prosper_rating': 'B', 'expected_return': 0.068},
    {'id': 'loan21', 'prosper_rating': 'AA', 'expected_return': 0.05},
    {'id': 'loan22', 'prosper_rating': 'A', 'expected_return': 0.059},
    {'id': 'loan23', 'prosper_rating': 'B', 'expected_return': 0.073},
    {'id': 'loan24', 'prosper_rating': 'C', 'expected_return': 0.082},
    {'id': 'loan25', 'prosper_rating': 'B', 'expected_return': 0.064},
    {'id': 'loan26', 'prosper_rating': 'AA', 'expected_return': 0.052},
    {'id': 'loan27', 'prosper_rating': 'A', 'expected_return': 0.061},
    {'id': 'loan28', 'prosper_rating': 'B', 'expected_return': 0.074},
    {'id': 'loan29', 'prosper_rating': 'C', 'expected_return': 0.083},
    {'id': 'loan30', 'prosper_rating': 'B', 'expected_return': 0.066},
    {'id': 'loan31', 'prosper_rating': 'AA', 'expected_return': 0.03},
    {'id': 'loan32', 'prosper_rating': 'A', 'expected_return': 0.068},
    {'id': 'loan33', 'prosper_rating': 'B', 'expected_return': 0.076},
    {'id': 'loan34', 'prosper_rating': 'C', 'expected_return': 0.059},
    {'id': 'loan35', 'prosper_rating': 'B', 'expected_return': 0.054},
    {'id': 'loan36', 'prosper_rating': 'AA', 'expected_return': 0.018},
    {'id': 'loan37', 'prosper_rating': 'A', 'expected_return': 0.071},
    {'id': 'loan38', 'prosper_rating': 'B', 'expected_return': 0.077},
    {'id': 'loan39', 'prosper_rating': 'C', 'expected_return': 0.063},
    {'id': 'loan40', 'prosper_rating': 'B', 'expected_return': 0.056},
    {'id': 'loan41', 'prosper_rating': 'AA', 'expected_return': 0.028},
    {'id': 'loan42', 'prosper_rating': 'A', 'expected_return': 0.073},
    {'id': 'loan43', 'prosper_rating': 'B', 'expected_return': 0.079},
    {'id': 'loan44', 'prosper_rating': 'C', 'expected_return': 0.065},
    {'id': 'loan45', 'prosper_rating': 'B', 'expected_return': 0.059},
]

portfolio = [
    {'id': 'loan46', 'prosper_rating': 'AA', 'expected_return': 0.025},
    {'id': 'loan47', 'prosper_rating': 'A', 'expected_return': 0.077},
    {'id': 'loan48', 'prosper_rating': 'B', 'expected_return': 0.082},
    {'id': 'loan49', 'prosper_rating': 'C', 'expected_return': 0.068},
    {'id': 'loan50', 'prosper_rating': 'B', 'expected_return': 0.062},
    {'id': 'loan51', 'prosper_rating': 'AA', 'expected_return': 0.022},
    {'id': 'loan52', 'prosper_rating': 'A', 'expected_return': 0.075},
    {'id': 'loan53', 'prosper_rating': 'B', 'expected_return': 0.078},
    {'id': 'loan54', 'prosper_rating': 'C', 'expected_return': 0.067},
    {'id': 'loan55', 'prosper_rating': 'B', 'expected_return': 0.061},
    {'id': 'loan56', 'prosper_rating': 'AA', 'expected_return': 0.021},
    {'id': 'loan57', 'prosper_rating': 'A', 'expected_return': 0.074},
    {'id': 'loan58', 'prosper_rating': 'B', 'expected_return': 0.077},
    {'id': 'loan59', 'prosper_rating': 'C', 'expected_return': 0.066},
    {'id': 'loan60', 'prosper_rating': 'B', 'expected_return': 0.06},
]

solvers = ['GLPK_CMD', 'PYGLPK', 'CPLEX_CMD', 'CPLEX_PY', 'GUROBI', 'GUROBI_CMD', 'MOSEK', 'XPRESS', 'XPRESS', 'XPRESS_PY', 'PULP_CBC_CMD', 'COIN_CMD', 'COINMP_DLL', 'CHOCO_CMD', 'MIPCL_CMD', 'SCIP_CMD', 'HiGHS_CMD']



# for solver in solvers:
#     print(f"Using solver: {solver}")
#     selected_loans = optimize_portfolio(5, listings, portfolio=portfolio, optimization_solver='PULP_CBC_CMD', risk_free_rate=0.02, risk_preference=risk_preference)
#     print(f"Selected loans: {selected_loans}\n")


selected_loans = optimize_portfolio(5, listings, portfolio=portfolio, optimization_solver='PULP_CBC_CMD', risk_free_rate=0.02, risk_weight=0.5)
print(f"Selected loans (weighted preferences): {selected_loans}")

selected_loans2 = optimize_portfolio(5, listings, portfolio=portfolio, optimization_solver='PULP_CBC_CMD', risk_free_rate=0.02)
print(f"Selected loans (unweighted preference): {selected_loans2}")