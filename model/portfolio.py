import pandas as pd
import numpy as np
import os

train_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'clean', 'loans_data_train.csv')
val_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'clean', 'loans_data_val.csv')


train_data = pd.read_csv(train_path)
val_data = pd.read_csv(val_path)

grouped = train_data.groupby('prosper_rating')
result = grouped['loan_status'].mean() 
train_data['default_probability'] = train_data['prosper_rating'].map(result)

train_data = train_data[['prosper_rating', 'default_probability', 'loan_status']]

cov_matrix = np.diag(result * (1 - result))
'''
grouped = loan_data.groupby('prosper_rating')['default_probability'].mean()

# Create a covariance matrix with diagonal elements equal to the variances of default probabilities
cov_matrix = np.diag(grouped ** 2)

# Use the covariance matrix in portfolio optimization
risk_model = RiskModel(cov_matrix)
ef = EfficientFrontier(expected_returns=None, risk_model=risk_model)

# Optimize your portfolio based on your objectives
# Example: Minimize volatility
weights = ef.min_volatility()

print(weights)
'''