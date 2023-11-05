import pandas as pd
# from sklearn.linear_model import LogisticRegression
import os


train_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'clean', 'balanced_mega_training.csv')
val_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'clean', 'balanced_mega_val.csv')

train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)

# column_order = [
#     'fico_score',
#     'lender_yield',
#     'income_range',
#     'listing_monthly_payment',
#     'stated_monthly_income',
#     'lender_indicator',
#     'prior_prosper_loans',
#     'dti_wprosper_loan',
#     'months_employed',
#     'income_verifiable',
#     'listing_category_id',
#     'employment_status_description',
#     'loan_status',
# ]
column_order = [
    'fico_score',
    'income_range',
    'dti_wprosper_loan',
    'loan_status',
]

train_df = train_df[column_order]
val_df = val_df[column_order]

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
print(f'Reordered training data to {train_path}')
print(f'Reordered val data to {val_path}')