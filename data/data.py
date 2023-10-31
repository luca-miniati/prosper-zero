import pandas as pd
import pickle
import os
import glob
import re
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")


if not os.path.exists('data/raw'):
    os.makedirs('data/raw')
if not os.path.exists('data/clean'):
    os.makedirs('data/clean')

all_listings = glob.glob('data/raw/Listings*.csv')
all_listings.sort()
all_loans = glob.glob('data/raw/Loans*.csv')
all_loans.sort()

for listings_path, loans_path in tqdm(zip(all_listings, all_loans)):
    if not os.path.exists(listings_path):
        print(f'File {listings_path} not found')

    if not os.path.exists(loans_path):
        print(f'File {loans_path} not found')


    listings = pd.read_csv(listings_path, encoding_errors='replace', low_memory=False)
    loans = pd.read_csv(loans_path)

    listings_rename_columns = {
        'loan_origination_date': 'origination_date',
        'listing_amount': 'amount_borrowed',
        'borrower_rate': 'borrower_rate',
        'prosper_rating': 'prosper_rating'
    }
    merge_columns = list(listings_rename_columns.values())

    listings.rename(listings_rename_columns, axis=1, inplace=True)

    listings.dropna(subset=merge_columns, inplace=True)
    listings['origination_date'] = pd.to_datetime(
        listings['origination_date'])

    loans.dropna(subset=merge_columns+['loan_status'], inplace=True)
    loans['origination_date'] = pd.to_datetime(loans['origination_date'])

    listings_no_duplicates = listings.loc[~listings.duplicated(
        subset=merge_columns, keep=False)]
    loans_no_duplicates = loans.loc[~loans.duplicated(
        subset=merge_columns, keep=False)]

    if listings_no_duplicates['fico_score'].isna().sum() > listings_no_duplicates['TUFicoRange'].isna().sum():
        listings_no_duplicates['fico_score'] = listings_no_duplicates.loc[:, 'TUFicoRange']
    if listings_no_duplicates['dti_wprosper_loan'].isna().sum() > listings_no_duplicates['CombinedDtiwProsperLoan'].isna().sum():
        listings_no_duplicates['dti_wprosper_loan'] = listings_no_duplicates.loc[:, 'CombinedDtiwProsperLoan']
    if listings_no_duplicates['stated_monthly_income'].isna().sum() > listings_no_duplicates['CombinedStatedMonthlyIncome'].isna().sum():
        listings_no_duplicates['stated_monthly_income'] = listings_no_duplicates.loc[:, 'CombinedStatedMonthlyIncome']

    listings_columns = [
        'fico_score',
        'employment_status_description',
        'dti_wprosper_loan',
        'prior_prosper_loans',
        'prior_prosper_loans_active',
        'investment_typeid',
        'borrower_apr',
        'income_verifiable',
        'listing_category_id',
        'months_employed',
        'income_range',
        'prosper_score',
        'prosper_rating',
        'listing_monthly_payment',
        'stated_monthly_income',
        'lender_indicator',
        'lender_yield',
        'occupation',
        'amount_borrowed',
        'borrower_rate',
        'origination_date',
    ]
    listings_no_duplicates = listings_no_duplicates.loc[:, listings_columns].dropna()
    loans_no_duplicates = loans_no_duplicates[merge_columns+['loan_status']]

    listings_final = pd.merge(
        listings_no_duplicates,
        loans_no_duplicates,
        on=merge_columns,
        how='inner',
        validate='1:1'
    )

    drop_cols = [
        'amount_borrowed',
        'origination_date', 
        'prosper_rating',
        'borrower_apr',
        'prosper_score',
        'borrower_rate'
    ]

    listings_final.drop('origination_date', axis=1, inplace=True)

    listings_final_no_encode = listings_final.copy()

    if not os.path.exists('label_encoders'):
        os.makedirs('label_encoders')

    categorical_cols = ['fico_score', 'employment_status_description', 'income_verifiable', 'occupation']

    for col in categorical_cols:
        with open(f'label_encoders/le_{col}.pkl', 'rb') as f:
            le = pickle.load(f)

        listings_final[col] = le.transform(listings_final[col])

    listings_final = listings_final[listings_final['dti_wprosper_loan'] < 5] 

    listings_final['loan_status'] = listings_final['loan_status'].apply(
        lambda x: 1 if x in [2, 3] else 0)
    

    
    start, end = re.search(r'_(\d{4})(?:\d{4})(?:to)(\d{4})', listings_path).groups()
    listings_final.to_csv(f'data/clean/{start}_{end}.csv', index=False)
    # listings_final_no_encode.to_csv(f'data/clean/{start}_{end}_no_encode.csv', index=False)
    print(f'WRITTEN: {start}-{end}')


