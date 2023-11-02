import pandas as pd
import pickle
import os
import glob
import re
import warnings
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

warnings.filterwarnings("ignore")


if not os.path.exists('data/raw'):
    os.makedirs('data/raw')
if not os.path.exists('data/clean'):
    os.makedirs('data/clean')

all_loans = glob.glob('data/raw/Loans*.csv')
all_loans.sort()

global_loans_data = pd.DataFrame()

for loans_path in tqdm(all_loans):

    if not os.path.exists(loans_path):
        print(f'File {loans_path} not found')

    loans = pd.read_csv(loans_path)

    merge_columns = ['amount_borrowed', "borrower_rate", 'prosper_rating', 'loan_status']


    loans.dropna(subset=merge_columns, inplace=True)
    loans['origination_date'] = pd.to_datetime(loans['origination_date'])

    loans_no_duplicates = loans.loc[~loans.duplicated(
        subset=merge_columns, keep=False)]
    loans_no_duplicates = loans_no_duplicates[merge_columns]

    drop_cols = [
        'amount_borrowed',
        'origination_date', 
        'prosper_rating',
        'borrower_apr',
        'prosper_score',
        'borrower_rate'
    ]

    if not os.path.exists('label_encoders'):
        os.makedirs('label_encoders')

    le = LabelEncoder()
    loans_no_duplicates['prosper_rating'] = le.fit_transform(loans_no_duplicates['prosper_rating'])

    with open('label_encoders/le_prosper_rating.pkl', 'wb') as f:
        pickle.dump(le, f)

    global_loans_data = pd.concat([global_loans_data, loans_no_duplicates])
    print(f' Processed loan data: {loans_path}')

gloabl_loans_data = global_loans_data[['prosper_rating', 'loan_status', 'amount_borrowed', 'borrower_rate']]

# Removing rows with 0 and 6 values and encoding other values to 0 and 1
global_loans_data = global_loans_data[(global_loans_data['loan_status'] != 0) & (global_loans_data['loan_status'] != 6)]
global_loans_data['loan_status'] = global_loans_data['loan_status'].replace({2: 0, 3: 0, 4: 1, 5: 1})

# Scaling the other columns
global_loans_data['amount_borrowed'] /= 30000
global_loans_data['prosper_rating'] /= 10

train_data, validation_data = train_test_split(global_loans_data, test_size=0.3, random_state=42)

train_data.to_csv('data/clean/loans_data_train.csv', index=False)
validation_data.to_csv('data/clean/loans_data_val.csv', index=False)

print('Loans training data saved to loans_data_train.csv')
print('Loans val data saved to loans_data_val.csv')

