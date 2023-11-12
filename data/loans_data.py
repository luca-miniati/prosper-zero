import pandas as pd
import pickle
import os
import glob
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

    merge_columns = ['prosper_rating', 'loan_status']

    loans = loans.loc[:,merge_columns]
    loans.dropna(inplace=True)


    # drop_cols = [
    #     'amount_borrowed',
    #     'origination_date', 
    #     'prosper_rating',
    #     'borrower_apr',
    #     'prosper_score',
    #     'borrower_rate'
    # ]

    if not os.path.exists('label_encoders'):
        os.makedirs('label_encoders')

    le = LabelEncoder()
    loans['prosper_rating'] = le.fit_transform(loans['prosper_rating'])

    with open('label_encoders/le_prosper_rating.pkl', 'wb') as f:
        pickle.dump(le, f)

    global_loans_data = pd.concat([global_loans_data, loans])
    print(f' Processed loan data: {loans_path}')

gloabl_loans_data = global_loans_data[['prosper_rating', 'loan_status']]

# Removing rows with 0 and 6 values and encoding other values to 0 and 1
# global_loans_data = global_loans_data[(global_loans_data['loan_status'] != 0) and (global_loans_data['loan_status'] != 6)]
global_loans_data.drop(global_loans_data[global_loans_data['loan_status'] == 0].index)
global_loans_data.drop(global_loans_data[global_loans_data['loan_status'] == 6].index)
global_loans_data.drop(global_loans_data[global_loans_data['loan_status'] == 5].index)
global_loans_data['loan_status'] = global_loans_data['loan_status'].replace({1: 0, 2: 1, 3: 1, 4: 0})

# Scaling the other columns
# global_loans_data['amount_borrowed'] /= 30000
global_loans_data['prosper_rating'] /= 10

train_data, validation_data = train_test_split(global_loans_data, test_size=0.3, random_state=42)

train_data.to_csv('data/clean/loans_data_train.csv', index=False)
validation_data.to_csv('data/clean/loans_data_val.csv', index=False)

print('Loans training data saved to loans_data_train.csv')
print('Loans val data saved to loans_data_val.csv')

train_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'clean', 'loans_data_train.csv')
val_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'clean', 'loans_data_val.csv')

df = pd.read_csv(train_path)

loan_status_0 = df[df['loan_status'] == 0]
loan_status_1 = df[df['loan_status'] == 1]

sampled_loan_status = loan_status_0.sample(len(loan_status_1))

balanced_df = pd.concat([sampled_loan_status, loan_status_1])
balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)

balanced_df.to_csv('data/clean/balanced_loan_train.csv', index=False)

print(f'Balanced training file saved to balanced_loan_train.csv')

df = pd.read_csv(val_path)

loan_status_0 = df[df['loan_status'] == 0]
loan_status_1 = df[df['loan_status'] == 1]

sampled_loan_status = loan_status_0.sample(len(loan_status_1))

balanced_df = pd.concat([sampled_loan_status, loan_status_1])
balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)

balanced_df.to_csv('data/clean/balanced_loan_val.csv', index=False)

print(f'Balanced val file saved to balanced_loan_val.csv')
