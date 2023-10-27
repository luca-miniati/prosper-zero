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

    listings_final.drop('origination_date', axis=1, inplace=True)

    listings_final_no_encode = listings_final.copy()

    if not os.path.exists('label_encoders'):
        os.makedirs('label_encoders')

    categorical_cols = ['fico_score', 'employment_status_description', 'income_verifiable', 'occupation', 'prosper_rating']

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


# Directory containing the individual training files
training_files_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'clean')

# Output mega training and validation file paths
mega_training_file = 'mega_training.csv'
mega_validation_file = 'mega_val.csv'

# List of training files
training_files = [os.path.join(training_files_dir, filename) for filename in os.listdir(training_files_dir) if filename.endswith('.csv')]

# Initialize mega training and validation DataFrames
mega_training_df = pd.DataFrame()
mega_validation_df = pd.DataFrame()

training_dfs = []
validation_dfs = []

# Split ratio (70% training, 30% validation)
split_ratio = 0.7
scale_factors = {
    'fico_score' : 10,
    'investment_typeid' : 10,
    'listing_category_id' : 20,
    'prosper_score' : 20,
    'income_range' : 10,
    'months_employed' : 200,
    'occupation' : 200,
    'listing_monthly_payment' : 1000,
    'stated_monthly_income' : 30000,
    'amount_borrowed' : 40000,
    'prior_prosper_loans' : 10
}

# Process each training file
for training_file in training_files:
    try:
        # Load the data from the current training file
        current_data = pd.read_csv(training_file)
        for column, scale_factor in scale_factors.items():
                current_data[column] /= scale_factor

        
        # Split the data into training and validation subsets
        num_training_samples = int(len(current_data) * split_ratio)
        training_subset = current_data[:num_training_samples]
        validation_subset = current_data[num_training_samples:]
        
        # Append the subsets to the lists
        training_dfs.append(training_subset)
        validation_dfs.append(validation_subset)
    except Exception as e:
        print(f"Error processing {training_file}: {str(e)}")

              
mega_training_df = pd.concat(training_dfs, ignore_index=True)
mega_validation_df = pd.concat(validation_dfs, ignore_index=True)

# Save the mega training and validation files
mega_training_df.to_csv(mega_training_file, index=False)
mega_validation_df.to_csv(mega_validation_file, index=False)

print(f'Total training file saved to {mega_training_file}')
print(f'Total validation file saved to {mega_validation_file}')

train_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'clean', 'mega_training.csv')
val_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'clean', 'mega_val.csv')

df = pd.read_csv(train_path)

loan_status_0 = df[df['loan_status'] == 0]
loan_status_1 = df[df['loan_status'] == 1]

sampled_loan_status = loan_status_0.sample(len(loan_status_1))

balanced_df = pd.concat([sampled_loan_status, loan_status_1])
balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)

balanced_df.to_csv('balanced_mega_training.csv', index=False)

print(f'Balanced training file saved to balanced_mega_training.csv')

df = pd.read_csv(val_path)

loan_status_0 = df[df['loan_status'] == 0]
loan_status_1 = df[df['loan_status'] == 1]

sampled_loan_status = loan_status_0.sample(len(loan_status_1))

balanced_df = pd.concat([sampled_loan_status, loan_status_1])
balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)

balanced_df.to_csv('balanced_mega_val.csv', index=False)

print(f'Balanced val file saved to balanced_mega_val.csv')
