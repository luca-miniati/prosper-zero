import pandas as pd
import os

drop_cols = [
        'amount_borrowed',
        'prosper_rating',
        'borrower_apr',
        'prosper_score',
        'borrower_rate',
        'prior_prosper_loans_active',
        'occupation',
        'investment_typeid'
    ]
# Directory containing the individual training files
training_files_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'clean')

# Output mega training and validation file paths
mega_training_file = 'data/clean/mega_training.csv'
mega_validation_file = 'data/clean/mega_val.csv'

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
    'income_range' : 10,
    'months_employed' : 200,
    'occupation' : 200,
    'listing_monthly_payment' : 1000,
    'stated_monthly_income' : 30000,
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

mega_training_df.drop(drop_cols, axis=1, inplace=True)
mega_validation_df.drop(drop_cols, axis=1, inplace=True)

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

balanced_df.to_csv('data/clean/balanced_mega_training.csv', index=False)

print(f'Balanced training file saved to balanced_mega_training.csv')

df = pd.read_csv(val_path)

loan_status_0 = df[df['loan_status'] == 0]
loan_status_1 = df[df['loan_status'] == 1]

sampled_loan_status = loan_status_0.sample(len(loan_status_1))

balanced_df = pd.concat([sampled_loan_status, loan_status_1])
balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)

balanced_df.to_csv('data/clean/balanced_mega_val.csv', index=False)

print(f'Balanced val file saved to balanced_mega_val.csv')
