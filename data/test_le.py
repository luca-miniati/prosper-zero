import pandas as pd
import numpy as np
import pickle
import glob
import re

all_listings = glob.glob('data/raw/Listings*.csv')

for listings_path in all_listings:
    start, end = re.search(r'_(\d{4})(?:\d{4})(?:to)(\d{4})', listings_path).groups()
    df = pd.read_csv(f'data/clean/{start}_{end}.csv')
    df_no_encode = pd.read_csv(f'data/clean/{start}_{end}_no_encode.csv')

    encoded_columns = ['employment_status_description', 'income_verifiable', 'occupation', 'prosper_rating']

    for col in encoded_columns:
        with open(f'label_encoders/le_{col}.pkl', 'rb') as f:
            le = pickle.load(f)
        inv_col = le.inverse_transform(df[col])
        assert np.array_equal(inv_col, df_no_encode[col].values)

print('ALL TESTS PASSED!')