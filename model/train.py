import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os
import seaborn as sns
import matplotlib.pyplot as plt

train_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'clean', 'balanced_mega_training.csv')
val_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'clean', 'balanced_mega_val.csv')


train_data = pd.read_csv(train_path)
train_data = train_data.iloc[:100000]
val_data = pd.read_csv(val_path)
val_data = val_data.iloc[:40000]

# correlation_matrix = train_data.corr()

# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
# plt.show()




X_train = train_data.drop('loan_status', axis=1)
y_train = train_data['loan_status']

X_validation = val_data.drop('loan_status', axis=1)
y_validation = val_data['loan_status']


model = LogisticRegression(max_iter=500)
print(f'Data loaded and model initialized')
model.fit(X_train, y_train)

validation_accuracy = model.score(X_validation, y_validation)

print(validation_accuracy)
predictions = model.predict(X_validation)
