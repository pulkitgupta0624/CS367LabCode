import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Load the dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/new-thyroid.data'
column_names = ['Class', 'T3', 'T4', 'TSH']
data = pd.read_csv(url, names=column_names, delim_whitespace=True)

# Convert class labels to string for clarity
class_map = {1: 'Hyperthyroid', 2: 'Hypothyroid', 3: 'Normal'}
data['Class'] = data['Class'].map(class_map)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['Class']), data['Class'], test_size=0.3, random_state=42)

# Add labels to training set
X_train['Class'] = y_train

# Define the structure of the Bayesian Network
# You might want to adjust this according to the domain knowledge
model = BayesianNetwork([('T3', 'Class'), ('T4', 'Class'), ('TSH', 'Class')])

# Fit the model using Maximum Likelihood Estimation
model.fit(X_train, estimator=MaximumLikelihoodEstimator)

# Inference
inference = VariableElimination(model)

# Predict function using the trained model
def predict(model, X_test):
    y_pred = []
    for _, sample in X_test.iterrows():
        query = model.map_query(variables=['Class'], evidence=sample.to_dict())
        y_pred.append(query['Class'])
    return y_pred

# Perform predictions
y_pred = predict(inference, X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Expected output should show an accuracy >= 85%
