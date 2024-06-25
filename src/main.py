import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

import pickle
import random

import pandas as pd


data = pd.read_csv('data/filtered_CB_data.csv')


# Encode categorical variables
categorical_columns = [
    'country_code', 'state_code', 'region', 'city', 'status', 
    'category_list', 'category_groups_list', 'last_round_investment_type'
]

le = LabelEncoder()
for col in categorical_columns:
    data[col] = le.fit_transform(data[col].astype(str))


# Encode target variable
data['outcome'] = le.fit_transform(data['outcome'].astype(str))

y = data['outcome']

# Define features and target
X = data.drop(columns=[
    'uuid_org', 'name_org', 'permalink_org', 'domain', 'homepage_url', 
    'address', 'postal_code', 'short_description', 'facebook_url', 
    'linkedin_url', 'twitter_url', 'founded_on', 'last_funding_on', 
    'closed_on', 'total_funding_currency_code','outcome'
])


# Define classifiers
classifiers = {
    #'SVM': SVC(kernel='linear'), (TOO SLOW)
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Extra Trees': ExtraTreesClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

# Perform stratified k-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate classifiers and save results
results = {}

for clf_name, clf in classifiers.items():
    cv_results = cross_val_score(clf, X, y, cv=skf, scoring='accuracy')
    results[clf_name] = {
        'mean_accuracy': np.mean(cv_results),
        'std_accuracy': np.std(cv_results),
        'cv_results': cv_results
    }
    print(f"{clf_name}: Mean accuracy = {np.mean(cv_results):.4f}, Std = {np.std(cv_results):.4f}")

# Decode the target variable back to original labels for readability
data['outcome'] = le.inverse_transform(data['outcome'])

# Print distribution of the outcome variable
outcome_counts = data['outcome'].value_counts()
total = len(data)
print("\nClass value distribution of the outcome variable:")
print("Class\tFrequency\tRatio")
outcome_distribution = {}
for cls, count in outcome_counts.items():
    ratio = (count / total) * 100
    print(f"{cls}\t{count}\t{ratio:.2f}%")
    outcome_distribution[cls] = {
        'count': count,
        'ratio': ratio
    }

# Save results and variables to a file using pickle
file_number = random.randint(1000, 9999)
file_path = f"results/model_results_{file_number}.pkl"

with open(file_path, 'wb') as file:
    pickle.dump({
        'results': results,
        'outcome_distribution': outcome_distribution,
        'classes': le.classes_
    }, file)

print(f"\nResults and variables have been saved to 'results/model_results_{file_number}.pkl'")