import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
import pickle


# Path definitions

base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, '../../data/csvs')
pkl_path = os.path.join(base_path, '../../data/pkls')
template_path = os.path.join(base_path, '../../frontend/templates')

def analyze_numerical_features():
    with open(os.path.join(pkl_path, 'final_model.pkl'), 'rb') as file:
        classifier = pickle.load(file)

    with open(os.path.join(pkl_path, 'column_names.pkl'), 'rb') as file:
        column_names = pickle.load(file)

    # Assuming data is loaded from the same file and preprocessed in the same way
    data = pd.read_csv(os.path.join(data_path, 'unique_filtered_final_with_target_variable.csv'))

    # Encode categorical variables
    categorical_columns = [
        'country_code', 'region', 'city', 
        'category_list', 'last_round_investment_type'
    ]

    encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        encoders[col] = le

    # Encode target variable
    target_encoder = LabelEncoder()
    data['outcome'] = target_encoder.fit_transform(data['outcome'].astype(str))

    # Define features and print included and excluded features
    excluded_features = [
        'uuid_org', 'name_org', 'permalink_org', 'domain', 'homepage_url', 
        'address', 'postal_code', 'short_description', 'facebook_url', 
        'linkedin_url', 'twitter_url', 'founded_on', 'last_funding_on', 
        'closed_on', 'total_funding_currency_code', 'outcome', 'state_code', 
        'status', 'total_funding', 'category_groups_list', 'founders_degree_count_mean'
    ]
    X = data.drop(columns=excluded_features)
    
    # Ensure all feature names are valid
    numerical_features = [
        'num_funding_rounds', 'total_funding_usd', 'round_count', 
        'last_round_timelapse_months', 'age_months', 'raised_amount_usd', 
        'last_round_raised_amount_usd', 'last_round_post_money_valuation',
        'last_round_investor_count', 'founders_dif_country_count', 
        'founders_male_count', 'founders_female_count', 
        'founders_degree_count_total', 'founders_degree_count_max'
    ]
    numerical_features = [feature for feature in numerical_features if feature in X.columns]

    for feature in numerical_features:
        x_values = np.linspace(X[feature].min(), X[feature].max(), 100)
        y_values = []
        for x in x_values:
            test_sample = X.iloc[0].copy()
            test_sample[feature] = x
            test_sample_df = pd.DataFrame([test_sample], columns=column_names)
            y_values.append(classifier.predict_proba(test_sample_df)[0, 1])

        plt.figure()
        plt.plot(x_values, y_values)
        plt.title(f'Effect of {feature} on Positive Classification')
        plt.xlabel(f'{feature}')
        plt.ylabel('Probability of Positive Classification')
        plt.savefig(os.path.join(base_path, f'../../data/pngs/{feature}_effect.png'))
        plt.close()

def train_model(data):
    # Encode categorical variables
    categorical_columns = [
        'country_code', 'region', 'city', 
        'category_list', 'last_round_investment_type'
    ]

    encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        encoders[col] = le

    # Encode target variable
    target_encoder = LabelEncoder()
    data['outcome'] = target_encoder.fit_transform(data['outcome'].astype(str))

    # Define features and print included and excluded features
    excluded_features = [
        'uuid_org', 'name_org', 'permalink_org', 'domain', 'homepage_url', 
        'address', 'postal_code', 'short_description', 'facebook_url', 
        'linkedin_url', 'twitter_url', 'founded_on', 'last_funding_on', 
        'closed_on', 'total_funding_currency_code', 'outcome', 'state_code', 
        'status', 'total_funding', 'category_groups_list', 'founders_degree_count_mean'
    ]
    X = data.drop(columns=excluded_features)

    # Save the column names
    column_names = X.columns.tolist()

    # Binary target for the specified classification
    data['CL/NE_vs_FR/AC/IP'] = data['outcome'].apply(lambda x: 1 if x in target_encoder.transform(['FR', 'AC', 'IP']) else 0)

    # Define classifier
    classifier = RandomForestClassifier()

    # Train and evaluate classifier
    results = {}
    positive_predictions = {}

    target = 'CL/NE_vs_FR/AC/IP'
    y = data[target]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    precision_scores = []
    recall_scores = []

    feature_importances = []
    
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        y_proba = classifier.predict_proba(X_test)[:, 1]  # Probability of the positive class
        
        precision_scores.append(precision_score(y_test, y_pred, zero_division=0))
        recall_scores.append(recall_score(y_test, y_pred, zero_division=0))
        
        feature_importances.append(classifier.feature_importances_)
        
        # Identify true positive predictions with their probabilities
        for i in range(len(y_test)):
            if y_pred[i] == 1 and y_test.iloc[i] == y_pred[i]:
                index = X_test.index[i]
                if index not in positive_predictions:
                    positive_predictions[index] = {}
                positive_predictions[index][target] = y_proba[i]
    
    results = {
        'mean_precision': np.mean(precision_scores),
        'std_precision': np.std(precision_scores),
        'mean_recall': np.mean(recall_scores),
        'std_recall': np.std(recall_scores)
    }

    mean_feature_importances = np.mean(feature_importances, axis=0)
    feature_importance_df = pd.DataFrame({'Feature': column_names, 'Importance': mean_feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    print(f"{target}: Mean precision = {np.mean(precision_scores):.4f}, Std = {np.std(precision_scores):.4f}")
    print(f"{target}: Mean recall = {np.mean(recall_scores):.4f}, Std = {np.std(recall_scores):.4f}")
    print("Feature importances:")
    print(feature_importance_df)
    
    # Train on the full dataset for predictions
    classifier.fit(X, y)
    data[f'{target}_Prediction'] = classifier.predict(X)
    data[f'{target}_Confidence'] = classifier.predict_proba(X)[:, 1]

    # Save training and evaluation results to a file
    with open(os.path.join(pkl_path, 'model_results.pkl'), 'wb') as file:
        pickle.dump({
            'results': results,
            'positive_predictions': positive_predictions,
            'feature_importances': feature_importance_df
        }, file)

    # Save the trained classifier
    with open(os.path.join(pkl_path, 'final_model.pkl'), 'wb') as file:
        pickle.dump(classifier, file)

    # Save the label encoders
    with open(os.path.join(pkl_path, 'label_encoders.pkl'), 'wb') as file:
        pickle.dump(encoders, file)

    # Save the column names
    with open(os.path.join(pkl_path, 'column_names.pkl'), 'wb') as file:
        pickle.dump(column_names, file)

    # Save the target encoder
    with open(os.path.join(pkl_path, 'target_encoder.pkl'), 'wb') as file:
        pickle.dump(target_encoder, file)