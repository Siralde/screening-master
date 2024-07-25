import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
import pickle
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

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
    with open('model_results.pkl', 'wb') as file:
        pickle.dump({
            'results': results,
            'positive_predictions': positive_predictions,
            'feature_importances': feature_importance_df
        }, file)

    # Save the trained classifier
    with open('final_model.pkl', 'wb') as file:
        pickle.dump(classifier, file)

    # Save the label encoders
    with open('label_encoders.pkl', 'wb') as file:
        pickle.dump(encoders, file)

    # Save the column names
    with open('column_names.pkl', 'wb') as file:
        pickle.dump(column_names, file)

    # Save the target encoder
    with open('target_encoder.pkl', 'wb') as file:
        pickle.dump(target_encoder, file)

def analyze_numerical_features():
    with open('final_model.pkl', 'rb') as file:
        classifier = pickle.load(file)

    with open('column_names.pkl', 'rb') as file:
        column_names = pickle.load(file)

    # Assuming data is loaded from the same file and preprocessed in the same way
    data = pd.read_csv('unique_filtered_final_with_target_variable.csv')

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
        plt.savefig(f'{feature}_effect.png')
        plt.close()

@app.route("/", methods=["GET", "POST"])
def test_novel_datapoint():
    print("Index route accessed")
    if request.method == "POST":
        try:
            new_company_info = {
                'country_code': request.form['country_code'],
                'region': request.form['region'],
                'city': request.form['city'],
                'category_list': request.form['category_list'],
                'last_round_investment_type': request.form['last_round_investment_type'],
                'num_funding_rounds': int(request.form['num_funding_rounds']),
                'total_funding_usd': float(request.form['total_funding_usd']),
                'age_months': int(request.form['age_months']),
                'has_facebook_url': int(request.form.get('has_facebook_url', 0)),
                'has_twitter_url': int(request.form.get('has_twitter_url', 0)),
                'has_linkedin_url': int(request.form.get('has_linkedin_url', 0)),
                'round_count': int(request.form['round_count']),
                'raised_amount_usd': float(request.form['raised_amount_usd']),
                'last_round_raised_amount_usd': float(request.form['last_round_raised_amount_usd']),
                'last_round_post_money_valuation': float(request.form['last_round_post_money_valuation']),
                'last_round_timelapse_months': int(request.form['last_round_timelapse_months']),
                'last_round_investor_count': int(request.form['last_round_investor_count']),
                'founders_dif_country_count': int(request.form['founders_dif_country_count']),
                'founders_male_count': int(request.form['founders_male_count']),
                'founders_female_count': int(request.form['founders_female_count']),
                'founders_degree_count_total': int(request.form['founders_degree_count_total']),
                'founders_degree_count_max': int(request.form['founders_degree_count_max'])
            }

            print("New company info collected:")
            print(new_company_info)

            with open('./final_model.pkl', 'rb') as file:
                classifier = pickle.load(file)
                print("Classifier loaded")
            
            with open('./label_encoders.pkl', 'rb') as file:
                encoders = pickle.load(file)
                print("Encoders loaded")
            
            with open('./column_names.pkl', 'rb') as file:
                column_names = pickle.load(file)
                print("Column names loaded")

            def encode_and_handle_unseen(column, value):
                encoder = encoders[column]
                if value not in encoder.classes_:
                    encoder.classes_ = np.append(encoder.classes_, value)
                return encoder.transform([value])[0]

            new_company_df = pd.DataFrame([new_company_info])
            print("New company DataFrame created:")
            print(new_company_df)

            categorical_columns = [
                'country_code', 'region', 'city', 'category_list',
                'last_round_investment_type'
            ]
            for col in categorical_columns:
                new_company_df[col] = new_company_df[col].apply(lambda x: encode_and_handle_unseen(col, x))
                print(f"Encoded {col}:")
                print(new_company_df[col])

            new_company_df = new_company_df.reindex(columns=column_names, fill_value=0)
            print("Reindexed DataFrame:")
            print(new_company_df)

            prediction = int(classifier.predict(new_company_df)[0])
            confidence = float(classifier.predict_proba(new_company_df)[:, 1][0])
            if prediction == 0:
                confidence = 1 - confidence
            confidence = confidence * 100
            print(f"Prediction: {prediction}")
            print(f"Confidence: {confidence}")

            results = {
                "CL/NE_vs_FR/AC/IP Prediction": (prediction, confidence)
            }

            print("Results calculated")
            return jsonify(results=results)

        except Exception as e:
            print(f"An error occurred: {e}")
            return jsonify(error=str(e))

    return render_template("index.html")

def main():
    print("Main function")
    data = pd.read_csv('unique_filtered_final_with_target_variable.csv')
    data_column_names = data.columns
    print("Data loaded")
    analyze_numerical_features()

if __name__ == "__main__":
    main()
    print("Starting Flask app")
    app.run(debug=True, host='0.0.0.0', port=8080, use_reloader=False)
