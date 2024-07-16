import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score
import pickle
import pandas as pd
import ipywidgets as widgets
from IPython.display import display


def train_model():
    #load data
    data = pd.read_csv('unique_filtered_final_with_target_variable.csv')

    data_column_names = data.columns
    print("Column names in the data.csv:")
    for name in data_column_names:
        print(name)

    # Assuming data is already loaded in 'data' DataFrame

    # Encode categorical variables
    categorical_columns = [
        'country_code', 'region', 'city', 
        'category_list', 'category_groups_list', 'last_round_investment_type'
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
        'status', 'total_funding', 'category_groups_list'
    ]
    X = data.drop(columns=excluded_features)

    # Save the column names
    column_names = X.columns.tolist()

    # Binary targets for specified classifications
    data['IPO_vs_Other'] = (data['outcome'] == target_encoder.transform(['IP'])[0]).astype(int)
    data['FR_vs_Other'] = (data['outcome'] == target_encoder.transform(['FR'])[0]).astype(int)
    data['NE_vs_Other'] = (data['outcome'] == target_encoder.transform(['NE'])[0]).astype(int)
    data['AC_vs_Other'] = (data['outcome'] == target_encoder.transform(['AC'])[0]).astype(int)
    data['CL_vs_Other'] = (data['outcome'] == target_encoder.transform(['CL'])[0]).astype(int)

    # Define classifiers
    classifiers = {
        'IPO_vs_Other_RF': RandomForestClassifier(),
        'FR_vs_Other_RF': RandomForestClassifier(),
        'NE_vs_Other_RF': RandomForestClassifier(),
        'CL_vs_Other_RF': RandomForestClassifier(),
        'AC_vs_Other_GB': GradientBoostingClassifier()
    }

    # Train and evaluate classifiers
    results = {}
    positive_predictions = {
        'IPO_vs_Other_RF': {}, 'FR_vs_Other_RF': {}, 'NE_vs_Other_RF': {}, 'CL_vs_Other_RF': {}, 'AC_vs_Other_GB': {}
    }

    for target, clf in classifiers.items():
        outcome, model_type = target.rsplit('_', 1)
        y = data[outcome]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        precision_scores = []
        recall_scores = []
        
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)[:, 1]  # Probability of the positive class
            
            precision_scores.append(precision_score(y_test, y_pred, zero_division=0))
            recall_scores.append(recall_score(y_test, y_pred, zero_division=0))
            
            # Identify true positive predictions with their probabilities
            for i in range(len(y_test)):
                if y_pred[i] == 1 and y_test.iloc[i] == y_pred[i]:
                    index = X_test.index[i]
                    if index not in positive_predictions[target]:
                        positive_predictions[target][index] = {}
                    positive_predictions[target][index][outcome] = y_proba[i]
        
        results[target] = {
            'mean_precision': np.mean(precision_scores),
            'std_precision': np.std(precision_scores),
            'mean_recall': np.mean(recall_scores),
            'std_recall': np.std(recall_scores)
        }
        
        # Train on the full dataset for predictions
        clf.fit(X, y)
        data[f'{outcome}_{model_type}_Prediction'] = clf.predict(X)
        data[f'{outcome}_{model_type}_Probability'] = clf.predict_proba(X)[:, 1]

        print(f"{target}: Mean precision = {np.mean(precision_scores):.4f}, Std = {np.std(precision_scores):.4f}")
        print(f"{target}: Mean recall = {np.mean(recall_scores):.4f}, Std = {np.std(recall_scores):.4f}")

    # Save training and evaluation results to a file
    with open('model_training_results.pkl', 'wb') as file:
        pickle.dump({
            'results': results,
            'positive_predictions': positive_predictions
        }, file)

    # Save the trained classifiers
    with open('final_models.pkl', 'wb') as file:
        pickle.dump(classifiers, file)

    # Save the label encoders
    with open('label_encoders.pkl', 'wb') as file:
        pickle.dump(encoders, file)

    # Save the column names
    with open('column_names.pkl', 'wb') as file:
        pickle.dump(column_names, file)

    # Save the target encoder
    with open('target_encoder.pkl', 'wb') as file:
        pickle.dump(target_encoder, file)

    print("Training and evaluation results have been saved to 'model_training_results.pkl'")
    print("Trained models have been saved to 'final_models.pkl'")
    print("Label encoders have been saved to 'label_encoders.pkl'")
    print("Column names have been saved to 'column_names.pkl'")
    print("Target encoder has been saved to 'target_encoder.pkl'")

def save_sample_of_training_predictions(): # this one might not be working accurately
    # Load the results from the training phase
    with open('model_training_results.pkl', 'rb') as file:
        data_dict = pickle.load(file)

    results = data_dict['results']
    positive_predictions = data_dict['positive_predictions']

    # Load the original data (assuming it is still available)
    # data = pd.read_csv('path_to_data.csv')  # Uncomment and provide the path if necessary

    # Combine probabilities for each true positive prediction
    combined_predictions = []
    for idx in positive_predictions['IPO_vs_Other_RF'].keys():
        combined_prediction = {
            'index': idx,
            'IPO_probability': positive_predictions['IPO_vs_Other_RF'][idx].get('IPO_vs_Other', 0),
            'FR_probability': positive_predictions['FR_vs_Other_RF'].get(idx, {}).get('FR_vs_Other', 0),
            'AC_probability': positive_predictions['AC_vs_Other_GB'].get(idx, {}).get('AC_vs_Other', 0)
        }
        combined_predictions.append(combined_prediction)

    # Convert to DataFrame
    combined_predictions_df = pd.DataFrame(combined_predictions)

    # Merge with original data for full company details
    if not combined_predictions_df.empty:
        combined_predictions_df = combined_predictions_df.merge(data, left_on='index', right_index=True)

    # Select a sample of 15 from each category (IPO, FR, AC)
    ipo_sample = combined_predictions_df.sample(n=15, random_state=42) if len(combined_predictions_df) > 15 else combined_predictions_df
    fr_sample = combined_predictions_df.sample(n=15, random_state=42) if len(combined_predictions_df) > 15 else combined_predictions_df
    ac_sample = combined_predictions_df.sample(n=15, random_state=42) if len(combined_predictions_df) > 15 else combined_predictions_df

    # Formatting the results for better readability
    def format_predictions(df, title):
        formatted = f"# {title}\n\n"
        for idx, row in df.iterrows():
            formatted += f"## Company {idx + 1}\n\n"
            formatted += f"**IPO Probability:** {row['IPO_probability']:.4f}\n"
            formatted += f"**FR Probability:** {row['FR_probability']:.4f}\n"
            formatted += f"**AC Probability:** {row['AC_probability']:.4f}\n\n"
            for col in df.columns:
                if col not in ['index', 'IPO_probability', 'FR_probability', 'AC_probability']:
                    formatted += f"**{col}:** {row[col]}\n\n"
            formatted += "\n\n"
        return formatted

    ipo_formatted = format_predictions(ipo_sample, "IPO Predictions")
    fr_formatted = format_predictions(fr_sample, "Funding Round Predictions")
    ac_formatted = format_predictions(ac_sample, "Acquisition Predictions")

    # Save formatted results to a markdown file
    with open('positive_predictions.md', 'w') as file:
        file.write(ipo_formatted)
        file.write("\n\n")
        file.write(fr_formatted)
        file.write("\n\n")
        file.write(ac_formatted)

    print("Positive predictions have been saved to 'positive_predictions.md'")

    # Save the results and positive predictions to a file using pickle
    with open('model_results.pkl', 'wb') as file:
        pickle.dump({
            'results': results,
            'positive_predictions': pd.concat([ipo_sample, fr_sample, ac_sample])
        }, file)

    print("Results and variables have been saved to 'model_results.pkl'")

def test_novel_datapoint():
    # Load the models
    with open('./final_models.pkl', 'rb') as file:
        classifiers = pickle.load(file)

    # Load the encoders
    with open('./label_encoders.pkl', 'rb') as file:
        encoders = pickle.load(file)

    # Load the column names
    with open('./column_names.pkl', 'rb') as file:
        column_names = pickle.load(file)

    # Function to prompt the user for new company information
    def get_new_company_info():
        new_company = {
            'country_code': country_code.value,
            'region': region.value,
            'city': city.value,
            'category_list': category_list.value,
            'category_groups_list': category_groups_list.value,
            'last_round_investment_type': last_round_investment_type.value,
            'num_funding_rounds': num_funding_rounds.value,
            'total_funding_usd': total_funding_usd.value,
            'age_months': age_months.value,
            'has_facebook_url': int(has_facebook_url.value),
            'has_twitter_url': int(has_twitter_url.value),
            'has_linkedin_url': int(has_linkedin_url.value),
            'round_count': round_count.value,
            'raised_amount_usd': raised_amount_usd.value,
            'last_round_raised_amount_usd': last_round_raised_amount_usd.value,
            'last_round_post_money_valuation': last_round_post_money_valuation.value,
            'last_round_timelapse_months': last_round_timelapse_months.value,
            'investor_countup': investor_countup.value,
            'last_round_investor_count': last_round_investor_count.value,
            'founders_dif_country_count': founders_dif_country_count.value,
            'founders_male_count': founders_male_count.value,
            'founders_female_count': founders_female_count.value,
            'founders_degree_count_total': founders_degree_count_total.value,
            'founders_degree_count_max': founders_degree_count_max.value,
            'founders_degree_count_mean': founders_degree_count_mean.value
        }
        return new_company

    # Creating input widgets with example values and adjusted description width
    widget_layout = widgets.Layout(width='400px')
    widget_desc_layout = widgets.Layout(width='200px')

    country_code = widgets.Text(value='USA', description="Country Code:", layout=widget_layout, style={'description_width': 'initial'})
    region = widgets.Text(value='California', description="Region:", layout=widget_layout, style={'description_width': 'initial'})
    city = widgets.Text(value='Santa Clara', description="City:", layout=widget_layout, style={'description_width': 'initial'})
    category_list = widgets.Text(value='Artificial Intelligence (AI),E-Commerce,Software', description="Category List:", layout=widget_layout, style={'description_width': 'initial'})
    category_groups_list = widgets.Text(value='Artificial Intelligence (AI),Commerce and Shopping,Data and Analytics,Science and Engineering,Software', description="Category Groups List:", layout=widget_layout, style={'description_width': 'initial'})
    last_round_investment_type = widgets.Text(value='seed', description="Last Round Investment Type:", layout=widget_layout, style={'description_width': 'initial'})
    num_funding_rounds = widgets.IntText(value=1, description="Number of Funding Rounds:", layout=widget_layout, style={'description_width': 'initial'})
    total_funding_usd = widgets.FloatText(value=4000000.0, description="Total Funding USD:", layout=widget_layout, style={'description_width': 'initial'})
    age_months = widgets.IntText(value=2, description="Age in Months:", layout=widget_layout, style={'description_width': 'initial'})
    has_facebook_url = widgets.Checkbox(value=True, description="Has Facebook URL:", layout=widget_layout, style={'description_width': 'initial'})
    has_twitter_url = widgets.Checkbox(value=True, description="Has Twitter URL:", layout=widget_layout, style={'description_width': 'initial'})
    has_linkedin_url = widgets.Checkbox(value=True, description="Has LinkedIn URL:", layout=widget_layout, style={'description_width': 'initial'})
    round_count = widgets.IntText(value=1, description="Round Count:", layout=widget_layout, style={'description_width': 'initial'})
    raised_amount_usd = widgets.FloatText(value=0.0, description="Raised Amount USD:", layout=widget_layout, style={'description_width': 'initial'})
    last_round_raised_amount_usd = widgets.FloatText(value=0.0, description="Last Round Raised Amount USD:", layout=widget_layout, style={'description_width': 'initial'})
    last_round_post_money_valuation = widgets.FloatText(value=0.0, description="Last Round Post Money Valuation:", layout=widget_layout, style={'description_width': 'initial'})
    last_round_timelapse_months = widgets.IntText(value=0, description="Last Round Timelapse Months:", layout=widget_layout, style={'description_width': 'initial'})
    investor_countup = widgets.IntText(value=0, description="Investor Count:", layout=widget_layout, style={'description_width': 'initial'})
    last_round_investor_count = widgets.IntText(value=0, description="Last Round Investor Count:", layout=widget_layout, style={'description_width': 'initial'})
    founders_dif_country_count = widgets.IntText(value=0, description="Founders Different Country Count:", layout=widget_layout, style={'description_width': 'initial'})
    founders_male_count = widgets.IntText(value=0, description="Founders Male Count:", layout=widget_layout, style={'description_width': 'initial'})
    founders_female_count = widgets.IntText(value=0, description="Founders Female Count:", layout=widget_layout, style={'description_width': 'initial'})
    founders_degree_count_total = widgets.IntText(value=0, description="Founders Degree Count Total:", layout=widget_layout, style={'description_width': 'initial'})
    founders_degree_count_max = widgets.IntText(value=0, description="Founders Degree Count Max:", layout=widget_layout, style={'description_width': 'initial'})
    founders_degree_count_mean = widgets.FloatText(value=0.0, description="Founders Degree Count Mean:", layout=widget_layout, style={'description_width': 'initial'})

    # Display widgets
    display(
        country_code, region, city, category_list, category_groups_list, last_round_investment_type,
        num_funding_rounds, total_funding_usd, age_months, has_facebook_url, has_twitter_url, has_linkedin_url,
        round_count, raised_amount_usd, last_round_raised_amount_usd, last_round_post_money_valuation,
        last_round_timelapse_months, investor_countup, last_round_investor_count, founders_dif_country_count,
        founders_male_count, founders_female_count, founders_degree_count_total, founders_degree_count_max,
        founders_degree_count_mean
    )

    # Button to trigger the prediction
    button = widgets.Button(description="Submit and Predict")
    output = widgets.Output()

    def encode_and_handle_unseen(column, value):
        encoder = encoders[column]
        if value not in encoder.classes_:
            # Add new class to the encoder
            encoder.classes_ = np.append(encoder.classes_, value)
        return encoder.transform([value])[0]

    def on_button_clicked(b):
        with output:
            output.clear_output()
            new_company_info = get_new_company_info()

            new_company_df = pd.DataFrame([new_company_info])

            # Apply the same preprocessing steps as used for training
            categorical_columns = [
                'country_code', 'region', 'city', 'category_list', 
                'category_groups_list', 'last_round_investment_type'
            ]
            for col in categorical_columns:
                new_company_df[col] = new_company_df[col].apply(lambda x: encode_and_handle_unseen(col, x))

            # Ensure the order and presence of columns match the training phase
            new_company_df = new_company_df.reindex(columns=column_names, fill_value=0)

            predictions = {}
            probabilities = {}
            
            for target, clf in classifiers.items():
                predictions[target] = clf.predict(new_company_df)[0]
                probabilities[target] = clf.predict_proba(new_company_df)[:, 1][0]

            print("\nPredicted outcomes and probabilities for the new company:")
            print(f"IPO Prediction: {predictions['IPO_vs_Other_RF']}, Probability: {probabilities['IPO_vs_Other_RF']:.4f}")
            print(f"FR Prediction: {predictions['FR_vs_Other_RF']}, Probability: {probabilities['FR_vs_Other_RF']:.4f}")
            print(f"NE Prediction: {predictions['NE_vs_Other_RF']}, Probability: {probabilities['NE_vs_Other_RF']:.4f}")
            print(f"CL Prediction: {predictions['CL_vs_Other_RF']}, Probability: {probabilities['CL_vs_Other_RF']:.4f}")
            print(f"AC Prediction: {predictions['AC_vs_Other_GB']}, Probability: {probabilities['AC_vs_Other_GB']:.4f}")

    button.on_click(on_button_clicked)
    display(button, output)


    import pickle

    with open('model_results.pkl', 'rb') as file:
        data = pickle.load(file)

    results = data['results']
    outcome_distribution = data['outcome_distribution']
    classes = data['classes']

    # Print saved results
    for clf_name, result in results.items():
        print(f"{clf_name}: Mean accuracy = {result['mean_accuracy']:.4f}, Std = {result['std_accuracy']:.4f}")

    # Print saved outcome distribution
    print("\nClass value distribution of the outcome variable:")
    print("Class\tFrequency\tRatio")
    for cls, stats in outcome_distribution.items():
        print(f"{cls}\t{stats['count']}\t{stats['ratio']:.2f}%")

def main():
    train_model()
    save_sample_of_training_predictions()
    test_novel_datapoint()