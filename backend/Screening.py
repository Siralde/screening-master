import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flasgger import Swagger, swag_from
import json
from pickle import load

# Local Imports
from functions.models import train_model, analyze_numerical_features

base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, 'data/csvs')
pkl_path = os.path.join(base_path, 'data/pkls')
template_path = os.path.join(base_path, '../frontend/templates')
static_path = os.path.join(base_path, '../frontend/static')
csv_path = os.path.join(base_path, 'data/csvs/unique_filtered_final_with_target_variable.csv')

df = None # pd.read_csv(csv_path)

swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "My API",
        "description": "API for predicting company success rates.",
        "version": "1.0.0",
    },
    "host": "screening-master-3.onrender.com", 
    "basePath": "/", 
    "schemes": [
        "https", 
    ],
    "externalDocs": {
        "description": "Find more info here",
        "url": "https://screening-master.apidocumentation.com/"
    }
}


app = Flask(__name__, template_folder=template_path, static_folder=static_path)
swagger = Swagger(app, template=swagger_template)

# List of required files
required_files = [
    'final_model.pkl',
    'label_encoders.pkl',
    'column_names.pkl',
    'target_encoder.pkl'
]

missing_files = [file for file in required_files if not os.path.exists(os.path.join(pkl_path, file))]

if missing_files:
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"ERROR: The following required files are missing: {', '.join(missing_files)}\n Check the README.md for how to run the server in a way that generates the necessary files")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    
    quit
else:
    # Load the files if all are present
    with open(os.path.join(pkl_path, 'final_model.pkl'), 'rb') as file:
        classifier = load(file)
    with open(os.path.join(pkl_path, 'label_encoders.pkl'), 'rb') as file:
        encoders = load(file)
    with open(os.path.join(pkl_path, 'column_names.pkl'), 'rb') as file:
        column_names = load(file)
    with open(os.path.join(pkl_path, 'target_encoder.pkl'), 'rb') as file:
        target_encoder = load(file)

### Routes:

# Home Page
@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")

# Predictions page endpoint
@app.route("/predict", methods=["GET"])
@swag_from('yml_files/predict_get.yml')
def predict_get():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
@swag_from('yml_files/predict_post.yml')
def predict():
    if request.method == "POST":
        try:
            new_company_info = {
                'country_code': request.form['company_country_code'],
                'region': request.form['company_region'],
                'city': request.form['company_city'],
                'category_list': request.form['company_category_list'],
                'last_round_investment_type': request.form['company_last_round_investment_type'],
                'num_funding_rounds': int(request.form['company_num_funding_rounds']),
                'total_funding_usd': float(request.form['company_total_funding_usd'].replace(',', '')),
                'age_months': int(request.form['company_age_months']),
                'has_facebook_url': int(request.form.get('company_has_facebook_url', 0)),
                'has_twitter_url': int(request.form.get('company_has_twitter_url', 0)),
                'has_linkedin_url': int(request.form.get('company_has_linkedin_url', 0)),
                'round_count': int(request.form['company_round_count']),
                'raised_amount_usd': float(request.form['company_raised_amount_usd'].replace(',', '')),
                'last_round_raised_amount_usd': float(request.form['company_last_round_raised_amount_usd'].replace(',', '')),
                'last_round_post_money_valuation': float(request.form['company_last_round_post_money_valuation'].replace(',', '')),
                'last_round_timelapse_months': int(request.form['company_last_round_timelapse_months']),
                'last_round_investor_count': int(request.form['company_last_round_investor_count']),
                'founders_dif_country_count': int(request.form['company_founders_dif_country_count']),
                'founders_male_count': int(request.form['company_founders_male_count']),
                'founders_female_count': int(request.form['company_founders_female_count']),
                'founders_degree_count_total': int(request.form['company_founders_degree_count_total']),
                'founders_degree_count_max': int(request.form['company_founders_degree_count_max'])
            }

            def encode_and_handle_unseen(column, value):
                encoder = encoders[column]
                if value not in encoder.classes_:
                    encoder.classes_ = np.append(encoder.classes_, value)
                return encoder.transform([value])[0]

            new_company_df = pd.DataFrame([new_company_info])

            categorical_columns = [
                'country_code', 'region', 'city', 'category_list',
                'last_round_investment_type'
            ]
            for col in categorical_columns:
                new_company_df[col] = new_company_df[col].apply(lambda x: encode_and_handle_unseen(col, x))

            new_company_df = new_company_df.reindex(columns=column_names, fill_value=0)
            
            prediction = int(classifier.predict(new_company_df)[0])
            confidence = float(classifier.predict_proba(new_company_df)[:, 1][0])
            confidence = confidence * 100
            if prediction == 0:
                confidence = 100 - confidence
            prediction_name = "Closed/No Event" if prediction == 0 else "Funding Round/Acquisition/IPO"

            results = {
                "Prediction": prediction_name,
                "Confidence": f"{confidence:.2f}"
            }

            return jsonify(results)

        except Exception as e:
            print(f"An error occurred: {e}")
            return jsonify(error=str(e))

    return render_template("index.html")

@app.route('/search_companies', methods=['GET'])
@swag_from({
    'responses': {
        200: {
            'description': 'A list of companies matching the search string',
            'schema': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'company_name': {'type': 'string'},
                        'other_column': {'type': 'integer'}
                    }
                }
            }
        }
    },
    'parameters': [
        {
            'name': 'company_name',
            'in': 'query',
            'type': 'string',
            'required': True,
            'description': 'The name of the company to search for'
        }
    ],
    'tags': ['Company Search']
})
def search_companies():
    search_string = request.args.get('company_name', '').lower()
    if not search_string:
        return render_template('search_companies.html', results=[])

    filtered_df = df[df['name_org'].str.contains(search_string, case=False, na=False)]

    excluded_features = [
        'uuid_org', 'permalink_org', 'domain', 'homepage_url', 
        'address', 'postal_code', 'short_description', 'facebook_url', 
        'linkedin_url', 'twitter_url', 'founded_on', 'last_funding_on', 
        'closed_on', 'total_funding_currency_code', 'outcome', 'state_code', 
        'status', 'total_funding', 'category_groups_list', 'founders_degree_count_mean'
    ]
    X = filtered_df.drop(columns=excluded_features)
    print(X.columns)
    result = X.to_dict(orient='records')

    return render_template('search_companies.html', results=result) 

@app.route('/openapi.json')
def get_openapi_spec():
    with open('openapi.json') as json_file:
        return json.load(json_file)

def main():
    print("Main function")
    file_path = os.path.join(pkl_path, 'final_model.pkl')
    if not os.path.exists(file_path):
        if not os.path.exists(os.path.join(data_path, 'unique_filtered_final_with_target_variable.csv')):
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"CSV named unique_filtered_final_with_target_variable.csv containing CrunchBase Data is missing from csvs folder")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        else:
            data = pd.read_csv(os.path.join(data_path, 'unique_filtered_final_with_target_variable.csv'))
            print("Training Models and populating pkls folder.")
            train_model(data=data)
    # Uncomment the below line if you desire extra details about the model performance.
    #analyze_numerical_features()




if __name__ == "__main__":
    main()

    # Stores the OpenAPI for Scalar
    with app.app_context():
        openapi_spec = swagger.get_apispecs()
        with open('openapi.json', 'w') as json_file:
            json.dump(openapi_spec, json_file, indent=2)
    
    
    print("Starting Flask app")
    app.run(debug=True)
    
