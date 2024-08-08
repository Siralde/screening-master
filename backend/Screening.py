import os
import numpy as np
import pandas as pd
import pickle
from flask import Flask, render_template, request, jsonify
from flasgger import Swagger, swag_from
import json
import psutil

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 ** 2  

initial_memory = get_memory_usage()

# Local Imports
from functions.models import train_model, analyze_numerical_features

base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, 'data/csvs')
pkl_path = os.path.join(base_path, 'data/pkls')
template_path = os.path.join(base_path, '../frontend/templates')
csv_path = os.path.join(base_path, 'data/csvs/unique_filtered_final_with_target_variable.csv')

#df = pd.read_csv(csv_path)

app = Flask(__name__, template_folder=template_path)
swagger = Swagger(app)

### Routes:

@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")

@app.route("/predict", methods=["GET", "POST"])
@swag_from({
    'tags': ['Prediction Endpoints'],
    'description': 'Enter company information and receive a rating describing its ',
    'parameters': [
        {
            'name': 'company_country_code',
            'in': 'formData',
            'type': 'string',
            'required': True
        },
        {
            'name': 'company_region',
            'in': 'formData',
            'type': 'string',
            'required': True
        },
        {
            'name': 'company_city',
            'in': 'formData',
            'type': 'string',
            'required': True
        },
        {
            'name': 'company_category_list',
            'in': 'formData',
            'type': 'string',
            'required': True
        },
        {
            'name': 'company_last_round_investment_type',
            'in': 'formData',
            'type': 'string',
            'required': True
        },
        {
            'name': 'company_num_funding_rounds',
            'in': 'formData',
            'type': 'integer',
            'required': True
        },
        {
            'name': 'company_total_funding_usd',
            'in': 'formData',
            'type': 'number',
            'required': True
        },
        {
            'name': 'company_age_months',
            'in': 'formData',
            'type': 'integer',
            'required': True
        },
        {
            'name': 'company_has_facebook_url',
            'in': 'formData',
            'type': 'integer',
            'required': False
        },
        {
            'name': 'company_has_twitter_url',
            'in': 'formData',
            'type': 'integer',
            'required': False
        },
        {
            'name': 'company_has_linkedin_url',
            'in': 'formData',
            'type': 'integer',
            'required': False
        },
        {
            'name': 'company_round_count',
            'in': 'formData',
            'type': 'integer',
            'required': True
        },
        {
            'name': 'company_raised_amount_usd',
            'in': 'formData',
            'type': 'number',
            'required': True
        },
        {
            'name': 'company_last_round_raised_amount_usd',
            'in': 'formData',
            'type': 'number',
            'required': True
        },
        {
            'name': 'company_last_round_post_money_valuation',
            'in': 'formData',
            'type': 'number',
            'required': True
        },
        {
            'name': 'company_last_round_timelapse_months',
            'in': 'formData',
            'type': 'integer',
            'required': True
        },
        {
            'name': 'company_last_round_investor_count',
            'in': 'formData',
            'type': 'integer',
            'required': True
        },
        {
            'name': 'company_founders_dif_country_count',
            'in': 'formData',
            'type': 'integer',
            'required': True
        },
        {
            'name': 'company_founders_male_count',
            'in': 'formData',
            'type': 'integer',
            'required': True
        },
        {
            'name': 'company_founders_female_count',
            'in': 'formData',
            'type': 'integer',
            'required': True
        },
        {
            'name': 'company_founders_degree_count_total',
            'in': 'formData',
            'type': 'integer',
            'required': True
        },
        {
            'name': 'company_founders_degree_count_max',
            'in': 'formData',
            'type': 'integer',
            'required': True
        }
    ],
    'responses': {
        '200': {
            'description': 'Prediction result',
            'schema': {
                'type': 'object',
                'properties': {
                    'Prediction': {
                        'type': 'string'
                    },
                    'Confidence': {
                        'type': 'string'
                    }
                },
                'example': {
                    'Prediction': 'Funding Round/Acquisition/IPO',
                    'Confidence': '85.00'
                }
            }
        },
        '400': {
            'description': 'Bad Request',
            'schema': {
                'type': 'object',
                'properties': {
                    'error': {
                        'type': 'string'
                    }
                },
                'example': {
                    'error': 'Invalid input data'
                }
            }
        }
    }
})
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

            print("New company info collected:")
            print(new_company_info)

            with open(os.path.join(pkl_path, 'final_model.pkl'), 'rb') as file:
                classifier = pickle.load(file)
                print("Classifier loaded")
            
            with open(os.path.join(pkl_path, 'label_encoders.pkl'), 'rb') as file:
                encoders = pickle.load(file)
                print("Encoders loaded")
            
            with open(os.path.join(pkl_path, 'column_names.pkl'), 'rb') as file:
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
            confidence = confidence * 100
            if prediction == 0:
                confidence = 100 - confidence
            prediction_name = "Closed/No Event" if prediction == 0 else "Funding Round/Acquisition/IPO"
            print(f"Prediction: {prediction_name}")
            print(f"Confidence: {confidence:.2f}%")

            results = {
                "Prediction": prediction_name,
                "Confidence": f"{confidence:.2f}"
            }

            print("Results calculated")
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


# Function to get memory usage
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 ** 2  



def main():

    print("Main function")
    #data = pd.read_csv(os.path.join(data_path, 'unique_filtered_final_with_target_variable.csv'))
    
    file_path = os.path.join(pkl_path, 'final_model.pkl')
    # if not os.path.exists(file_path):
    #     train_model(data=data)
    
    #data_column_names = data.columns
    print("Data loaded")
    #analyze_numerical_features()




if __name__ == "__main__":
    main()
    with app.app_context():
        openapi_spec = swagger.get_apispecs()
        with open('openapi.json', 'w') as json_file:
            json.dump(openapi_spec, json_file, indent=2)
    print("Starting Flask app")
    final_memory = get_memory_usage()
    print(f"Memory usage before: {initial_memory} MB")
    print(f"Memory usage after: {final_memory} MB")
    print(f"Memory used by code: {final_memory - initial_memory} MB")
    app.run(debug=True, host='0.0.0.0', port=8000, use_reloader=True)
    
