
import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
from pickle import load
from fastapi.templating import Jinja2Templates
import json
from pathlib import Path
from datetime import datetime, timedelta
from passlib.context import CryptContext
from jose import JWTError, jwt
from dotenv import load_dotenv


# Local Imports
from functions.models import train_model, analyze_numerical_features

# Load environment variables from .env file
load_dotenv()


base_path = Path(__file__).resolve().parent
data_path = base_path / 'data/csvs'
pkl_path = base_path / 'data/pkls'
template_path = base_path / '../frontend/templates'
static_path = base_path / '../frontend/static'
csv_path = base_path / 'data/csvs/unique_filtered_final_with_target_variable.csv'

app = FastAPI(
    title="My API",
    description="API for predicting company success rates.",
    version="1.0.0"
)

# Mounting static files
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")



# Load the files if all are present
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

# Secret key for JWT
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 token URL
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Dummy user data
fake_users_db = {
    "testuser": {
        "username": "testuser",
        "full_name": "Test User",
        "email": "testuser@example.com",
        "hashed_password": pwd_context.hash("testpassword"),
        "disabled": False,
    }
}

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

def authenticate_user(db, username: str, password: str):
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}




# Assume the existing FastAPI app and configurations are in place

df = None # call pd.read_csv() on the file with the company data

# Path Setup
template_path = Path(__file__).resolve().parent / '../frontend/templates'
static_path = Path(__file__).resolve().parent / '../frontend/static'
pkl_path = Path(__file__).resolve().parent / 'data/pkls'

templates = Jinja2Templates(directory=str(template_path))

# Assuming the model and other necessary variables are already loaded as shown before.

# Home Page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

# Predictions page endpoint
@app.get("/predict", response_class=HTMLResponse)
async def predict_get(request: Request, current_user: dict = Depends(get_current_active_user)):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=JSONResponse)
async def predict(
    current_user: dict = Depends(get_current_active_user),
    company_country_code: str = Form(...),
    company_region: str = Form(...),
    company_city: str = Form(...),
    company_category_list: str = Form(...),
    company_last_round_investment_type: str = Form(...),
    company_num_funding_rounds: int = Form(...),
    company_total_funding_usd: float = Form(...),
    company_age_months: int = Form(...),
    company_has_facebook_url: int = Form(0),
    company_has_twitter_url: int = Form(0),
    company_has_linkedin_url: int = Form(0),
    company_round_count: int = Form(...),
    company_raised_amount_usd: float = Form(...),
    company_last_round_raised_amount_usd: float = Form(...),
    company_last_round_post_money_valuation: float = Form(...),
    company_last_round_timelapse_months: int = Form(...),
    company_last_round_investor_count: int = Form(...),
    company_founders_dif_country_count: int = Form(...),
    company_founders_male_count: int = Form(...),
    company_founders_female_count: int = Form(...),
    company_founders_degree_count_total: int = Form(...),
    company_founders_degree_count_max: int = Form(...)
):
    try:
        new_company_info = {
            'country_code': company_country_code,
            'region': company_region,
            'city': company_city,
            'category_list': company_category_list,
            'last_round_investment_type': company_last_round_investment_type,
            'num_funding_rounds': company_num_funding_rounds,
            'total_funding_usd': company_total_funding_usd,
            'age_months': company_age_months,
            'has_facebook_url': company_has_facebook_url,
            'has_twitter_url': company_has_twitter_url,
            'has_linkedin_url': company_has_linkedin_url,
            'round_count': company_round_count,
            'raised_amount_usd': company_raised_amount_usd,
            'last_round_raised_amount_usd': company_last_round_raised_amount_usd,
            'last_round_post_money_valuation': company_last_round_post_money_valuation,
            'last_round_timelapse_months': company_last_round_timelapse_months,
            'last_round_investor_count': company_last_round_investor_count,
            'founders_dif_country_count': company_founders_dif_country_count,
            'founders_male_count': company_founders_male_count,
            'founders_female_count': company_founders_female_count,
            'founders_degree_count_total': company_founders_degree_count_total,
            'founders_degree_count_max': company_founders_degree_count_max
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

        return JSONResponse(content=results)

    except Exception as e:
        print(f"An error occurred: {e}")
        return JSONResponse(content={"error": str(e)})

# Search Companies
@app.get("/search_companies", response_class=HTMLResponse)
async def search_companies(
                           request: Request,
                            company_name: str,
                            current_user: dict = Depends(get_current_active_user)):
    search_string = company_name.lower()
    if not search_string:
        return templates.TemplateResponse('search_companies.html', {"request": request, "results": []})

    filtered_df = df[df['name_org'].str.contains(search_string, case=False, na=False)]

    excluded_features = [
        'uuid_org', 'permalink_org', 'domain', 'homepage_url', 
        'address', 'postal_code', 'short_description', 'facebook_url', 
        'linkedin_url', 'twitter_url', 'founded_on', 'last_funding_on', 
        'closed_on', 'total_funding_currency_code', 'outcome', 'state_code', 
        'status', 'total_funding', 'category_groups_list', 'founders_degree_count_mean'
    ]
    X = filtered_df.drop(columns=excluded_features)
    result = X.to_dict(orient='records')

    return templates.TemplateResponse('search_companies.html', {"request": request, "results": result})

# Serve OpenAPI Specification
@app.get('/openapi.json', response_class=JSONResponse)
async def get_openapi_spec(current_user: dict = Depends(get_current_active_user)):
    with open('openapi.json') as json_file:
        return JSONResponse(content=json.load(json_file))



if __name__ == "__main__":
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

    