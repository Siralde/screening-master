from fastapi import FastAPI, HTTPException, UploadFile, File # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore # For Secure Access
from fastapi.responses import StreamingResponse, FileResponse
import logging
import os
import gzip
import shutil
from datetime import datetime
import tarfile
import pandas as pd
import io
import pickle
# Local Imports
import functions.data_cleaning as Clean
import functions.model_results as Models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Start App

app = FastAPI()


# CORS - Allowed Origins

origins = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:4173",
    "http://localhost:3000",
]

app.add_middleware(CORSMiddleware,
                   allow_origins=origins,
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"],
                   )


@app.post(
    "/get-data-and-model/", 
    summary="Uploads data and retrieve a trained model",
    description="""
    Given a compressed archive folder (.gz) from CrunchBase, this endpoint will
    go through the data, clean it, and use it to train a model. It will then save
    the results of the training locally.
    """ ,
    tags=["data-cleaning-model-training"],
    deprecated=False,
    operation_id="clean_data_and_train_model",
    response_description="A file containing the results of the model training",
)
async def get_data_and_model(folder: UploadFile = File(...)):
    if not folder.filename.endswith(".gz"):
        return "Invalid file format. Please upload a .gz folder."
    
    # Saving and extracting the data:

    try:
        # Save the uploaded file
        upload_path = f"/tmp/{folder.filename}"
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(folder.file, buffer)

        # Decompress the .gz file to get the .tar file
        tar_path = upload_path.replace(".gz", "")
        with gzip.open(upload_path, "rb") as f_in:
            with open(tar_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Remove the original .gz file
        os.remove(upload_path)

        # Extract the .tar file
        extracted_folder = tar_path.replace(".tar", "")
        os.makedirs(extracted_folder, exist_ok=True)
        
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=extracted_folder)

        # Remove the .tar file
        os.remove(tar_path)

        logger.info(f"File uploaded and extracted successfully extracted_path: {extracted_folder}")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the file: {str(e)}")

    # Cleaning the data:

    try:

        csv_files = {f: os.path.join(extracted_folder, f) for f in os.listdir(extracted_folder) if f.endswith(".csv")}

        # Ensure that the required CSV files are present
        required_files = [
                            "organizations.csv",
                            "funding_rounds.csv",
                            "acquisitions.csv",
                            "ipos.csv",
                            "investments.csv",
                            "people.csv",
                            "degrees.csv",
                        ]
        
        for required_file in required_files:
            if required_file not in csv_files:
                raise HTTPException(status_code=400, detail=f"Missing required file: {required_file}")

        logger.info("All required files are present")

        dataframe = Clean.clean_data(organization_path=csv_files["organizations.csv"], 
                                    funding_rounds_path=csv_files["funding_rounds.csv"],
                                    acquisitions_path=csv_files["acquisitions.csv"],
                                    ipos_path=csv_files["ipos.csv"],
                                    investments_path=csv_files["investments.csv"],
                                    people_path=csv_files["people.csv"],
                                    degrees_path=csv_files["degrees.csv"],
                                    start_date=None, 
                                    end_date=None,
                                    sim_start_date=None,
                                    sim_end_date=None)
        
        logger.info("Data is cleaned")

        if dataframe.empty:
            logger.error(f"Error: Cleaned Dataset is empty")

        results = Models.train_models(dataframe)

        # Create a unique filename for the result file
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"model_results_{current_time}.pkl"

        headers = {
            'Content-Disposition': f'attachment; filename="{result_filename}"'
        }

        # Return the file as a response
        logging.info("Returning results")
        return FileResponse(results, 
                            media_type='application/octet-stream', 
                            filename=result_filename, 
                            headers=headers)
    
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=501, 
                            detail=f"An error occurred while cleaning the data: {str(e)}")


@app.post(
    "/get-model/", 
    summary="Trains and retrieves a model given clean data",
    description="""
    Given a clean data csv, this endpoint will use it to train a model. It will then save
    the results of the training locally.
    """ ,
    tags=["model-training"],
    deprecated=False,
    operation_id="train_model",
    response_description="A string containing the status of the operation and possible errors",
)
async def get_model(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        return "Invalid file format. Please upload a .csv file."
    
    # Saving and extracting the data:

    

    try:
        # Read the uploaded CSV file into a DataFrame
        contents = await file.read()
        dataframe = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # Process the data and train models
        results = Models.train_models(dataframe)

        # Create a unique filename for the result file
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"model_results_{current_time}.pkl"

        headers = {
            'Content-Disposition': f'attachment; filename="{result_filename}"'
        }
        
        logging.info("Returning results")
        
        # Create an io.BytesIO buffer
        buffer = io.BytesIO()

        # Serialize the dictionary using pickle
        pickle.dump(results, buffer)

        # Ensure the buffer's position is at the beginning
        buffer.seek(0)

        # Create a unique filename for the result file
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"model_results_{current_time}.pkl"


        # Return the file as a response
        headers = {
            'Content-Disposition': f'attachment; filename="{result_filename}"'
        }

        # Return the file as a streaming response
        return StreamingResponse(buffer, 
                                 media_type='application/octet-stream', 
                                 headers=headers)
    
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=501, 
                            detail=f"An error occurred while cleaning the data: {str(e)}")