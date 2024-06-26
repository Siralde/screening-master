import pytest
from fastapi.testclient import TestClient
from main import app
import os
import pandas as pd
import io
import pickle
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = TestClient(app)

# Path to your pre-existing .gz file
GZ_FILE_PATH = "data/CB_data_folder.gz"

@pytest.fixture
def gz_file():
    assert os.path.exists(GZ_FILE_PATH), f"Test .gz file not found at {GZ_FILE_PATH}"
    return GZ_FILE_PATH

def test_model_training():
    # Path to the local CSV file
    csv_file_path = "data/small_CB_data.csv"
    
    # Open the CSV file and send it to the endpoint
    with open(csv_file_path, 'rb') as csv_file:
        response = client.post("/get-model/", files={"file": ("sample.csv", csv_file, "text/csv")})
    
    # Check the response status code
    assert response.status_code == 200

    # Save the response content to a file-like object to test its contents
    pkl_buffer = io.BytesIO(response.content)

    # Load the pickle content to check if it's correct
    data = pickle.load(pkl_buffer)

    # Extract results from the pickled data
    results = data['results']
    outcome_distribution = data['outcome_distribution']
    classes = data['classes']

    # Print saved results
    for clf_name, result in results.items():
        logging.info(f"{clf_name}: Mean accuracy = {result['mean_accuracy']:.4f}, Std = {result['std_accuracy']:.4f}")

    # Print saved outcome distribution
    logging.info("\nClass value distribution of the outcome variable:")
    logging.info("Class\tFrequency\tRatio")
    for cls, stats in outcome_distribution.items():
        logging.info(f"{cls}\t{stats['count']}\t{stats['ratio']:.2f}%")

  
