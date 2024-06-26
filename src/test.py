import pytest
from fastapi.testclient import TestClient
from main import app
import os


client = TestClient(app)

# Path to your pre-existing .gz file
GZ_FILE_PATH = "data/CB_data_folder.gz"

@pytest.fixture
def gz_file():
    assert os.path.exists(GZ_FILE_PATH), f"Test .gz file not found at {GZ_FILE_PATH}"
    return GZ_FILE_PATH



  
