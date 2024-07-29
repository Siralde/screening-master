# screening-master

Screening project of my master

## Running Instructions:

- Open a terminal:

- Change directory into the repository folder:

  - cd screening-master

- Create a virtual environment:

  - python -m venv venv

- Run the virtual environment:

  - venv/Scripts/activate

- Then install necessary packages:

  - pip install -r requirements.txt

- Finally, run the backend:

  - If it's the first time running, make sure the train_models function line is not commented out at the bottom of the Screening.py file.
  - After it's been run once and the models have been trained, the line can be commented out once more.
  - python backend/Screening.py

- Press CTRL+C to quit and shutdown the server

- type 'deactivate' in terminal to exit the virtual environment
