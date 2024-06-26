# screening-master

Screening project of my master

## Running Instructions:

- Open a terminal:

- Change directory into the src folder:

  - cd src

- Create a virtual environment:

  - python -m venv .venv

- Run the virtual environment:

  - .venv/Scripts/activate

- Then install necessary packages:

  - pip install -r ../requirements.txt

- Finally, run the backend:

  - uvicorn main:app (add "--reload" in order to enable development mode)

- Press CTRL+C to quit and shutdown the server

- type 'deactivate' in terminal to exit the virtual environment

## Running tests

- instead of running "uvicorn main:app" instead run "pytest test.py"
