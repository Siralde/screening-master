# Screening Master

This project is part of my master's program.

## Running Instructions

1. **Open a terminal**:
2. **Change directory into the repository folder**:

<<<<<<< HEAD
   ```sh
   cd screening-master
   ```

3. **Create a virtual environment**:
=======
  - Change directory into the repository
  - e.g. "cd screening-master"
>>>>>>> master

   ```sh
   python -m venv venv
   ```

<<<<<<< HEAD
4. **Activate the virtual environment**:
=======
  - "python -m venv venv"
>>>>>>> master

   ```sh
   venv/Scripts/activate
   ```

<<<<<<< HEAD
5. **Install the necessary packages**:
=======
  - "venv/Scripts/activate"
>>>>>>> master

   ```sh
   pip install -r requirements.txt
   ```

<<<<<<< HEAD
6. **Run the backend**:

   - If it's the first time running the models will take time to train

   ```sh
   python backend/Screening.py
   ```
=======
  - "pip install -r requirements.txt"

- Finally, run the server:

  - "python Screening.py"
  - The server will be running on: http://192.168.68.157:8080/
  - If not, check the terminal if the address was updated.
>>>>>>> master

   - Otherwise run the following to make the server run quicker

<<<<<<< HEAD
   ```sh
   flask run
   ```

7. **To quit and shutdown the server**:
   Press `CTRL+C`

8. **To exit the virtual environment**:
   ```sh
   deactivate
   ```
=======
- type 'deactivate' in terminal to exit the virtual environment
>>>>>>> master
