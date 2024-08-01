# Screening Master

This project is part of my master's program.

## Running Instructions

1. **Open a terminal**:
2. **Change directory into the repository folder**:

   ```sh
   cd screening-master
   ```

3. **Create a virtual environment**:

   ```sh
   python -m venv venv
   ```

4. **Activate the virtual environment**:

   ```sh
   venv/Scripts/activate
   ```

5. **Install the necessary packages**:

   ```sh
   pip install -r requirements.txt
   ```

6. **Run the backend**:

   - If it's the first time running, ensure the `train_models` function line is not commented out at the bottom of the `Screening.py` file.
   - After the models have been trained once, you can comment out that line again.

   ```sh
   python backend/Screening.py
   ```

7. **To quit and shutdown the server**:
   Press `CTRL+C`

8. **To exit the virtual environment**:
   ```sh
   deactivate
   ```
