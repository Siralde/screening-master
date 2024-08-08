# Screening Master

This project is part of my master's program.

## Running Instructions

1. **Open a terminal**:

2. **Clone the repository**:

   ```sh
   git clone <repository-link>
   ```

3. **Change directory into the repository folder**:

   ```sh
   cd screening-master
   ```

4. **Fetch the LFS stored models**

   ```sh
   git lfs pull
   ```

5. **Create a virtual environment**:

   ```sh
   python -m venv venv
   ```

6. **Activate the virtual environment**:

   ```sh
   venv/Scripts/activate
   ```

7. **Install the necessary packages**:

   ```sh
   pip install -r requirements.txt
   ```

8. **Run the backend**:

   - If it's the first time running the models will take time to train

   ```sh
   python backend/Screening.py
   ```

   - Otherwise run the following to make the server run quicker

   ```sh
   flask run
   ```

9. **To quit and shutdown the server**:
   Press `CTRL+C`

10. **To exit the virtual environment**:

```sh
deactivate
```
