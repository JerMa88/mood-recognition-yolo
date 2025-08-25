# Mood 
Facial emotion detection using YOLO-V8

## Installation
1. Clone the repository:
   ```bash
   git clone
   cd mood-recognition-yolo
   ```
2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
4. Download the dataset from kaggle:
    ```bash
    curl -L -o ./data/train/face-expression-recognition-dataset.zip\
    https://www.kaggle.com/api/v1/datasets/download/jonathanoheix/face-expression-recognition-dataset
    unzip ./data/train/face-expression-recognition-dataset.zip -d ./
    ```
   Alternatively, you can download the dataset manually:

   - Create a Kaggle account if you don't have one.
   - Go to the dataset page: https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset
   - Click on "Download" to get the dataset.
   - Unzip the downloaded file and place it in the `images/` directory.

