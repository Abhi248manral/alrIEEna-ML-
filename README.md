# ML Fault Detection Project

## Overview
This is a machine learning project for detecting faulty devices. We're working with a dataset that has 47 different features (F01 through F47) which are essentially measurements from an embedded detector. 

The goal is pretty straightforward: figure out if a device is working fine or if it's faulty. We're using a Random Forest model here to handle the binary classification.
- `0` means the device is Normal
- `1` means the device is Faulty

The script trains on the `TRAIN.csv` file, runs the model against `TEST.csv`, and dumps the final predictions into a new `FINAL.csv` file exactly how it's required.

## Model & Implementation Details
- **Algorithm**: Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`)
- **Hyperparameters**: 
  - `n_estimators=100`: Uses 100 decision trees in the forest to ensure stable predictions while avoiding excessive computation.
  - `random_state=0`: Ensures the splits are reproducible each time the model is trained.
  - `n_jobs=-1`: Utilizes all available CPU cores to speed up the training phase.
- **Why Random Forest?**: Random Forests are highly robust against overfitting and require minimal data preprocessing (like scaling or normalization). They naturally handle high-dimensional spaces (like our 47 features) and are excellent right out of the box for tabular sensor data.
- **Source Code Data Handling**: The prediction script uses Pandas (`pd.read_csv`) to load the datasets and strictly separates the feature columns from the `ID` and `Class` columns dynamically. 

## Setup
Install the requirements:
```bash
pip install pandas scikit-learn
```

Make sure your `TRAIN.csv` and `TEST.csv` files are right here in the main project folder before running anything.

## How to Run

Just execute the python script:
```bash
python generate_predictions.py
```

What the script does behind the scenes:
- Loads up both datasets
- Sorts out the features and target labels
- Trains a RandomForestClassifier using the training data
- Generates predictions for each row in the test set
- Writes everything out to `FINAL.csv` keeping just the `ID` and `CLASS` columns.

After it's done, you'll see `FINAL.csv` pop up in the folder.
