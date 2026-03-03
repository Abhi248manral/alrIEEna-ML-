import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os

def main():
    train_csv = 'TRAIN1.csv' if os.path.exists('TRAIN1.csv') else 'TRAIN.csv'
    test_csv = 'TEST1.csv' if os.path.exists('TEST1.csv') else 'TEST.csv'

    # Load data
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # Label col
    target_col = 'Class' if 'Class' in train_df.columns else 'CLASS'

    # ID col in test
    id_col = 'ID' if 'ID' in test_df.columns else 'id'

    # Ensure ID doesn't get used as a feature if it's in train for some reason
    features = [c for c in train_df.columns if c.lower() != 'class' and c.lower() != 'id']

    X_train = train_df[features]
    y_train = train_df[target_col]

    X_test = test_df[features]
    test_ids = test_df[id_col]

    # Model definition
    # Using RandomForest as directly suggested
    clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)

    # Train
    clf.fit(X_train, y_train)

    # Predict
    predictions = clf.predict(X_test)

    # Save
    out_df = pd.DataFrame({
        'ID': test_ids,
        'CLASS': predictions
    })
    
    out_df.to_csv('FINAL.csv', index=False)

if __name__ == '__main__':
    main()
