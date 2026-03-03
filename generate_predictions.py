import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os
import numpy as np

def main():
    train_csv = 'TRAIN1.csv' if os.path.exists('TRAIN1.csv') else 'TRAIN.csv'
    test_csv = 'TEST1.csv' if os.path.exists('TEST1.csv') else 'TEST.csv'

    temp_df = pd.read_csv(train_csv, nrows=0)
    target_col = 'Class' if 'Class' in temp_df.columns else 'CLASS'
    features = [c for c in temp_df.columns if c.lower() not in ['class', 'id']]

    dtypes_mapping = {f: np.float32 for f in features}
    dtypes_mapping[target_col] = np.int8

    train_df = pd.read_csv(train_csv, usecols=features + [target_col], dtype=dtypes_mapping)
    
    X_train = train_df[features]
    y_train = train_df[target_col]

    temp_test_df = pd.read_csv(test_csv, nrows=0)
    id_col = 'ID' if 'ID' in temp_test_df.columns else 'id'
    
    test_dtypes_mapping = {f: np.float32 for f in features}
    test_dtypes_mapping[id_col] = np.int32

    test_df = pd.read_csv(test_csv, usecols=features + [id_col], dtype=test_dtypes_mapping)

    X_test = test_df[features]
    test_ids = test_df[id_col]

    clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    clf.fit(X_train, y_train)

    out_df = pd.DataFrame({'ID': test_ids, 'CLASS': clf.predict(X_test)})
    out_df.to_csv('FINAL.csv', index=False)

if __name__ == '__main__':
    main()
