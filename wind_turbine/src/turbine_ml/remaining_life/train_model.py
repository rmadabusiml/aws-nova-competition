from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold
from src.turbine_ml.common_utils import save_model, upload_to_partitioned_s3
import pandas as pd

def train_life_model(features_df):
    features = ['rpm', 'temperature', 'age_days', 
               'rpm_variance',
               'days_since_last_maintenance', 'humidity']
    
    X = features_df[features]
    y = features_df['days_until_maintenance']
    
    # Group by turbine to prevent data leakage
    groups = features_df['device_id']
    model = XGBRegressor(objective='reg:squarederror', 
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            early_stopping_rounds=20)
    
    cv = GroupKFold(n_splits=3)
    for train_idx, test_idx in cv.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)
    
    return model
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--features")
    parser.add_argument("--model_name")
    args = parser.parse_args()
    
    df = pd.read_csv(args.features)
    model = train_life_model(df)
    save_model(model, args.model_name)
    
    upload_to_partitioned_s3(
        args.model_name,
        "wind_turbine/remaining_life/models"
    )