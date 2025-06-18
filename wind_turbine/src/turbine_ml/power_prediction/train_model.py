import argparse
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from src.turbine_ml.common_utils import save_model, upload_to_partitioned_s3

def train_power_model(features_df):
    features = ['rpm', 'angle', 'temperature', 'humidity', 'windspeed',
                'power_lag_1', 'power_lag_6', 'rpm_lag_1', 
                'power_rolling_avg_6h', 'hour', 'day_of_week']
    
    X = features_df[features]
    y = features_df['target_power']
    
    # Time-based cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        early_stopping_rounds=20
    )
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features")
    parser.add_argument("--model_name")
    args = parser.parse_args()
    
    df = pd.read_csv(args.features)
    model = train_power_model(df)
    save_model(model, args.model_name)
    
    upload_to_partitioned_s3(
        args.model_name,
        "wind_turbine/power_prediction/models"
    )
