from src.turbine_ml.common_utils import load_model, prepare_prediction_data
from src.turbine_ml.common_utils import upload_to_partitioned_s3
import pandas as pd
import argparse

# First function: Load model once
def load_prediction_model(model_path='data/output/power/power_model.pkl'):
    """Load the trained power prediction model once"""
    return load_model(model_path)

def create_power_features(df):
    """Create time-series features for power prediction"""
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['device_id', 'timestamp'])
    
    # Lag features
    for lag in [1, 2, 3, 6, 12]:
        df[f'power_lag_{lag}'] = df.groupby('device_id')['power'].shift(lag)
        df[f'rpm_lag_{lag}'] = df.groupby('device_id')['rpm'].shift(lag)
    
    # Rolling features
    df['power_rolling_avg_6h'] = df.groupby('device_id')['power'].transform(
        lambda x: x.rolling(6, min_periods=1).mean()
    )
    
    # Time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    return df

# Second function: Process predictions in a loop
def predict_power_for_records(new_data_path, model=None):
    """Process each turbine record individually for prediction"""
    
    # Load model if not provided
    if model is None:
        model = load_prediction_model()
    
    # Read CSV file
    raw_df = pd.read_csv(new_data_path)
    # print(raw_df.head())
    
    # Prepare container for results
    results = []
    
    # Process each record individually
    for index, row in raw_df.iterrows():
        # Convert single row to DataFrame
        single_record = pd.DataFrame([row])
        
        # Prepare data for prediction
        prepared_record = prepare_prediction_data(single_record)
        with_features = create_power_features(prepared_record)
        
        # Make prediction if features are available
        if len(with_features) > 0 and all(feature in with_features.columns for feature in model.feature_names_in_):
            prediction = model.predict(with_features[model.feature_names_in_])[0]
            
            # Store result
            results.append({
                'device_id': row['device_id'],
                'timestamp': row['timestamp'],
                'target_power': prediction
            })
    
    # Convert results to DataFrame
    return pd.DataFrame(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--new_data")
    parser.add_argument("--model")
    parser.add_argument("--output")
    args = parser.parse_args()
    
    model = load_prediction_model(args.model)
    predictions = predict_power_for_records(args.new_data, model)
    predictions.to_csv(args.output, index=False)
    
    upload_to_partitioned_s3(
        args.output,
        "wind_turbine/power_prediction/predictions"
    )

