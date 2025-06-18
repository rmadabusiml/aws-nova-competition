import pandas as pd
from src.turbine_ml.common_utils import upload_to_partitioned_s3

def create_power_features(df, horizon=6):
    """Create time-series features for power prediction"""
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['device_id', 'timestamp'])
    
    # Create target (6 hours ahead)
    df['target_power'] = df.groupby('device_id')['power'].shift(-horizon)
    
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
    
    # Filter out incomplete final window
    df = df[df['target_power'].notna()]
    
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data")
    parser.add_argument("--output_file")
    args = parser.parse_args()
    
    df = pd.read_csv(args.input_data)
    processed_df = create_power_features(df)
    processed_df.to_csv(args.output_file, index=False)
    
    upload_to_partitioned_s3(
        args.output_file,
        "wind_turbine/power_prediction/feature_engineering"
    )
