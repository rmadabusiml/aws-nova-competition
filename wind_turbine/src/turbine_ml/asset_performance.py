import pandas as pd
import numpy as np
from datetime import datetime
import boto3
from src.turbine_ml.common_utils import prepare_prediction_data, upload_to_partitioned_s3
from src.helper.hydrate_db import load_asset_optimization_data_from_csv

# Load the data
catalog_df = pd.read_csv('data/turbine_catalog.csv')
new_data_df = pd.read_csv('data/new_turbine_data.csv')

# First function: Load model once
def load_prediction_model(model_path='data/output/power/power_model.pkl'):
    """Load the trained power prediction model once"""
    from src.turbine_ml.common_utils import load_model
    return load_model(model_path)

power_model = load_prediction_model('data/output/power/power_model.pkl')
life_model = load_prediction_model('data/output/remaining_life/remaining_life_model.pkl')

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

def predict_power_for_records(single_record):
    # Prepare data for prediction
    prepared_record = prepare_prediction_data(single_record)
    with_features = create_power_features(prepared_record)
    
    # Make prediction if features are available
    if len(with_features) > 0 and all(feature in with_features.columns for feature in power_model.feature_names_in_):
        prediction = power_model.predict(with_features[power_model.feature_names_in_])[0]
    else:
        prediction = None

    return prediction

def create_life_features(df):
    """Create features for remaining useful life prediction"""
    # Merge with maintenance data
    df = pd.merge(df, catalog_df[['turbine_id', 'last_maintenance', 'install_date']],
                  left_on='device_id', right_on='turbine_id')
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate days until next maintenance
    df['days_until_maintenance'] = (pd.to_datetime(df['last_maintenance']) - 
                                    df['timestamp']).dt.days
    
    # Equipment age
    df['age_days'] = (df['timestamp'] - 
                      pd.to_datetime(df['install_date'])).dt.days
    # Maintenance history
    df['days_since_last_maintenance'] = (df['timestamp'] - 
                                        pd.to_datetime(df['last_maintenance'])).dt.days
    
    return df.dropna()

def predict_life_for_records(single_record):
    # Prepare data for prediction
    prepared_record = prepare_prediction_data(single_record)
    with_features = create_life_features(prepared_record)
    
    # Make prediction if features are available
    if len(with_features) > 0 and all(feature in with_features.columns for feature in life_model.feature_names_in_):
        prediction = life_model.predict(with_features[life_model.feature_names_in_])[0]
    else:
        prediction = None

    return prediction

def optimize_turbine_performance(turbine_id, model_power, model_life, rpm_range=range(5, 15)):
    results = []
    matching_row = new_data_df[new_data_df['device_id'] == turbine_id]

    if not matching_row.empty:
        data_row = matching_row.iloc[0]
        
        base_payload = {
            'timestamp': data_row['timestamp'],
            'device_id': data_row['device_id'],
            'angle': data_row['angle'],
            'temperature': data_row['temperature'],
            'humidity': data_row['humidity'],
            'windspeed': data_row['windspeed'],
            'power': data_row['power'],
            'days_since_install': data_row['days_since_install'],
            'rpm_variance': data_row['rpm_variance'],
            'maintenance_flag': data_row['maintenance_flag']
        }

        for rpm in rpm_range:
            payload = base_payload.copy()
            payload['rpm'] = rpm
            # Convert payload to DataFrame for model prediction
            df_payload = pd.DataFrame([payload])
            predicted_power = predict_power_for_records(df_payload)
            predicted_life = predict_life_for_records(df_payload)
            results.append((rpm, predicted_power, predicted_life))

    return pd.DataFrame(results, columns=['RPM', 'Expected Power', 'Expected Life'])

def calculate_profit_metrics(prediction_df, electricity_price=100):
    df = prediction_df.copy()
    df['Revenue'] = electricity_price * df['Expected Power'] * 365
    df['Cost'] = (365 / df['Expected Life']) * electricity_price * df['Expected Power'] * 24
    df['Profit'] = df['Revenue'] - df['Cost']
    return df

if __name__ == "__main__":
    # Loop through each turbine and find optimal RPM and profit
    results_list = []
    for idx, row in catalog_df.iterrows():
        turbine_id = row['turbine_id']
        optimization_df = optimize_turbine_performance(turbine_id, power_model, life_model)
        # print(optimization_df)
        profit_df = calculate_profit_metrics(optimization_df)
        
        # Find optimal RPM
        optimal_idx = profit_df['Profit'].idxmax()
        optimal_row = profit_df.loc[optimal_idx]
        results_list.append({
            'turbine_id': turbine_id,
            'assessed_date': datetime.now().strftime('%Y-%m-%d'),
            'optimal_rpm': optimal_row['RPM'],
            'cost': optimal_row['Cost'],
            'revenue': optimal_row['Revenue'],
            'profit': optimal_row['Profit']
        })

    final_df = pd.DataFrame(results_list)
    print(final_df)

    csv_filename = f'data/output/asset_performance.csv'
    final_df.to_csv(csv_filename, index=False)

    load_asset_optimization_data_from_csv(csv_filename)

    upload_to_partitioned_s3(
        csv_filename,
        "wind_turbine/asset_performance"
    )
