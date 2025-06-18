from src.turbine_ml.common_utils import load_model, prepare_prediction_data, upload_to_partitioned_s3
from feature_engineering import create_life_features
import pandas as pd

# First function: Load model once
def load_prediction_model(model_path='data/output/remaining_life/remaining_life_model.pkl'):
    """Load the trained remaining life prediction model once"""
    from src.turbine_ml.common_utils import load_model
    return load_model(model_path)

def create_life_features(ts_df, catalog_df):
    """Create features for remaining useful life prediction"""
    # Merge with maintenance data
    df = pd.merge(ts_df, catalog_df[['turbine_id', 'last_maintenance', 'install_date']],
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

# Second function: Process predictions in a loop
def predict_remaining_life_for_records(new_data_path, model=None):
    """Process each turbine record individually for prediction"""

    catalog_df = pd.read_csv('data/turbine_catalog.csv')
    
    if model is None:
        model = load_prediction_model()
    
    raw_df = pd.read_csv(new_data_path)
    
    # Prepare container for results
    results = []
    
    # Process each record individually
    for index, row in raw_df.iterrows():
        # Convert single row to DataFrame
        single_record = pd.DataFrame([row])
        # print(single_record.head())
        
        # Prepare data for prediction
        prepared_record = prepare_prediction_data(single_record)
        
        with_features = create_life_features(prepared_record, catalog_df)
        
        # Make prediction if features are available
        if len(with_features) > 0 and all(feature in with_features.columns for feature in model.feature_names_in_):
            prediction = model.predict(with_features[model.feature_names_in_])[0]
            
            # Store result
            results.append({
                'device_id': row['device_id'],
                'timestamp': row['timestamp'],
                'remaining_life': prediction
            })
    
    # Convert results to DataFrame
    return pd.DataFrame(results)

model = load_prediction_model('data/output/remaining_life/remaining_life_model.pkl')
predictions = predict_remaining_life_for_records('data/new_turbine_data.csv', model)
predictions.to_csv('data/output/remaining_life/predictions.csv', index=False)