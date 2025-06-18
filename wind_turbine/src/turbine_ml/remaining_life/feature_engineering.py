import pandas as pd
from src.turbine_ml.common_utils import upload_to_partitioned_s3

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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data")
    parser.add_argument("--output_file")
    args = parser.parse_args()
    
    df = pd.read_csv(args.input_data)
    catalog_df = pd.read_csv('data/turbine_catalog.csv')
    processed_df = create_life_features(df, catalog_df)
    processed_df.to_csv(args.output_file, index=False)

    print("DONE feature engineering")
    
    upload_to_partitioned_s3(
        args.output_file,
        "wind_turbine/remaining_life/feature_engineering"
    )