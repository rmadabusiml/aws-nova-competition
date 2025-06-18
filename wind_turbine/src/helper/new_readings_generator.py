import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from turbine_data_generator import generate_turbine_data
from src.turbine_ml.common_utils import upload_to_partitioned_s3

# Load your catalog
turbine_catalog = pd.read_csv('data/turbine_catalog.csv')

# Generate time-series data
ts_data = generate_turbine_data(turbine_catalog, start_date='2025-01-01', end_date='2025-01-01')
ts_data.to_csv('data/new_turbine_data.csv', index=False)

upload_to_partitioned_s3(
    'data/new_turbine_data.csv',
    "wind_turbine/new_readings"
)