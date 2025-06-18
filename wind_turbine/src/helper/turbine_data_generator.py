import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import boto3
from io import BytesIO
from src.turbine_ml.common_utils import upload_to_partitioned_s3

# Initialize Faker and random seed
fake = Faker()
np.random.seed(42)
random.seed(42)

# ======================
# 1. Turbine Catalog Data
# ======================

def generate_turbine_catalog(num_turbines=50):
    models = ['GE-2.8', 'Vestas-V120', 'Siemens-SWT3.2', 'Goldwind-GW140']
    states = ['TX', 'IA', 'CA', 'OK', 'KS', 'IL']
    
    catalog = []
    for i in range(1, num_turbines+1):
        state = random.choice(states)
        install_date = fake.date_between(start_date='-10y', end_date='-2y')
        last_maintenance = fake.date_between(start_date=install_date, end_date='today')
        
        catalog.append({
            'turbine_id': f"WT-{i:03d}",
            'name': f"{fake.city()} Wind Farm",
            'model': random.choice(models),
            'install_date': install_date,
            'last_maintenance': last_maintenance,
            'state': state,
            'lat': np.random.uniform(32.5, 42.5) if state == 'TX' else np.random.uniform(34.0, 45.0),
            'lon': np.random.uniform(-102.0, -94.0) if state == 'TX' else np.random.uniform(-118.0, -87.0),
            'capacity_mw': random.choice([2.8, 3.0, 3.2, 4.0])
        })
    
    return pd.DataFrame(catalog)

# Generate and save catalog
turbine_catalog = generate_turbine_catalog(50)
turbine_catalog.to_csv('data/turbine_catalog.csv', index=False)

# ========================
# 2. Time-Series Data Generation
# ========================

def generate_turbine_data(catalog, start_date='2025-01-01', end_date='2025-04-30'):
    all_data = []
    
    for _, turbine in catalog.iterrows():
        date_range = pd.date_range(start=start_date, end=end_date, freq='5min')
        num_records = len(date_range)
        
        # Base data generation
        data = pd.DataFrame({
            'timestamp': date_range,
            'device_id': turbine['turbine_id'],
            'rpm': np.random.normal(8, 0.5, num_records).clip(6, 10),
            'angle': np.random.uniform(5, 10, num_records),
            'humidity': np.random.normal(70, 5, num_records).clip(50, 90),
        })

        # Better temperature modeling with daily/seasonal patterns
        hour_of_day = pd.to_datetime(data['timestamp']).dt.hour
        day_of_year = pd.to_datetime(data['timestamp']).dt.dayofyear

        # Daily pattern: cooler at night/morning, warmer in afternoon
        daily_pattern = -np.cos(hour_of_day * 2 * np.pi / 24) * 8  

        # Seasonal pattern (assuming Northern Hemisphere)
        seasonal_pattern = -np.cos(day_of_year * 2 * np.pi / 365) * 15

        # Base temperature (average) + patterns + random variation
        data['temperature'] = 25 + daily_pattern + seasonal_pattern + np.random.normal(0, 3, num_records)

        # Add realistic relationships
        data['windspeed'] = data['rpm']/1.5 + np.random.normal(0, 0.5, num_records)

        # Wind power varies with cube of wind speed (physics-based)
        theoretical_power = 0.5 * 1.225 * (np.pi * 50**2) * data['windspeed']**3 * 0.4 / 1000

        # Add efficiency factors
        temperature_efficiency = 1.0 - 0.005 * np.abs(data['temperature'] - 15)  # Optimal at 15°C

        # Calculate actual power with proper capacity limits based on turbine type
        data['power'] = (theoretical_power * temperature_efficiency).clip(0, turbine['capacity_mw']*1000)

        # Add natural fluctuations (turbulence, measurement errors)
        data['power'] += np.random.normal(0, data['power']*0.05, num_records)

        # Create specific fault patterns (beyond random anomalies)
        data['days_since_install'] = (pd.to_datetime(data['timestamp']) - pd.to_datetime(turbine['install_date'])).dt.days

        # Increasing vibration with age
        data['rpm_variance'] = 0.05 * np.sqrt(data['days_since_install']/365) * np.random.normal(1, 0.2, num_records)
        data['rpm'] += data['rpm'] * data['rpm_variance'] * np.sin(np.arange(num_records) * 0.5)

        # Add maintenance flags (last 30 days before maintenance)
        last_maintenance = pd.to_datetime(turbine['last_maintenance'])
        data['maintenance_flag'] = (pd.to_datetime(data['timestamp']) > 
                                   (last_maintenance - pd.DateOffset(days=30))).astype(int)

        # Add temperature rise before failure
        pre_failure_mask = (data['maintenance_flag'] == 1) & (np.random.rand(num_records) < 0.3)
        data.loc[pre_failure_mask, 'temperature'] += np.random.gamma(5, 4, sum(pre_failure_mask))
        data.loc[pre_failure_mask, 'power'] *= 0.7 + 0.3 * np.random.rand(sum(pre_failure_mask))
        
        all_data.append(data)
    
    return pd.concat(all_data)

if __name__ == "__main__":
    # Generate time-series data
    ts_data = generate_turbine_data(turbine_catalog)
    ts_data.to_csv('data/turbine_data.csv', index=False)

    upload_to_partitioned_s3(
        'data/turbine_data.csv',
        "wind_turbine/turbine_data"
    )
