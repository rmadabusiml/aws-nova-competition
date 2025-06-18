import boto3
import csv
from botocore.exceptions import ClientError

# Initialize DynamoDB resource
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')

def create_wind_turbine_table():
    try:
        table = dynamodb.create_table(
            TableName='WT_Catalog',
            KeySchema=[
                {
                    'AttributeName': 'turbine_id',
                    'KeyType': 'HASH'  # Partition key
                }
            ],
            AttributeDefinitions=[
                {
                    'AttributeName': 'turbine_id',
                    'AttributeType': 'S'
                }
            ],
            ProvisionedThroughput={
                'ReadCapacityUnits': 5,
                'WriteCapacityUnits': 5
            }
        )
        table.wait_until_exists()
        print("Table created successfully")
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceInUseException':
            print("Table already exists")
        else:
            print(f"Unexpected error: {e}")

def create_wind_turbine_asset_optimization_table():
    try:
        table = dynamodb.create_table(
            TableName='WT_Asset_Optimization',
            KeySchema=[
                {
                    'AttributeName': 'turbine_id',
                    'KeyType': 'HASH'  # Partition key
                },
                {
                    'AttributeName': 'assessed_date',
                    'KeyType': 'RANGE'  # Sort key
                }
            ],
            AttributeDefinitions=[
                {
                    'AttributeName': 'turbine_id',
                    'AttributeType': 'S'
                },
                {
                    'AttributeName': 'assessed_date',
                    'AttributeType': 'S'  # Stored as ISO 8601 string
                }
            ],
            ProvisionedThroughput={
                'ReadCapacityUnits': 5,
                'WriteCapacityUnits': 5
            }
        )
        table.wait_until_exists()
        print("Table created successfully")
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceInUseException':
            print("Table already exists")
        else:
            print(f"Unexpected error: {e}")


def load_data_from_csv(file_path):
    table = dynamodb.Table('WT_Catalog')
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        with table.batch_writer() as batch:
            for row in reader:
                batch.put_item(Item=row)
    print("Data loaded successfully")

def load_asset_optimization_data_from_csv(file_path):
    table = dynamodb.Table('WT_Asset_Optimization')
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        with table.batch_writer() as batch:
            for row in reader:
                batch.put_item(Item=row)
    print("Asset Optimization Data loaded successfully")

if __name__ == '__main__':
    create_wind_turbine_table()
    create_wind_turbine_asset_optimization_table()
    load_data_from_csv('data/turbine_catalog.csv')
    