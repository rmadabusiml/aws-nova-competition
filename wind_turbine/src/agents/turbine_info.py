import boto3
import json
import os
from boto3.dynamodb.conditions import Key, Attr
from datetime import datetime

# Initialize DynamoDB resource
dynamodb_resource = boto3.resource('dynamodb')
catalog_table = os.getenv('catalog_table', 'WT_Catalog')
optimization_table = os.getenv('optimization_table', 'WT_Asset_Optimization')

def get_named_parameter(event, name):
    """Extract a named parameter from the event"""
    return next(item for item in event['parameters'] if item['name'] == name)['value']

def populate_function_response(event, response_body):
    """Format the response for Bedrock Agent"""
    return {'response': {'actionGroup': event['actionGroup'], 'function': event['function'],
        'functionResponse': {'responseBody': {'TEXT': {'body': json.dumps(response_body)}}}}}

def query_turbine_catalog(turbine_id=None, state=None, model=None, install_date=None, 
                          maintenance_date=None):
    """Query the WT_Catalog table with various filters"""
    try:
        table = dynamodb_resource.Table(catalog_table)
        
        if turbine_id:
            # Direct lookup by primary key
            response = table.get_item(Key={'turbine_id': turbine_id})
            return response.get('Item', {})
        
        # Otherwise, we need to scan with filters
        filter_expressions = []
        
        if state:
            filter_expressions.append(Attr('state').eq(state))
        
        if model:
            filter_expressions.append(Attr('model').eq(model))
        
        if install_date:
            filter_expressions.append(Attr('install_date').eq(install_date))
        
        if maintenance_date:
            filter_expressions.append(Attr('last_maintenance').eq(maintenance_date))
        
        # Combine filter expressions if there are any
        if filter_expressions:
            filter_expression = filter_expressions[0]
            for expr in filter_expressions[1:]:
                filter_expression = filter_expression & expr
            
            response = table.scan(FilterExpression=filter_expression)
        else:
            # Get all items if no filters
            response = table.scan()
        
        return response.get('Items', [])
    
    except Exception as e:
        print(f"Error querying turbine catalog: {str(e)}")
        return {"error": str(e)}

def query_asset_optimization(turbine_id=None, assessed_date=None):
    """Query the WT_Asset_Optimization table with composite key"""
    try:
        table = dynamodb_resource.Table(optimization_table)
        
        if turbine_id and assessed_date:
            # Query using composite primary key
            key_condition = Key('turbine_id').eq(turbine_id) & Key('assessed_date').eq(assessed_date)
            response = table.query(KeyConditionExpression=key_condition)
            return response.get('Items', [])
        elif turbine_id:
            # Query just by partition key
            key_condition = Key('turbine_id').eq(turbine_id)
            response = table.query(KeyConditionExpression=key_condition)
            return response.get('Items', [])
        elif assessed_date:
            # We need to scan since we can't query on just the sort key
            filter_expression = Attr('assessed_date').eq(assessed_date)
            response = table.scan(FilterExpression=filter_expression)
            return response.get('Items', [])
        else:
            # Get all items
            response = table.scan()
            return response.get('Items', [])
    
    except Exception as e:
        print(f"Error querying asset optimization: {str(e)}")
        return {"error": str(e)}

def count_turbines_by_attribute(attribute_name):
    """Count turbines grouped by a specific attribute"""
    try:
        items = query_turbine_catalog()
        counts = {}
        
        for item in items:
            attr_value = item.get(attribute_name)
            if attr_value:
                counts[attr_value] = counts.get(attr_value, 0) + 1
        
        return counts
    except Exception as e:
        print(f"Error counting turbines by {attribute_name}: {str(e)}")
        return {"error": str(e)}

def get_turbine_metrics(turbine_id, assessed_date, metrics=None):
    """Get specific metrics for a turbine on a given date"""
    try:
        data = query_asset_optimization(turbine_id, assessed_date)
        
        if not data:
            return {"error": f"No data found for turbine {turbine_id} on {assessed_date}"}
        
        if metrics:
            # Return only requested metrics
            result = {}
            for item in data:
                turbine = item['turbine_id']
                result[turbine] = {metric: item.get(metric) for metric in metrics if metric in item}
            return result
        else:
            # Return all metrics
            return data[0] if data else {}
    
    except Exception as e:
        print(f"Error getting turbine metrics: {str(e)}")
        return {"error": str(e)}

def lambda_handler(event, context):
    """Main Lambda handler"""
    print("Received event:", json.dumps(event, indent=2))
    
    function = event.get('function', '')
    parameters = event.get('parameters', [])
    result = None
    
    try:
        if function == 'get_turbine_by_id':
            turbine_id = get_named_parameter(event, 'turbine_id')
            result = query_turbine_catalog(turbine_id=turbine_id)
        
        elif function == 'get_turbines_by_state':
            state = get_named_parameter(event, 'state')
            result = query_turbine_catalog(state=state)
        
        elif function == 'get_turbines_by_model':
            model = get_named_parameter(event, 'model')
            result = query_turbine_catalog(model=model)
        
        elif function == 'get_turbine_performance':
            turbine_id = get_named_parameter(event, 'turbine_id')
            assessed_date = get_named_parameter(event, 'assessed_date')
            metrics = None
            
            # Optional parameter for specific metrics
            for param in parameters:
                if param['name'] == 'metrics':
                    metrics = param['value'].split(',')
                    break
            
            result = get_turbine_metrics(turbine_id, assessed_date, metrics)
        
        elif function == 'get_all_turbine_performances':
            assessed_date = get_named_parameter(event, 'assessed_date')
            result = query_asset_optimization(assessed_date=assessed_date)
        
        elif function == 'count_turbines_by_state':
            result = count_turbines_by_attribute('state')
        
        elif function == 'count_turbines_by_model':
            result = count_turbines_by_attribute('model')
        
        else:
            result = {"error": f"Unknown function: {function}"}
    
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        result = {"error": f"Error processing request: {str(e)}"}
    
    response = populate_function_response(event, result)
    print("Response:", json.dumps(response, indent=2))
    return response
