import boto3
import json
import os
from datetime import datetime
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def get_named_parameter(event, name):
    """Extract a named parameter from the event"""
    return next(item for item in event['parameters'] if item['name'] == name)['value']

def populate_function_response(event, response_body):
    """Format the response for Bedrock Agent"""
    return {'response': {'actionGroup': event['actionGroup'], 'function': event['function'],
        'functionResponse': {'responseBody': {'TEXT': {'body': json.dumps(response_body)}}}}}

def compute_savings(monthly_cost: float) -> float:
    """
    Tool to compute the potential savings when switching to solar energy based on the user's monthly electricity cost.
    
    Args:
        monthly_cost (float): The user's current monthly electricity cost.
    
    Returns:
        dict: A dictionary containing:
            - 'number_of_panels': The estimated number of solar panels required.
            - 'installation_cost': The estimated installation cost.
            - 'net_savings_10_years': The net savings over 10 years after installation costs.
    """
    def calculate_solar_savings(monthly_cost):
        # Assumptions for the calculation
        cost_per_kWh = 0.28  
        cost_per_watt = 1.20  
        sunlight_hours_per_day = 5.5  
        panel_wattage = 350  
        system_lifetime_years = 10  

        # Monthly electricity consumption in kWh
        monthly_consumption_kWh = monthly_cost / cost_per_kWh
        
        # Required system size in kW
        daily_energy_production = monthly_consumption_kWh / 30
        system_size_kW = daily_energy_production / sunlight_hours_per_day
        
        # Number of panels and installation cost
        number_of_panels = system_size_kW * 1000 / panel_wattage
        installation_cost = system_size_kW * 1000 * cost_per_watt
        
        # Annual and net savings
        annual_savings = monthly_cost * 12
        total_savings_10_years = annual_savings * system_lifetime_years
        net_savings = total_savings_10_years - installation_cost
        
        logger.info(f"Number of panels: {number_of_panels}")
        logger.info(f"Installation cost: {installation_cost}")
        logger.info(f"Net savings over 10 years: {net_savings}")
        
        return {
            "number_of_panels": round(number_of_panels),
            "installation_cost": round(installation_cost, 2),
            "net_savings_10_years": round(net_savings, 2)
        }

    logger.info(f"Monthly cost: {monthly_cost}")

    # Return calculated solar savings
    return calculate_solar_savings(monthly_cost)

def lambda_handler(event, context):
    """Main Lambda handler"""
    print("Received event:", json.dumps(event, indent=2))
    
    function = event.get('function', '')
    parameters = event.get('parameters', [])
    result = None
    
    try:
        if function == 'compute_savings':
            monthly_cost = float(get_named_parameter(event, 'monthly_cost'))
            logger.info(f"Monthly cost: {monthly_cost}")
            result = compute_savings(monthly_cost)
            logger.info(f"Result: {result}")
        
        else:
            result = {"error": f"Unknown function: {function}"}
    
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        result = {"error": f"Error processing request: {str(e)}"}
    
    response = populate_function_response(event, result)
    print("Response:", json.dumps(response, indent=2))
    return response
