import base64
import boto3
import json
import os
import tempfile
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize clients
bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
s3_client = boto3.client('s3')

# Image mapping dictionary
solar_panel_images = {
    "eversource": "eversource_tx.png",
    "ameren": "ameren_il.png",
    "rocky mountain power": "rmp_ut.png",
    "national grid": "ng_ny.png",
}

def download_and_encode(energy_company_name):
    """Download image from S3 and return base64 encoded string"""
    try:
        bucket = 'handsonllms-raghu'
        local_path = os.path.join(tempfile.gettempdir(), solar_panel_images[energy_company_name])
        
        # Download image from S3
        s3_client.download_file(bucket, f'solar_panel/images/{solar_panel_images[energy_company_name]}', local_path)
        
        # Encode to base64
        with open(local_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
            
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        return None

def extract_energry_company_name(query, chat_history):
    """Use Nova Micro model to extract energy company name from query"""
    try:
        response = bedrock_runtime.invoke_model(
            modelId=os.environ.get('solar_panel_image_company_name_llm', 'amazon.nova-micro-v1:0'),
            body=json.dumps({
                "schemaVersion": "messages-v1",
                "messages": [{
                    "role": "user",
                    "content": [{
                        "text": f"""Extract the energy company name from this query such as eversource, ameren, rocky mountain power, national grid etc.
                        Only respond with the company name. Query: {query}. Chat History: {chat_history}"""
                    }]
                }]
            })
        )
        return json.loads(response['body'].read())['output']['message']['content'][0]['text'].strip()
        
    except Exception as e:
        logger.error(f"ID extraction error: {str(e)}")
        return None

def lambda_handler(event, context):
    try:
        # Parse input
        logger.info(f"event: {event}")
        query = event.get('query', '')
        chat_history = event.get('chatHistory', [])
        logger.info(f"chat_history: {chat_history}")
        
        # Step 1: Extract energy company name using Nova Micro
        energy_company_name = extract_energry_company_name(query, chat_history).lower()
        logger.info(f"Extracted energy company name: {energy_company_name}")

        if not energy_company_name or energy_company_name not in solar_panel_images:
            return {
                "body": json.dumps({"response": "Could not identify valid energy company name in query"})
            }
        
        # Step 2: Download and encode image
        base64_image = download_and_encode(energy_company_name)
        if not base64_image:
            return {
                "body": json.dumps({"response": "Error processing energy company image"})
            }
        
        # Step 3: Create multimodal payload for Nova Lite
        system_prompt = [{
            "text": "You are a Electricity Utility Bill Analysis Assistant. Analyze the provided image of an electricity utility bill and extract only the amount due and return the amount in numerical format such as 123.45 without currency symbol. DO NOT provide any other details such as company name, bill date etc."
        }]
        
        messages = [{
            "role": "user",
            "content": [
                {
                    "image": {
                        "format": "png",
                        "source": {"bytes": base64_image}
                    }
                },
                {
                    "text": f"chat_history: {chat_history}\n\n  {query} Provide only the amount due in numeric format such as 123.45 without currency symbol."
                }
            ]
        }]
        
        # Invoke Nova Lite model
        response = bedrock_runtime.invoke_model(
            modelId=os.environ.get('solar_panel_image_amount_llm', 'amazon.nova-lite-v1:0'),
            body=json.dumps({
                "schemaVersion": "messages-v1",
                "messages": messages,
                "system": system_prompt,
                "inferenceConfig": {
                    "maxTokens": 512,
                    "temperature": 0.2,
                    "topP": 0.9
                }
            })
        )
        
        # Parse model response
        model_output = json.loads(response['body'].read())
        analysis = model_output['output']['message']['content'][0]['text']
        
        return {
            "body": json.dumps({
                "response": analysis,
                "energy_company_name": energy_company_name,
                "image_used": solar_panel_images[energy_company_name]
            })
        }
        
    except Exception as e:
        logger.error(f"Lambda execution error: {str(e)}")
        return {
            "body": json.dumps({"response": "Error processing your request"})
        }
