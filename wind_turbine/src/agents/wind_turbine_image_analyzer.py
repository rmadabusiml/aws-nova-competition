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

# environment variables
bucket = os.environ.get('wind_turbine_image_bucket')
modelId = os.environ.get('wind_turbine_image_llm', 'amazon.nova-lite-v1:0')

# Image mapping dictionary
wind_turbine_images = {
    "WT-007": "wind_turbine_water_pooling_basement.png",
    "WT-045": "wind_turbine_soil_cracking.png",
    "WT-028": "wind_turbine_tilt.png",
    "WT-016": "wind_turbine_foundation_2.png",
    "WT-035": "wind_turbine_grout_spalling.png"
}

def download_and_encode(image_key):
    """Download image from S3 and return base64 encoded string"""
    try:
        # bucket = 'handsonllms-raghu'
        local_path = os.path.join(tempfile.gettempdir(), wind_turbine_images[image_key])
        
        # Download image from S3
        s3_client.download_file(bucket, f'wind_turbine/images/{wind_turbine_images[image_key]}', local_path)
        
        # Encode to base64
        with open(local_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
            
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        return None

def extract_turbine_id(query, chat_history):
    """Use Nova Micro model to extract turbine ID from query"""
    try:
        response = bedrock_runtime.invoke_model(
            modelId=modelId,
            body=json.dumps({
                "schemaVersion": "messages-v1",
                "messages": [{
                    "role": "user",
                    "content": [{
                        "text": f"""Extract the wind turbine ID from this query. 
                        Only respond with the ID in format WT-XXX. Query: {query}. Chat History: {chat_history}"""
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
        
        # Step 1: Extract turbine ID using Nova Micro
        turbine_id = extract_turbine_id(query, chat_history)
        logger.info(f"Extracted turbine ID: {turbine_id}")

        if not turbine_id or turbine_id not in wind_turbine_images:
            return {
                "body": json.dumps({"response": "Could not identify valid turbine ID in query"})
            }
        
        # Step 2: Download and encode image
        base64_image = download_and_encode(turbine_id)
        if not base64_image:
            return {
                "body": json.dumps({"response": "Error processing turbine image"})
            }
        
        # Step 3: Create multimodal payload for Nova Lite
        system_prompt = [{
            "text": "You are a Wind Turbine Analysis Assistant. Analyze the provided image and query to identify foundation issues, structural problems of a wind turbine such as Grout Spalling, Grout Cracking, Pedestal Spalling, Pedestal cracks, Water in basement, hardware corrosion of nuts and bolts, soil cracking, etc. Only provide the issues thats seen in the image and DO NOT provide any recommendation to fix or maintenance instructions. The output should be brief, concise, and in less than 5 sentences."
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
                    "text": f"chat_history: {chat_history}\n\n  {query} Provide a detailed analysis in 5 sentences focusing on visible issues."
                }
            ]
        }]
        
        # Invoke Nova Lite model
        response = bedrock_runtime.invoke_model(
            modelId="amazon.nova-lite-v1:0",
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
                "turbine_id": turbine_id,
                "image_used": wind_turbine_images[turbine_id]
            })
        }
        
    except Exception as e:
        logger.error(f"Lambda execution error: {str(e)}")
        return {
            "body": json.dumps({"response": "Error processing your request"})
        }
