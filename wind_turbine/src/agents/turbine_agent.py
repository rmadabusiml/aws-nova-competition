import os
import boto3
import json
import uuid
from src.helper.bedrock_agent_helper import AgentsForAmazonBedrock
from src.helper.knowledge_base_helper import KnowledgeBasesForAmazonBedrock

# Initialize the helper
agents = AgentsForAmazonBedrock()
kb = KnowledgeBasesForAmazonBedrock()

# Generate a unique ID for resources
resource_suffix = str(uuid.uuid4())[:8]

# Agent configuration
turbine_agent_name = f"turbine-agent-{resource_suffix}"
turbine_lambda_name = f"fn-turbine-agent-{resource_suffix}"

# DynamoDB table names
catalog_table = 'WT_Catalog'
optimization_table = 'WT_Asset_Optimization'

# DynamoDB configuration for the main table
# The sort key is for the asset optimization table
dynamoDB_args = [optimization_table, 'turbine_id', 'assessed_date']

# Get AWS account ID and region
account_id = boto3.client("sts").get_caller_identity()["Account"]
region = agents.get_region()

knowledge_base_name = f'{turbine_agent_name}-kb'
knowledge_base_description = "KB containing information on how forecasting process is done"
bucket_name = f'turbine-agent-kb-{account_id}-{resource_suffix}'

kb_id, ds_id = kb.create_or_retrieve_knowledge_base(
    knowledge_base_name,
    knowledge_base_description,
    bucket_name
)

print(f"Knowledge Base ID: {kb_id}")
print(f"Data Source ID: {ds_id}")

s3_client = boto3.client('s3', region)

def upload_directory(path, bucket_name):
    for root,dirs,files in os.walk(path):
        for file in files:
            file_to_upload = os.path.join(root,file)
            print(f"uploading file {file_to_upload} to {bucket_name}")
            s3_client.upload_file(file_to_upload,bucket_name,file)

upload_directory("kb_docs", bucket_name)
# sync knowledge base
kb.synchronize_data(kb_id, ds_id)


kb_info = kb.get_kb(kb_id)
kb_arn = kb_info['knowledgeBase']['knowledgeBaseArn']

print(f"Knowledge Base ARN: {kb_arn}")

kb_config = {
    'kb_id': kb_id,
    'kb_instruction': """Access this knowledge base when needing to explain various troubleshooting, maintenance, best practices, foundation relaed to Wind Turbine."""
}

# Additional IAM policy for catalog table access
additional_iam_policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "dynamodb:GetItem",
                "dynamodb:Query",
                "dynamodb:Scan"
            ],
            "Resource": f"arn:aws:dynamodb:{region}:{account_id}:table/{catalog_table}"
        }
    ]
}

# Use Amazon Nova model
agent_foundation_model = [os.environ.get('wind_turbine_agent_llm', 'amazon.nova-lite-v1:0')]

# Agent description and instructions
agent_description = "You are a Wind Turbine Assistant that helps users understand their wind turbine fleet and performance metrics."

agent_instruction = """
You are a Wind Turbine Assistant that helps users understand their wind turbine fleet and performance metrics.

Your capabilities include:
1. Retrieving details about specific wind turbines
2. Finding turbines by state, model, or other criteria
3. Providing performance metrics for turbines
4. Analyzing turbine data by state or model

Core behaviors:
1. Always use available information systems before asking users for additional details
2. Maintain a professional yet conversational tone
3. Provide clear, direct answers without referencing internal systems or data sources
4. Present information in an easy-to-understand manner

Response style:
- Be helpful and solution-oriented
- Use clear, non-technical language when possible
- Focus on providing actionable insights
- Maintain natural conversation flow
- Be concise yet informative
- Keep the response short and to the point within in 5 sentences long that is easy for any speech assistant to respond to the user
- Do not add extra information not required by the user
"""

# Create the agent
print(f"Creating agent: {turbine_agent_name}")

turbine_agent = agents.create_agent(
    turbine_agent_name,
    agent_description,
    agent_instruction,
    agent_foundation_model,
    kb_arns=[kb_arn]
)

turbine_agent_id = turbine_agent[0]

agents.associate_kb_with_agent(
    turbine_agent_id,
    kb_config['kb_instruction'],
    kb_config['kb_id']
)

# Define the function definitions for the action group
functions_def = [
    {
        "name": "get_turbine_by_id",
        "description": "Gets detailed information about a specific turbine by ID",
        "parameters": {
            "turbine_id": {
                "description": "Unique turbine identifier (e.g., WT-001)",
                "required": True,
                "type": "string"
            }
        }
    },
    {
        "name": "get_turbines_by_state",
        "description": "Gets a list of turbines in a specific state",
        "parameters": {
            "state": {
                "description": "US state abbreviation (e.g., TX, IL, IA)",
                "required": True,
                "type": "string"
            }
        }
    },
    {
        "name": "get_turbines_by_model",
        "description": "Gets a list of turbines of a specific model",
        "parameters": {
            "model": {
                "description": "Turbine model (e.g., GE-2.8, Vestas-V120)",
                "required": True,
                "type": "string"
            }
        }
    },
    {
        "name": "get_turbine_performance",
        "description": "Gets performance metrics for a specific turbine on a given date",
        "parameters": {
            "turbine_id": {
                "description": "Unique turbine identifier (e.g., WT-001)",
                "required": True,
                "type": "string"
            },
            "assessed_date": {
                "description": "Date of assessment in YYYY-MM-DD format",
                "required": True,
                "type": "string"
            },
            "metrics": {
                "description": "Comma-separated list of metrics to retrieve (e.g., optimal_rpm, cost,revenue,profit)",
                "required": False,
                "type": "string"
            }
        }
    },
    {
        "name": "get_all_turbine_performances",
        "description": "Gets performance data for all turbines on a specific date",
        "parameters": {
            "assessed_date": {
                "description": "Date of assessment in YYYY-MM-DD format",
                "required": True,
                "type": "string"
            }
        }
    },
    {
        "name": "count_turbines_by_state",
        "description": "Gets a count of turbines by state",
        "parameters": {}
    },
    {
        "name": "count_turbines_by_model",
        "description": "Gets a count of turbines by model",
        "parameters": {}
    }
]

# Create Lambda function and add action group to agent
print(f"Adding action group to agent with Lambda function: {turbine_lambda_name}")

agents.add_action_group_with_lambda(
    agent_name=turbine_agent_name,
    lambda_function_name=turbine_lambda_name,
    source_code_file="src/agents/turbine_info.py",
    agent_functions=functions_def,
    agent_action_group_name="turbine_actions",
    agent_action_group_description="Functions to get wind turbine catalog and performance data",
    additional_function_iam_policy=json.dumps(additional_iam_policy),
    dynamo_args=dynamoDB_args
)

# Create an alias for the agent for testing
print("Creating agent alias for testing")
turbine_agent_alias_id, turbine_agent_alias_arn = agents.create_agent_alias(
    turbine_agent_id, 'v1'
)

print(f"Agent ID: {turbine_agent_id}")
print(f"Agent Alias ID: {turbine_agent_alias_id}")
print(f"Agent Alias ARN: {turbine_agent_alias_arn}")
