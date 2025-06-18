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
solar_agent_name = f"solar-agent-{resource_suffix}"
solar_lambda_name = f"fn-solar-agent-{resource_suffix}"

# DynamoDB configuration for the main table
dynamoDB_args = None

# Get AWS account ID and region
account_id = boto3.client("sts").get_caller_identity()["Account"]
region = agents.get_region()

knowledge_base_name = f'{solar_agent_name}-kb'
knowledge_base_description = "KB containing information on clearning, maintenance, troubleshooting, and general information about Solar Panel"
bucket_name = f'solar-agent-kb-{account_id}-{resource_suffix}'

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

upload_directory("docs", bucket_name)
# sync knowledge base
kb.synchronize_data(kb_id, ds_id)


kb_info = kb.get_kb(kb_id)
kb_arn = kb_info['knowledgeBase']['knowledgeBaseArn']

print(f"Knowledge Base ARN: {kb_arn}")

kb_config = {
    'kb_id': kb_id,
    'kb_instruction': """Access this knowledge base when needing to explain various cleaning tips, troubleshooting, maintenance, best practices relaed to Solar Panel."""
}

# Additional IAM policy for catalog table access
additional_iam_policy = None

# Use Amazon Nova model
agent_foundation_model = [os.environ.get('solar_agent_llm', 'amazon.nova-lite-v1:0')]

# Agent description and instructions
agent_description = "You are a Solar Panel Assistant that helps users compute cost savings, various cleaning tips, troubleshooting, maintenance, and general information about Solar Panel."

agent_instruction = """
You are a Solar Panel Assistant that helps users compute cost savings, various cleaning tips, troubleshooting, maintenance, best practices, and general information about Solar Panel.

Your capabilities include:
1. Compute the potential savings when switching to solar energy based on the user's monthly electricity cost.
2. Provide various cleaning tips, troubleshooting, maintenance, best practices, and general information about Solar Panel.

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
print(f"Creating agent: {solar_agent_name}")

solar_agent = agents.create_agent(
    solar_agent_name,
    agent_description,
    agent_instruction,
    agent_foundation_model,
    kb_arns=[kb_arn]
)

solar_agent_id = solar_agent[0]

agents.associate_kb_with_agent(
    solar_agent_id,
    kb_config['kb_instruction'],
    kb_config['kb_id']
)

# Define the function definitions for the action group
functions_def = [
    {
        "name": "compute_savings",
        "description": "Compute the potential savings when switching to solar energy based on the user's monthly electricity cost.",
        "parameters": {
            "monthly_cost": {
                "description": "Monthly electricity cost in dollars",
                "required": True,
                "type": "number"
            }
        }
    }
]

# Create Lambda function and add action group to agent
print(f"Adding action group to agent with Lambda function: {solar_lambda_name}")

agents.add_action_group_with_lambda(
    agent_name=solar_agent_name,
    lambda_function_name=solar_lambda_name,
    source_code_file="src/agents/solar_info.py",
    agent_functions=functions_def,
    agent_action_group_name="solar_actions",
    agent_action_group_description="Functions to compute the potential savings when switching to solar energy based on the user's monthly electricity cost.",
    additional_function_iam_policy=None,
    dynamo_args=dynamoDB_args
)

# Create an alias for the agent for testing
print("Creating agent alias for testing")
solar_agent_alias_id, solar_agent_alias_arn = agents.create_agent_alias(
    solar_agent_id, 'v1'
)

print(f"Agent ID: {solar_agent_id}")
print(f"Agent Alias ID: {solar_agent_alias_id}")
print(f"Agent Alias ARN: {solar_agent_alias_arn}")
