from langchain_aws import ChatBedrockConverse
from dotenv import load_dotenv
import os
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

load_dotenv()

DEFAULT_MAX_TOKENS = 2048

nova_micro_llm = ChatBedrockConverse(
    model="amazon.nova-micro-v1:0",
    region_name=os.environ.get('AWS_REGION', 'us-east-1'),
    temperature=0,
    max_tokens=DEFAULT_MAX_TOKENS
)

nova_lite_llm = ChatBedrockConverse(
    model="amazon.nova-lite-v1:0", 
    region_name=os.environ.get('AWS_REGION', 'us-east-1'),
    temperature=0,
    max_tokens=DEFAULT_MAX_TOKENS
)

nova_pro_llm = ChatBedrockConverse(
    model="amazon.nova-pro-v1:0",   
    region_name=os.environ.get('AWS_REGION', 'us-east-1'), 
    temperature=0,
    max_tokens=DEFAULT_MAX_TOKENS
)
