from typing import Any
import sys, asyncio, uuid
import os
from datetime import datetime, timezone
from agent_squad.utils import Logger
from agent_squad.agents import SupervisorAgent, SupervisorAgentOptions, AgentResponse, AgentStreamResponse
from agent_squad.agents import LambdaAgent, LambdaAgentOptions
from agent_squad.agents import AmazonBedrockAgent, AmazonBedrockAgentOptions
from agent_squad.agents import BedrockLLMAgent, BedrockLLMAgentOptions
from agent_squad.classifiers import ClassifierResult
from agent_squad.types import ConversationMessage
from agent_squad.storage import InMemoryChatStorage
from agent_squad.orchestrator import AgentSquad, AgentSquadConfig
import json
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

memory_storage = InMemoryChatStorage()

turbine_catalog_agent = AmazonBedrockAgent(AmazonBedrockAgentOptions(
    name='Wind Turbine Catalog Agent',
    description='You are a Wind Turbine Assistant that helps users understand their wind turbine fleet. When asked for details of a Turbine then extract its catalog information. When asked for metrics such as optimal_rpm, cost, profit, revenue then extract the data from the Turbine asset optimization for a specific date and NOT look for its financial data. Look into this agent knowledge base for preventive maintenance, troubleshooting, best practices related to Wind Turbine. Keep the response short and to the point within in 5 sentences long that is easy for any speech assistant to respond to the user',
    agent_id='CMRE2AJPGN',
    agent_alias_id='NVCIPBOYRV',
    region='us-east-1',
    streaming=True
))

turbine_image_agent = LambdaAgent(LambdaAgentOptions(
    name='Wind Turbine Image Analysis Agent',
    description='You are a Wind Turbine Image Analysis Assistant that helps users to provide a brief description of problematic issues that is seen on the image specifically with respect to Wind Turbine onshore foundation, wear and tear, etc. This agent knows how to fetch the relevant image from S3 bucket for a turbine. Do not expect user to provide the image. Keep the response short and to the point within in 5 sentences long that is easy for any speech assistant to respond to the user',
    function_name='wind_turbine_image_analyzer',
    function_region='us-east-1',
))

supervisor = SupervisorAgent(SupervisorAgentOptions(
    name="Wind Turbine Supervisor Agent",
    description=(
        "You are a team supervisor managing a Turbine Catalog Agent and a Turbine Image Analysis Agent. "
        "For Turbine details, performance metrics, troubleshooting related queries, use Wind Turbine Catalog Agent. "
        "For Turbine foundation related queries, use Wind Turbine Image Analysis Agent. This agent knows how to fetch the relevant image from S3 bucket for a turbine. Do not expect user to provide the image."
        "Keep the response short and to the point within in 5 sentences long that is easy for any speech assistant to respond to the user"
    ),
    lead_agent=BedrockLLMAgent(BedrockLLMAgentOptions(
        name="LeadTurbineSupervisorAgent",
        description="You are a supervisor agent. You are responsible for managing the flow of the conversation. You are only allowed to manage the flow of the conversation. You are not allowed to answer questions about anything else. DO NOT suggest any follow up questions. Keep the response short, concise and within 5 sentences long",
        model_id=os.environ.get('turbine_supervisor_lead_agent_llm', 'anthropic.claude-3-5-sonnet-20240620-v1:0'),
        custom_system_prompt={
            'template': 'Keep the response short and to the point within in 5 sentences long that is easy for any speech assistant to respond to the user'
        },
        guardrail_config={
            'guardrailIdentifier': 'zx24scgaszcw',
            'guardrailVersion': 'DRAFT'
        },
    )),
    team=[turbine_catalog_agent, turbine_image_agent],
    trace=True,
    storage=memory_storage
))

async def handle_request(_orchestrator: AgentSquad, _user_input:str, _user_id:str, _session_id:str):
    classifier_result=ClassifierResult(selected_agent=supervisor, confidence=1.0)

    response:AgentResponse = await _orchestrator.agent_process_request(_user_input, _user_id, _session_id, classifier_result, {}, True)
    logger.info(f"response: {response}")

    # Print metadata
    print("\nMetadata:")
    print(f"Selected Agent: {response.metadata.agent_name}")
    if isinstance(response, AgentResponse) and response.streaming is False:
        # Handle regular response
        if isinstance(response.output, str):
            print(f"\033[34m{response.output}\033[0m")
        elif isinstance(response.output, ConversationMessage):
                print(f"\033[34m{response.output.content[0].get('text')}\033[0m")

if __name__ == "__main__":
    # Initialize orchestrator with configuration options
    orchestrator = AgentSquad(options=AgentSquadConfig(
        LOG_AGENT_CHAT=True,
        LOG_CLASSIFIER_CHAT=True,
        LOG_EXECUTION_TIMES=True,
        MAX_RETRIES=3,
        USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED=True,
        MAX_MESSAGE_PAIRS_PER_AGENT=10,
    ))

    orchestrator.add_agent(supervisor)

    USER_ID = str(uuid.uuid4())
    SESSION_ID = str(uuid.uuid4())

    while True:
        # Get user input
        user_input = input("\nYou: ").strip()

        if user_input.lower() == 'quit':
            print("Exiting the program. Goodbye!")
            sys.exit()

        # Run async function to process user input
        if user_input:
            asyncio.run(handle_request(orchestrator, user_input, USER_ID, SESSION_ID))