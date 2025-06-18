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
from solar_panel_agents.solar_insights_tool import solar_insights_tools, bedrock_solar_insights_tool_handler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

memory_storage = InMemoryChatStorage()

solar_panel_catalog_agent = AmazonBedrockAgent(AmazonBedrockAgentOptions(
    name='Solar Panel Catalog Agent',
    description='You are a Solar Panel Catalog Assistant that helps users compute cost savings, various cleaning tips, troubleshooting, maintenance, and general information about Solar Panel. Keep the response short and to the point within in 5 sentences long that is easy for any speech assistant to respond to the user',
    agent_id='FHQEEZVFM5',
    agent_alias_id='WVS0X7WVMA',
    region='us-east-1',
    streaming=True
))

electricity_utility_bill_image_analysis_agent = LambdaAgent(LambdaAgentOptions(
    name='Electricity Utility Bill Image Analysis Agent',
    description='You are an Electricity Utility Bill Image Analysis Assistant that extracts electricity bill amount due given a company name. Do not expect user to provide the monthly electricity cost or any sort of confirmation. Keep the response short and to the point within in 5 sentences long that is easy for any speech assistant to respond to the user',
    function_name='solar_panel_image_analyzer',
    function_region='us-east-1',
))

solar_insights_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
    name='solar_insights_agent',
    description='Agent specialized in providing solar insights for a given residence address. The response should contain max number of solar panels that can be installed, median solar panel configuration, roof orientation, annual carbon offset, and average sun light hours, and any other relevant information.',
    model_id = os.environ.get('solar_insights_agent_llm', 'anthropic.claude-3-haiku-20240307-v1:0'),
    # model_id="amazon.nova-pro-v1:0", # unable to extract the full address from the user input. After some prompt tweaking, Nova can do it but the response from the MCP tool is not consistent.
    tool_config={
        'tool': solar_insights_tools,
        'toolMaxRecursions': 5,
    }
))

supervisor = SupervisorAgent(SupervisorAgentOptions(
    name="Solar Panel Supervisor Agent",
    description=(
        "You are a team supervisor managing a Solar Panel Catalog Agent and a Electricity Utility Bill Image Analysis Agent. "
        "For Solar Panel cost savings given current electricity bill amount due, cleaning tips, troubleshooting, maintenance, and general information about Solar Panel, use Solar Panel Catalog Agent. This agent knows how to calculate the cost savings based on the monthly electricity cost along with cleaning tips, troubleshooting, maintenance, and general information about Solar Panel. Consult Electricity Utility Bill Image Analysis Agent to get the monthly electricity cost. DO NOT expect user to provide the monthly electricity cost or any sort of confirmation."
        "For extracting electricity bill amount due given a company's name, use Electricity Utility Bill Image Analysis Agent. This agent knows how to fetch the monthly electricity bill amount due for a given company name. Do Not expect user to provide the montly electricity cost or any sort of confirmation."
        "For Solar Insights, use Solar Insights Agent. This agent knows how to provide solar potential insights for a given address. The response should contain max number of solar panels that can be installed, median solar panel configuration, roof orientation, annual carbon offset, and average sun light hours, and any other relevant information."
        "Keep the response short and to the point within in 5 to 8 sentences long that is easy for any speech assistant to respond to the user"
    ),
    lead_agent=BedrockLLMAgent(BedrockLLMAgentOptions(
        name="LeadSolarPanelSupervisorAgent",
        description="You are a supervisor agent. You are responsible for managing the flow of the conversation. You are only allowed to manage the flow of the conversation. Keep the response short, concise and within 5 to 8 sentences long. You are not allowed to answer questions about anything else.",
        model_id=os.environ.get('solar_supervisor_lead_agent_llm', 'anthropic.claude-3-5-sonnet-20240620-v1:0'),
        custom_system_prompt={
            'template': 'Keep the response short and to the point within in 5 to 8 sentences long that is easy for any speech assistant to respond to the user.'
        }
    )),
    team=[solar_panel_catalog_agent, electricity_utility_bill_image_analysis_agent, solar_insights_agent],
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
    orchestrator = AgentSquad(
        options=AgentSquadConfig(
            LOG_AGENT_CHAT=True,
            LOG_CLASSIFIER_CHAT=True,
            LOG_EXECUTION_TIMES=True,
            MAX_RETRIES=3,
            USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED=True,
            MAX_MESSAGE_PAIRS_PER_AGENT=10,
        )
    )

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