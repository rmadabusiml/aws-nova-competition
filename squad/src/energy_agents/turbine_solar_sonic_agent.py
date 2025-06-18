from typing import Any
import sys, asyncio, uuid
import os
from datetime import datetime, timezone
from agent_squad.utils import Logger
from agent_squad.agents import SupervisorAgent, SupervisorAgentOptions, AgentResponse, AgentStreamResponse
from agent_squad.classifiers import BedrockClassifier, BedrockClassifierOptions
from agent_squad.classifiers import ClassifierResult
from agent_squad.types import ConversationMessage
from agent_squad.storage import InMemoryChatStorage
from agent_squad.orchestrator import AgentSquad, AgentSquadConfig
import json
from typing import List, Optional, Dict
import logging

from wind_turbine_agents.turbine_supervisor_agent import supervisor as turbine_supervisor_agent
from solar_panel_agents.solar_supervisor_agent import supervisor as solar_supervisor_agent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

memory_storage = InMemoryChatStorage()

custom_bedrock_classifier = BedrockClassifier(BedrockClassifierOptions(
        model_id=os.environ.get('turbine_solar_sonic_classifier_agent_llm', 'amazon.nova-pro-v1:0'),
        region='us-east-1',
        inference_config={
            'maxTokens': 2048,
            'temperature': 0.7,
            'topP': 0.9
        }
    ))

orchestrator = AgentSquad(classifier=custom_bedrock_classifier,options=AgentSquadConfig(
    LOG_AGENT_CHAT=True,
    LOG_CLASSIFIER_CHAT=True,
    LOG_EXECUTION_TIMES=True,
    MAX_RETRIES=3,
    USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED=True,
    MAX_MESSAGE_PAIRS_PER_AGENT=10,
))

orchestrator.add_agent(turbine_supervisor_agent)
orchestrator.add_agent(solar_supervisor_agent)

# USER_ID = str(uuid.uuid4())
# SESSION_ID = str(uuid.uuid4())

async def handle_request(_user_input:str, _user_id:str, _session_id:str):
    # classifier_result=ClassifierResult(selected_agent=supervisor, confidence=1.0)

    # response:AgentResponse = await _orchestrator.agent_process_request(_user_input, _user_id, _session_id, classifier_result, {}, True)

    response:AgentResponse = await orchestrator.route_request(
        _user_input,
        _user_id,
        _session_id
    )

    logger.info(f"response: {response}")
    final_response = None

    # Print metadata
    print("\nMetadata:")
    print(f"Selected Agent: {response.metadata.agent_name}")
    # if isinstance(response, AgentResponse) and response.streaming is False:
    if isinstance(response, AgentResponse):
        # Handle regular response
        if isinstance(response.output, str):
            print(f"\033[34m{response.output}\033[0m")
            final_response = response.output
        elif isinstance(response.output, ConversationMessage):
                print(f"\033[34m{response.output.content[0].get('text')}\033[0m")
                final_response = response.output.content[0].get('text')

    return final_response

# async def handle_request(_orchestrator: AgentSquad, _user_input:str, _user_id:str, _session_id:str):
#     # classifier_result=ClassifierResult(selected_agent=supervisor, confidence=1.0)

#     # response:AgentResponse = await _orchestrator.agent_process_request(_user_input, _user_id, _session_id, classifier_result, {}, True)

#     response:AgentResponse = await _orchestrator.route_request(
#         _user_input,
#         _user_id,
#         _session_id
#     )

#     logger.info(f"response: {response}")

#     # Print metadata
#     print("\nMetadata:")
#     print(f"Selected Agent: {response.metadata.agent_name}")
#     if isinstance(response, AgentResponse) and response.streaming is False:
#         # Handle regular response
#         if isinstance(response.output, str):
#             print(f"\033[34m{response.output}\033[0m")
#         elif isinstance(response.output, ConversationMessage):
#                 print(f"\033[34m{response.output.content[0].get('text')}\033[0m")
#     if response.streaming:
#          if isinstance(response.output, AsyncIterator):
#             async for chunk in response.output:
#                 if isinstance(chunk, AgentStreamResponse):
#                     print(f"\033[34m{chunk.text}\033[0m", end='', flush=True)
#                 else:
#                     print(f"\033[34m{chunk}\033[0m", end='', flush=True)


if __name__ == "__main__":
    # Initialize orchestrator with configuration options

    # custom_bedrock_classifier = BedrockClassifier(BedrockClassifierOptions(
    #     model_id='amazon.nova-pro-v1:0',
    #     region='us-east-1',
    #     inference_config={
    #         'maxTokens': 2048,
    #         'temperature': 0.7,
    #         'topP': 0.9
    #     }
    # ))

    # orchestrator = AgentSquad(classifier=custom_bedrock_classifier,options=AgentSquadConfig(
    #     LOG_AGENT_CHAT=True,
    #     LOG_CLASSIFIER_CHAT=True,
    #     LOG_EXECUTION_TIMES=True,
    #     MAX_RETRIES=3,
    #     USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED=True,
    #     MAX_MESSAGE_PAIRS_PER_AGENT=10,
    # ))

    # orchestrator.add_agent(turbine_supervisor_agent)
    # orchestrator.add_agent(solar_supervisor_agent)

    # USER_ID = str(uuid.uuid4())
    # SESSION_ID = str(uuid.uuid4())

    while True:
        # Get user input
        user_input = input("\nYou: ").strip()

        if user_input.lower() == 'quit':
            print("Exiting the program. Goodbye!")
            sys.exit()

        # Run async function to process user input
        if user_input:
            asyncio.run(handle_request(orchestrator, user_input, USER_ID, SESSION_ID))
    