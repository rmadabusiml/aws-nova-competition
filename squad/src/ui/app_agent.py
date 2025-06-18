import os
import streamlit as st
import asyncio
import logging
import uuid

from agent_squad.agents import SupervisorAgent, SupervisorAgentOptions, AgentResponse, AgentStreamResponse
from agent_squad.utils import Logger
from agent_squad.agents import SupervisorAgent, SupervisorAgentOptions, AgentResponse, AgentStreamResponse
from agent_squad.classifiers import BedrockClassifier, BedrockClassifierOptions
from agent_squad.classifiers import ClassifierResult
from agent_squad.types import ConversationMessage
from agent_squad.storage import InMemoryChatStorage
from agent_squad.orchestrator import AgentSquad, AgentSquadConfig
from wind_turbine_agents.turbine_supervisor_agent import supervisor as turbine_supervisor_agent
from solar_panel_agents.solar_supervisor_agent import supervisor as solar_supervisor_agent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

status_placeholder = st.empty()
output_placeholder = st.empty()

memory_storage = InMemoryChatStorage()

custom_bedrock_classifier = BedrockClassifier(BedrockClassifierOptions(
    model_id=os.environ.get('turbine_solar_chat_classifier_agent_llm', 'amazon.nova-pro-v1:0'),
    region='us-east-1',
    inference_config={
        'maxTokens': 2048,
        'temperature': 0.7,
        'topP': 0.9
    }
))

orchestrator = AgentSquad(classifier=custom_bedrock_classifier, storage=memory_storage,
    options=AgentSquadConfig(
        LOG_AGENT_CHAT=True,
        LOG_CLASSIFIER_CHAT=True,
        LOG_EXECUTION_TIMES=True,
        MAX_RETRIES=3,
        USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED=True,
        MAX_MESSAGE_PAIRS_PER_AGENT=10,
    )
)

orchestrator.add_agent(turbine_supervisor_agent)
orchestrator.add_agent(solar_supervisor_agent)

USER_ID = str(uuid.uuid4())
SESSION_ID = str(uuid.uuid4())

def display_agent():
    st.title("👩‍💻 Agent - A Virtual assistant for Wind Turbine and Solar Panel")

    turbine_question_expander = st.expander("Turbine Sample Questions")
    solar_question_expander = st.expander("Solar Sample Questions")
     
    with turbine_question_expander:
        st.markdown("""
        ### Knowledge Base:
        - What troubleshooting steps should I follow if a wind turbine is spinning but not delivering electrical output to the batteries or grid?
        - How can condition monitoring systems (CMS) be used to predict and prevent catastrophic drivetrain failures, and what data should I monitor?
        - How should I respond if I observe cracks wider than 0.3mm in the pedestal or grout during an inspection?
        - What troubleshooting steps should I follow if a wind turbine is spinning but not delivering electrical output to the batteries or grid?
        - What are the most common causes of drivetrain failure in wind turbines?
        - List at least 8 typical damages seen on wind turbine bearings. No need for explanation of each
        - causes of Electrical Fluting in Turbine
        - what causes fretting corrosion
        - Why the turbine comes to rest in the same horizontal position, regardless of wind direction
        - what are the Effects and impact of Grout - Spalling on turbine
        - Should I use strain or vibaration monitoring for bending and shear loads

        ### Guardrails:
        - What is your opinion on wind turbine subsidies?

        ### Asset Details:
        - How many turbines located in Texas?
        - Get the details of the turbine WT-035 (also try with WT-045, WT-007, WT-016)
        - What is the cost and profit of this turbine on May 12th, 2025

        ### Foundational Image Analysis:
        - What foundational issue observed with this turbine
        - What are the recommended preventative maintenance to address this foundational issue
        """)

    with solar_question_expander:
        st.markdown("""
        ### Knowledge Base:
        - How often should I clean my solar panels if I live in a dusty desert climate vs. a rainy coastal area?
        - What are the cost comparison factors between DIY cleaning vs professional services for a 30-panel system?
        - Can using a pressure washer void my solar panel warranty? What cleaning methods are prohibited?
        - How to To troubleshoot inverter issues in solar panel
        - What are the main causes of zero voltage issue in solar panel
    
        ### Tool Call - Cost savings:
        - Based on my monthly electricity bill from Eversource, what's the cost savings with solar panel
        - Based on my monthly electricity bill from National Grid, what's the cost savings with solar panel
        - Based on my monthly electricity bill from Ameren, what's the cost savings with solar panel
        - Based on my monthly electricity bill from Rocky Mountain Power, what's the cost savings with solar panel

        ### MCP Tool - Solar insights potential:
        - provide the solar insight potential for the address: 1300, Westborough Ln, Leander, TX-78641
        - provide the solar insight potential for the address: 1364, Brome Dr, Leander, TX-78641
    """)

    # User input section
    user_input = st.text_area("You:", key="input", placeholder="Ask me a question... ")
    col1, col2 = st.columns(2)
    
    # Create empty containers for dynamic content
    status_container = st.container()
    output_container = st.container()

    if 'processing' not in st.session_state:
        st.session_state.processing = False

    if col1.button("Submit", disabled=st.session_state.processing) or (user_input and user_input[-1] == "\n"):
        st.session_state.processing = True
        
        # Clear previous results
        status_container.empty()
        output_container.empty()
        
        # Display status and output in dedicated containers
        with status_container:
            # st.markdown("### Agent Status")
            status_placeholder = st.empty()
            
        with output_container:
            output_placeholder = st.empty()

        print(f"User ID: {USER_ID}")
        print(f"Session ID: {SESSION_ID}")

        asyncio.run(run_query(orchestrator, user_input.strip(), USER_ID, SESSION_ID, status_placeholder, output_placeholder))
        st.session_state.processing = False

        # Display the conversation history
        if 'conversations' in st.session_state:
            for question, final_response in reversed(st.session_state["conversations"]):
                st.info(f"\nQuestion:\n + {question}")
                st.success(f"\nFinal Response:\n +  {final_response}")

async def run_query(_orchestrator: AgentSquad, _user_input:str, _user_id:str, _session_id:str, status_placeholder, output_placeholder):
    try:
        # Initialize placeholders for status updates
        # response = await handle_ui_request(_user_input, _user_id, _session_id)
        response:AgentResponse = await _orchestrator.route_request(
            _user_input,
            _user_id,
            _session_id
        )

        logger.info(f"response: {response}")
        final_response = None

        # Print metadata
        print("\nMetadata:")
        print(f"Selected Agent: {response.metadata.agent_name}")

        if isinstance(response, AgentResponse):
            # Handle regular response
            if isinstance(response.output, str):
                print(f"\033[34m{response.output}\033[0m")
                final_response = response.output
            elif isinstance(response.output, ConversationMessage):
                    print(f"\033[34m{response.output.content[0].get('text')}\033[0m")
                    final_response = response.output.content[0].get('text')

        print(f"Response in UI: {final_response}")

        assistant_message = st.chat_message("assistant")
        assistant_message.text(final_response)

        status_placeholder.markdown("")
        output_placeholder.markdown("")

        st.session_state.setdefault("conversations", []).append((_user_input, final_response))

    except Exception as e:
        st.error(f"Error occurred: {e}")
