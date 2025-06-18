from agent_squad.utils import AgentTools, AgentTool
from dotenv import load_dotenv
import os

from InlineAgent.tools.mcp import MCPHttp, MCPStdio
from mcp import StdioServerParameters

from InlineAgent.action_group import ActionGroup
from InlineAgent.agent import InlineAgent
from InlineAgent import AgentAppConfig
from agent_squad.types import ConversationMessage, ParticipantRole
from typing import List, Dict, Any

config = AgentAppConfig()

async def get_solar_insights(address:str):
    solar_insights_mcp_client = await MCPHttp.create(url="http://localhost:8000/sse")
    print("Invoked Solar Insights MCP Tool here")

    try:
        solar_insights_action_group = ActionGroup(
            name="SolarInsightsGroup",
            mcp_clients=[solar_insights_mcp_client],
        )
        response = await InlineAgent(
            # foundation_model="amazon.nova-pro-v1:0", # unable to provide proper inputs to the tools. It complains about input mismatch and not able to translate user inputs into pydantic objects. Claude can do it.
            foundation_model=os.environ.get('solar_insights_mcp_tool_llm', 'anthropic.claude-3-haiku-20240307-v1:0'),
            instruction="""You are a friendly assistant that is responsible for resolving user queries related to Solar insights and potential based on a given address. Keep the response short and to the point within in 5 sentences long that is easy for any speech assistant to respond to the user""",
            agent_name="solar_insights_agent",
            action_groups=[
                solar_insights_action_group,
            ],
        ).invoke(
            input_text=f"What is the solar potential insights for the address {address}?"
        )
    finally:
        await solar_insights_mcp_client.cleanup()

    print("Solar Insights Tool Response: ", response)
    return response

solar_insights_tools:AgentTools = AgentTools(tools=[AgentTool(name="SolarInsights_Tool",
        description="Get the solar potential insights for a given address.",
        func=get_solar_insights
)])

async def bedrock_solar_insights_tool_handler(response: ConversationMessage, conversation: List[Dict[str, Any]]) -> ConversationMessage:
    response_content_blocks = response.content

    tool_results = []
    tool_response = None

    if not response_content_blocks:
        raise ValueError("No content blocks in response")

    for content_block in response_content_blocks:
        if "text" in content_block:
            pass

        if "toolUse" in content_block:
            tool_use_block = content_block["toolUse"]
            tool_use_name = tool_use_block.get("name")

            print("Tool Use Name: ", tool_use_name)
            print("Tool Use Input: ", tool_use_block["input"])

            if tool_use_name == "SolarInsights_Tool":
                tool_response = await get_solar_insights(tool_use_block["input"].get('address'))
                tool_results.append({
                    "toolResult": {
                        "toolUseId": tool_use_block["toolUseId"],
                        "content": [{"text": tool_response}],
                    }
                })

    # Embed the tool results in a new user message
    message = ConversationMessage(
            role=ParticipantRole.USER.value,
            content=tool_results)

    print("Conversation Message: ", message.content)

    return message