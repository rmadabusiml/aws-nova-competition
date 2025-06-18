from mcp import StdioServerParameters

from InlineAgent.tools import MCPHttp
from InlineAgent.action_group import ActionGroup
from InlineAgent.agent import InlineAgent

async def main():
    # Step 2: Create MCP Client
    solar_mcp_client = await MCPHttp.create(url="http://localhost:8000/sse")

    try:
        # Step 3: Define an action group
        solar_action_group = ActionGroup(
            name="SolarActionGroup",
            description="Helps user to get solar potential insights for location.",
            mcp_clients=[solar_mcp_client],
        )

        # Step 4: Invoke agent
        await InlineAgent(
            # Step 4.1: Provide the model
            foundation_model="anthropic.claude-3-haiku-20240307-v1:0",
            # Step 4.2: Concise instruction
            instruction="""You are a friendly assistant that provides solar potential insights for a location. The response should contain user friendly summary of the following information without lot of technical details:
            1. Solar potential insights such as max panels, panel dimensions, system lifespan, roof area, annual sunlight hours, median panels config, energy profiles, annual carbon offset
            2. Financial analysis such as monthly bill, savings, payback years, federal incentive, state incentive, utility incentive, net cost
            3. Roof analysis such as pitch, azimuth, solar exposure, orientation quality
            """,
            # Step 4.3: Provide the agent name and action group
            agent_name="solar_agent",
            action_groups=[solar_action_group],
        ).invoke(
            input_text="What is the solar potential insights for the address 1364, Westborough Lane, Leander, Texas, 78641?"
        )

    finally:

        await solar_mcp_client.cleanup()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
