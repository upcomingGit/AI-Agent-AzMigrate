from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
import json, os, logging, asyncio
from dotenv import load_dotenv


# Setup logging and load environment variables
logger = logging.getLogger(__name__)
print("[DEBUG] Loading environment variables...")
load_dotenv()
print("[DEBUG] Environment variables loaded.")


# Azure OpenAI configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1-mini")
print(f"[DEBUG] AZURE_OPENAI_ENDPOINT: {AZURE_OPENAI_ENDPOINT}")
print(f"[DEBUG] AZURE_OPENAI_MODEL: {AZURE_OPENAI_MODEL}")


# Initialize Azure credentials
print("[DEBUG] Initializing Azure credentials...")
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)
print("[DEBUG] Azure credentials initialized.")


async def run():
    print("[DEBUG] Initializing Azure OpenAI client...")
    # Initialize Azure OpenAI client
    client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT, 
            api_version="2024-04-01-preview", 
            azure_ad_token_provider=token_provider
        )
    print("[DEBUG] Azure OpenAI client initialized.")

    # MCP client configurations
    print("[DEBUG] Setting up MCP client parameters...")
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@azure/mcp@latest", "server", "start"],
        env=None
    )
    print(f"[DEBUG] MCP server_params: {server_params}")

    print("[DEBUG] Starting stdio_client...")
    async with stdio_client(server_params) as (read, write):
        print("[DEBUG] stdio_client started.")
        async with ClientSession(read, write) as session:
            print("[DEBUG] ClientSession started. Initializing session...")
            await session.initialize()
            print("[DEBUG] Session initialized.")

            # List available tools
            print("[DEBUG] Listing available tools...")
            tools = await session.list_tools()
            print(f"[DEBUG] Number of tools found: {len(tools.tools)}")
            for tool in tools.tools:
                print(f"[DEBUG] Tool: {tool.name} - {tool.description}")

            # Format tools for Azure OpenAI
            print("[DEBUG] Formatting tools for Azure OpenAI...")
            available_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                } for tool in tools.tools
            ]
            print(f"[DEBUG] Available tools formatted: {len(available_tools)}")

            # Start conversational loop
            messages = []
            print("[DEBUG] Entering conversational loop...")
            while True:
                try:
                    user_input = input("\nPrompt: ")
                    print(f"[DEBUG] User input: {user_input}")
                    messages.append({"role": "user", "content": user_input})

                    # First API call with tool configuration
                    print("[DEBUG] Sending first API call to Azure OpenAI...")
                    response = client.chat.completions.create(
                        model = AZURE_OPENAI_MODEL,
                        messages = messages,
                        tools = available_tools)
                    print("[DEBUG] First API call complete.")

                    # Process the model's response
                    response_message = response.choices[0].message
                    print(f"[DEBUG] Model response: {response_message}")
                    messages.append(response_message)

                    # Handle function calls
                    if response_message.tool_calls:
                        print(f"[DEBUG] Tool calls detected: {len(response_message.tool_calls)}")
                        for tool_call in response_message.tool_calls:
                            print(f"[DEBUG] Tool call: {tool_call.function.name}, args: {tool_call.function.arguments}")
                            function_args = json.loads(tool_call.function.arguments)
                            print(f"[DEBUG] Calling tool {tool_call.function.name} with args: {function_args}")
                            result = await session.call_tool(tool_call.function.name, function_args)
                            print(f"[DEBUG] Tool result: {result}")

                            # Add the tool response to the messages
                            messages.append({
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": tool_call.function.name,
                                "content": result.content,
                            })
                            print(f"[DEBUG] Tool response appended to messages.")
                    else:
                        logger.info("No tool calls were made by the model")
                        print("[DEBUG] No tool calls were made by the model.")

                    # Get the final response from the model
                    print("[DEBUG] Sending final API call to Azure OpenAI...")
                    final_response = client.chat.completions.create(
                        model = AZURE_OPENAI_MODEL,
                        messages = messages,
                        tools = available_tools)
                    print("[DEBUG] Final API call complete.")

                    for item in final_response.choices:
                        print(f"[DEBUG] Final response: {item.message.content}")
                        print(item.message.content)
                except Exception as e:
                    logger.error(f"Error in conversation loop: {e}")
                    print(f"[ERROR] An error occurred: {e}")

if __name__ == "__main__":
    print("[DEBUG] Starting main...")
    import asyncio
    asyncio.run(run())
    print("[DEBUG] main.py execution finished.")