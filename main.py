import os
import requests
import certifi
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_tavily import TavilySearch


# ==========================================================
# Load environment variables from .env file
#==========================================================
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# ===============================================================
# Initialize the Tavily search tool with API key and max results.
#================================================================
search_tool = TavilySearch(max_results=3, api_key=TAVILY_API_KEY)

#=============================================================
# Define a tool to fetch current weather using Open-Meteo.
#=============================================================
@tool
def get_weather_data(city: str) -> str:
    """Fetch current weather information for a city."""
    try:
        location_response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=15,
        )
        location_response.raise_for_status()
        locations = location_response.json().get("results", [])

        if not locations:
            return f"No location found for {city}."

        location = locations[0]
        weather_response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "current": "temperature_2m,relative_humidity_2m,weather_code",
                "timezone": "auto",
            },
            timeout=15,
        )
        weather_response.raise_for_status()
        current = weather_response.json()["current"]
    except requests.RequestException as exc:
        return f"Weather lookup failed for {city}: {exc}"
    except (KeyError, ValueError) as exc:
        return f"Weather service returned an invalid response for {city}: {exc}"

    weather_descriptions = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        56: "Light freezing drizzle",
        57: "Dense freezing drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        66: "Light freezing rain",
        67: "Heavy freezing rain",
        71: "Slight snow",
        73: "Moderate snow",
        75: "Heavy snow",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail",
    }

    return (
        f"City: {location['name']}, {location.get('country', 'Unknown country')}\n"
        f"Temperature: {current['temperature_2m']}°C\n"
        f"Weather: {weather_descriptions.get(current['weather_code'], 'Unknown conditions')}\n"
        f"Humidity: {current['relative_humidity_2m']}%\n"
        f"Observed at: {current['time']}"
    )


#=============================================================
# Initialize the ChatOpenAI model with GPT-4o.
#=============================================================
llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)

#=============================================================
# Create the AgentExecutor-compatible tool-calling agent.
#=============================================================
tools = [search_tool, get_weather_data]


#====================================================================
# Prompt template for the agent, including system and human messages.
#====================================================================
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. "
            "Use the available tools to answer questions accurately.",
        ),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
#=============================================================
# Create the agent and executor.
#=============================================================
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)
#=============================================================
# Create the AgentExecutor to handle the agent's execution.
#=============================================================
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,handle_parsing_errors=True )

response = agent_executor.invoke({"input": "Find the capital of France and find the current weather in that city."})

print(response["output"])