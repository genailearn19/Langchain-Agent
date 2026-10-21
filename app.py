import os
import requests
import certifi
import streamlit as st

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_tavily import TavilySearch

# ------------------------------------------------------------
# Page configuration and polished UI
# ------------------------------------------------------------
st.set_page_config(
    page_title="Eswara AI Explorer | Search & Weather",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
      html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
      .stApp { background: radial-gradient(circle at 8% 0%, rgba(82,104,255,.10), transparent 28%), #f7f8fc; }
      [data-testid="stSidebar"] { background: #101827; }
      [data-testid="stSidebar"] * { color: #edf2ff; }
      .hero {
        padding: 1.7rem 1.8rem; border-radius: 24px; color: white;
        background: linear-gradient(125deg, #111827 0%, #1e3a8a 58%, #4f46e5 100%);
        box-shadow: 0 14px 38px rgba(30,58,138,.18); margin-bottom: 1.25rem;
      }
      .hero h1 { font-family: 'Space Grotesk', sans-serif; font-size: clamp(2rem, 4vw, 3rem); margin: 0 0 .4rem 0; }
      .hero p { color: #dbeafe; margin: 0; font-size: 1.02rem; }
      .eyebrow { color: #a5b4fc; text-transform: uppercase; letter-spacing: .15em; font-size: .75rem; font-weight: 700; }
      .feature {
        background: white; border: 1px solid #e8ebf4; border-radius: 18px;
        padding: 1rem 1.1rem; min-height: 125px; box-shadow: 0 5px 18px rgba(15,23,42,.035);
      }
      .feature-icon { font-size: 1.5rem; }
      .feature-title { font-weight: 700; color: #172033; margin: .35rem 0 .2rem; }
      .feature-desc { color: #64748b; font-size: .88rem; }
      .small-muted { color: #64748b; font-size: .85rem; }
      div[data-testid="stChatMessage"] { border: 1px solid #e9edf5; border-radius: 18px; padding: .7rem 1rem; background: rgba(255,255,255,.85); }
      .stButton button { border-radius: 12px; font-weight: 600; }
      .st-key-quick-prompts .stButton button {
        background-color: #FDBA74; border-color: #FB923C; color: #431407;
      }
      .st-key-quick-prompts .stButton button:hover {
        background-color: #FB923C; border-color: #F97316; color: #431407;
      }
      div[data-testid="stChatInput"] { border-radius: 18px; }
      .status-pill { display:inline-block; padding: .3rem .65rem; border-radius: 999px; background: rgba(34,197,94,.14); color: #bbf7d0; font-size:.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Use the certificate bundle supplied by certifi and load local environment variables.
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


@tool
def get_weather_data(city: str) -> str:
    """Fetch current weather information for a city. Input should be a city name."""
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
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Fog", 48: "Depositing rime fog", 51: "Light drizzle",
        53: "Moderate drizzle", 55: "Dense drizzle", 56: "Light freezing drizzle",
        57: "Dense freezing drizzle", 61: "Slight rain", 63: "Moderate rain",
        65: "Heavy rain", 66: "Light freezing rain", 67: "Heavy freezing rain",
        71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow", 77: "Snow grains",
        80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
        85: "Slight snow showers", 86: "Heavy snow showers", 95: "Thunderstorm",
        96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
    }
    return (
        f"City: {location['name']}, {location.get('country', 'Unknown country')}\n"
        f"Temperature: {current['temperature_2m']}°C\n"
        f"Weather: {weather_descriptions.get(current['weather_code'], 'Unknown conditions')}\n"
        f"Humidity: {current['relative_humidity_2m']}%\n"
        f"Observed at: {current['time']}"
    )


@st.cache_resource(show_spinner=False)
def build_agent(openai_key: str, tavily_key: str):
    """Create the LangChain agent once per Streamlit session/runtime."""
    search_tool = TavilySearch(max_results=3, api_key=tavily_key)
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=openai_key,
    )
    tools = [search_tool, get_weather_data]
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are AI Explorer, a helpful assistant. Use the available tools "
                "when needed. For current events or web facts, use web search. For "
                "weather, use get_weather_data. Clearly distinguish sourced facts "
                "from uncertainty and keep answers well structured. Use Markdown.",
            ),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=6,
    )


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("## ✨Eswara AI Explorer")
    st.caption("Search smarter. Understand faster.")
    st.markdown("---")
    st.markdown("### Workspace")
    if st.button("＋  New conversation", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("### Connected tools")
    st.markdown("🌐 **Tavily Search**")
    st.caption("Web search for up-to-date information")
    st.markdown("🌦️ **Open-Meteo**")
    st.caption("Current weather and humidity")
    st.markdown("🧠 **GPT-4o**")
    st.caption("Tool-calling language model")
    st.markdown("---")
    st.markdown("### Quick prompts")
    quick_prompts = [
        "Find the capital of France and its current weather.",
        "What are the latest developments in generative AI?",
        "Compare current weather in Hyderabad and London.",
        "Explain how RAG works with a simple example.",
    ]
    with st.container(key="quick-prompts"):
        for i, q in enumerate(quick_prompts):
            if st.button(q, key=f"quick_{i}", use_container_width=True):
                st.session_state.pending_prompt = q
                st.rerun()
    st.markdown("---")
    st.caption("API keys are read from your local `.env` file. Never commit that file to Git.")

# ------------------------------------------------------------
# Main hero and feature cards
# ------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Eswara AI-powered workspace</div>
      <h1>Ask. Explore. Discover.</h1>
      <p>One assistant for live web research, weather insights, and everyday questions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="feature"><div class="feature-icon">🌐</div><div class="feature-title">Live web search</div><div class="feature-desc">Research current topics with Tavily-powered search.</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="feature"><div class="feature-icon">🌤️</div><div class="feature-title">Current weather</div><div class="feature-desc">Get temperature, humidity, and conditions for a city.</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="feature"><div class="feature-icon">💬</div><div class="feature-title">Natural conversation</div><div class="feature-desc">Ask follow-up questions in the same chat session.</div></div>', unsafe_allow_html=True)

st.write("")
st.markdown("### 💬 Your conversation")
st.markdown('<div class="small-muted">Ask a question below. The assistant will choose the right tool when needed.</div>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "✨"):
        st.markdown(message["content"])

pending = st.session_state.pending_prompt
if pending:
    st.session_state.pending_prompt = None

user_input = st.chat_input("Ask about anything, or request the weather in a city…")
if pending and not user_input:
    user_input = pending

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="✨"):
        if not OPENAI_API_KEY or not TAVILY_API_KEY:
            answer = (
                "To start the assistant, add `OPENAI_API_KEY` and `TAVILY_API_KEY` "
                "to your `.env` file in the project folder, then restart Streamlit."
            )
            st.warning(answer)
        else:
            try:
                with st.spinner("Thinking and checking the right tools…"):
                    executor = build_agent(OPENAI_API_KEY, TAVILY_API_KEY)
                    result = executor.invoke({"input": user_input})
                    answer = result.get("output", "I couldn't generate a response.")
                st.markdown(answer)
            except Exception as exc:
                answer = (
                    "Sorry, the request failed. Check your API keys, package versions, "
                    "and network connection, then try again.\n\n"
                    f"**Technical detail:** `{type(exc).__name__}: {exc}`"
                )
                st.error(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

if not st.session_state.messages:
    st.markdown(
        """
        <div style="margin-top:1rem;padding:1.2rem 1.4rem;border:1px dashed #cbd5e1;border-radius:18px;background:rgba(255,255,255,.7)">
          <div style="font-weight:700;color:#172033;margin-bottom:.35rem">Ready when you are 👋</div>
          <div style="color:#64748b">Try asking: <b>“Find the capital of France and find the current weather in that city.”</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
