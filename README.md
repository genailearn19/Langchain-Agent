# Langchain-Agent
Deployed and can be accessible - https://langchain-agent-51d5.onrender.com/ 


**#1. os**
Python built-in module
Used to access environment variables and operating-system settings.
Example: os.environ["OPENAI_API_KEY"]

**#2. requests**
HTTP library
Used to send HTTP requests to external APIs, such as a weather service.
Example: requests.get(url)

**#3. certifi**
SSL certificate bundle
Provides trusted Certificate Authority (CA) certificates to help verify HTTPS connections.

**#4. load_dotenv**
Loads configuration values and API keys from a .env file into environment variables.

**#5. ChatOpenA**I
Connects your LangChain application to an OpenAI chat model.

**#6. AgentExecutor, create_tool_calling_agent, tool**
Used to create an agent that decides when to call tools and executes those tool calls.

**#7. ChatPromptTemplate, MessagesPlaceholder**
Define instructions for the AI and provide a place for conversation history or intermediate agent messages.

**#8. TavilySearch**
Gives your agent access to web-search functionality through the Tavily search service.
