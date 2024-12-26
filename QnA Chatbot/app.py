import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

# Langsmith Tracking
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')
# os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

# Lets build a prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ('system', "You are helping a user with a question. Answer them correctly"),
        ("user", "Question: {question}"),
    ]
)

# function
def generate_response(question, api_key,engine,temperature,max_token):

    # Lets take the key
    groq_api_key = api_key
    llm = ChatGroq(model=engine, api_key=groq_api_key, temperature=temperature, max_tokens=max_token)
    outputParser = StrOutputParser()
    # Lets build a chain
    chain = prompt | llm | outputParser
    answer = chain.invoke({"question": question})
    return answer  

# Lets create frontend with streamlit
# Title
# st.title("QnA Chatbot with opensource GROQ API")

# sidebar
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your GROQ API Key", type="password")

# Select the Groq model
engine = st.sidebar.selectbox("Select the Engine", ["gemma2-9b-it", "llama-3.1-8b-instant",])

# Select the temperature and other params
temperature = st.sidebar.slider("Select the temperature", 0.0, 1.0, 0.5)
max_token = st.sidebar.slider("Select the max token", 0, 100, 50)

# Main interface for the user
st.write("# QnA Chatbot: Ask me anything")
user_input = st.text_input("Enter your question here:")

if user_input and api_key:
    response = generate_response(user_input, api_key, engine, temperature, max_token)
    st.write(f"Answer: {response}")

elif not api_key:
    st.write("Please enter your GROQ API Key in the settings")