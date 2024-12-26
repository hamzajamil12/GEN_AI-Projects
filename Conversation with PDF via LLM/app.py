import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
import chromadb

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv
import os

load_dotenv()
os.environ['HUGGINGFACE_API_KEY'] = os.getenv('HUGGINGFACE_API_KEY')
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# setting streamlit
st.title("Conversation with PDF via LLM")
st.write("Upload a PDF file to start a conversation")

# input the groq api key
api_key = st.text_input("Enter your GROQ API Key", type="password")

# checking if the api key is entered
if api_key:
    llm = ChatGroq(groq_api_key=api_key,model='llama-3.1-8b-instant')

    # Session
    session_id = st.text_input("Session ID", value="default_session")
    # statefully managed chat history (Have to think about this logic)
    if 'store' not in st.session_state:
        # I am creating this empty dict so i can store the chat history session
        st.session_state.store = {}

        # lets take the file
    uploaded_files = st.file_uploader("Choose a PDF file", type="pdf",accept_multiple_files=True)
    # lets apply conditions
    if uploaded_files:
        document=[]
        # lets loop over file
        for uploaded_file in uploaded_files:
            tempdf = f'./temp.pdf'
            with open(tempdf, 'wb') as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name

            loader = PyPDFLoader(tempdf)
            docs = loader.load()
            document.extend(docs)
        # split and embed the document text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(document)
        # This line of code just clears the cache of the system (just to remove tenat error)
        chromadb.api.client.SharedSystemClient.clear_system_cache()        

        vectorestore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorestore.as_retriever()
    
    ##################################################################################################
    
        # Lets write contextualized prompt for system
        context_q_system_prompt = (
            "Given a chat history and the latest question user asked"
            "You can also check from chat history and then give correct answer to user"
            "Without chat history you can not give answer"
            "Just give it back if needed"
        )
        # Lets write contextualized prompt
        context_q_prompt = ChatPromptTemplate(
            [
                ("system", context_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        # History Aware retriever
        history_aware_ret = create_history_aware_retriever(llm, retriever, context_q_prompt)

        # Answer the question
        # First lets create system prompt
        system_prompt = (
            "You are an assistant for question-answering task"
            "Use the context given below as for your answer"
            "If you can not find answer, just give it back and say i dont know"
            "Use three sentences maximum if you dont know the answer"
            "Answer the question correctly"
            "\n\n"
            "{context}"
        )
        # Lets create Question-Answer Prompt
        q_a_prompt = ChatPromptTemplate(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat history"),
                ("human", "{input}")
            ]
        )

        # First create queation answer chain
        q_a_chain = create_stuff_documents_chain(llm,q_a_prompt)
        # After creating question answer chain, lets create rag chain with history aware retriever and question answer chain
        rag_chain = create_retrieval_chain(history_aware_ret, q_a_chain)

        # create session function to get session history
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Ask a question")
        if user_input:
            # Lets run the chain
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id": session_id}
                }
            )

            # Lets display the answer
            # st.write(st.session_state.store)
            st.write("Assistant:", response["answer"])
            # st.write("Chat History:", session_history.messages)
