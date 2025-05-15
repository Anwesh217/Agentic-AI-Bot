import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from typing import List
import pytz
from datetime import datetime
import yaml


# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set page configuration
st.set_page_config(page_title="PDF Chatbot", layout="wide")

# App title and description
st.title("📄 PDF Chatbot")
st.markdown("""
Upload PDFs and chat with their content using AI. The system will:
1. Process your PDFs
2. Extract and organize the content
3. Allow you to ask questions about the documents
""")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "file_processed" not in st.session_state:
    st.session_state.file_processed = False
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None


# Function to extract text from PDFs
def extract_text_from_pdf(uploaded_file) -> List[str]:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name
        loader = PyPDFLoader(temp_path)
        documents = loader.load()
        os.remove(temp_path)
        return documents
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return []


# Function to process PDFs and create vector store
def process_pdf(uploaded_file):
    with st.spinner("Processing PDF... This may take a moment."):
        documents = extract_text_from_pdf(uploaded_file)
        if not documents:
            st.error("Could not extract text from the PDF. Please try another file.")
            return None
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        if not OPENAI_API_KEY:
            st.error("OpenAI API key not found in .env file.")
            return None
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store


# Function to create the LLM
def create_llm():
    if not OPENAI_API_KEY:
        st.error("OpenAI API key not found in .env file.")
        return None
    try:
        llm = ChatOpenAI(
            temperature=0.2,
            api_key=OPENAI_API_KEY,
            model_name="gpt-3.5-turbo"
        )
        return llm
    except Exception as e:
        st.error(f"Error creating LLM: {e}")
        return None


# Function to create agent tools
def create_tools(vector_store):
    llm = create_llm()
    if not llm:
        return []

    # Define the general knowledge tool first
    def general_knowledge_tool(query: str) -> str:
        """Tool to answer general knowledge questions."""
        if llm:
            return llm.invoke(query).content
        else:
            return "Could not generate a general response due to model error."

    general_knowledge = Tool(
        name="General_Knowledge",
        func=general_knowledge_tool,
        description="Useful for answering general knowledge questions when no relevant PDF content is found."
    )

    # Define the document-based tools
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    def get_relevant_documents(query: str) -> str:
        docs = retriever.get_relevant_documents(query)
        if not docs:
            return "No relevant information found."
        result = []
        for i, doc in enumerate(docs):
            result.append(f"Document {i+1}:\n{doc.page_content}\n")
        return "\n".join(result)

    search_tool = Tool(
        name="Document_Search",
        func=get_relevant_documents,
        description="Useful for searching information in the documents."
    )

    summarize_prompt = PromptTemplate(
        template="Summarize the following document: {context}",
        input_variables=["context"]
    )

    def summarize_doc(query):
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        summary_chain = summarize_prompt | llm
        return summary_chain.invoke({"context": context})

    summarize_tool = Tool(
        name="Summarize",
        func=summarize_doc,
        description="Useful for summarizing sections of the document."
    )

    def get_ireland_datetime(_input: str) -> str:
        tz = pytz.timezone("Europe/Dublin")
        now = datetime.now(tz)
        return now.strftime("Current date and time in Ireland: %A, %d %B %Y, %H:%M:%S")
    
    datetime_tool = Tool(
        name="Get_Current_Ireland_DateTime",
        func=get_ireland_datetime,
        description="Use this to get the current date and time in Ireland."
    )

    # Return the list of all tools including general knowledge tool
    return [general_knowledge, search_tool, summarize_tool, datetime_tool]

import yaml

def load_prompt_from_yaml(yaml_path):
    with open(yaml_path, 'r', encoding="utf-8") as file:
        data = yaml.safe_load(file)
    prompt_template = data.get("prompt_template")
    input_variables = data.get("input_variables")

    if not prompt_template or not input_variables:
        raise ValueError("YAML file must include 'prompt_template' and 'input_variables'.")

    return prompt_template, input_variables



# Function to create the agent with a modified prompt
def create_agent(tools, llm):

    prompt_template_str, input_vars = load_prompt_from_yaml("prompt.yaml")

    prompt = ChatPromptTemplate.from_template(prompt_template_str).partial(
        tools="\n".join(f"{tool.name}: {tool.description}" for tool in tools),
        tool_names=", ".join(tool.name for tool in tools),
        # agent_scratchpad is dynamically injected by LangChain
    )

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True  # Keep intermediate steps for debugging
    )
    return agent_executor


# Function to handle the chat with fallback to general LLM
def chat_with_pdf(agent_executor, query):
    try:
        # Invoke the agent with the input query
        response = agent_executor.invoke({"input": query})

        # Show intermediate reasoning steps and tool usage
        intermediate_steps = response.get("intermediate_steps", [])
        if intermediate_steps:
            with st.expander("🔍 Agent's Reasoning & Tools Used"):
                for i, step in enumerate(intermediate_steps):
                    action, observation = step
                    st.markdown(f"""
                    **Step {i+1}:**
                    - **Tool:** `{action.tool}`
                    - **Input:** `{action.tool_input}`
                    - **Observation:** `{observation}`  
                    """)
        else:
            st.info("ℹ️ No tools were used by the agent in this response.")

        # Extract the final answer from the response
        answer = response.get("output", "").strip()

        # Fallback logic for weak or missing answers
        if not answer or "not found in the document" in answer.lower() or "no relevant" in answer.lower():
            st.info("⚠️ No relevant information found in the document. Falling back to general knowledge...")
            llm = create_llm()
            if llm:
                try:
                    return llm.invoke(query).content
                except Exception as gen_error:
                    return f"General model fallback also failed: {gen_error}"
            else:
                return "⚠️ Could not generate a general response due to model initialization error."

        return answer

    except Exception as e:
        st.warning(f"⚠️ Agent execution failed, falling back to general knowledge: {e}")
        llm = create_llm()
        if llm:
            try:
                return llm.invoke(query).content
            except Exception as fallback_error:
                return f"⚠️ General fallback model also failed: {fallback_error}"
        else:
            return f"⚠️ Model error: {str(e)}"



# Sidebar for PDF upload (API key is now read from .env)
with st.sidebar:
    st.header("Configuration")
    st.markdown("OpenAI API key will be read from the `.env` file.")
    st.markdown("Make sure you have a `.env` file in the same directory with `OPENAI_API_KEY=YOUR_KEY`")
    st.divider()

    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file and st.button("Process PDF"):
        vector_store = process_pdf(uploaded_file)
        if vector_store:
            st.session_state.vector_store = vector_store
            st.session_state.file_processed = True
            st.success("PDF processed successfully! You can now chat with the document.")
            st.session_state.messages = []
            st.session_state.conversation_memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            llm = create_llm()
            if llm:
                tools = create_tools(st.session_state.vector_store)
                st.session_state.agent_executor = create_agent(tools, llm)
            else:
                st.error("Failed to initialize the language model.")
        else:
            st.error("PDF processing failed.")

# Main chat interface
st.header("Chat with PDF or ask general questions")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_query = st.chat_input("Ask a question...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    if st.session_state.agent_executor:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_with_pdf(st.session_state.agent_executor, user_query)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.conversation_memory.save_context(
                    {"input": user_query},
                    {"answer": response}
                )
    else:
        # Fallback to general ChatOpenAI response if no PDF is processed yet
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                llm = create_llm()
                if llm:
                    try:
                        response = llm.invoke(user_query).content
                        st.write(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.error("Language model could not be initialized. Check API key.")
