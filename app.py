import streamlit as st
from operator import itemgetter
from langchain_community.retrievers import AzureAISearchRetriever
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
import warnings
from dotenv import load_dotenv
import os

load_dotenv() 

warnings.filterwarnings("ignore")

memory = ConversationBufferMemory(
    memory_key = "chat_history", return_messages = True, output_key = "answer"
)

model = ChatOpenAI(model="gpt-3.5-turbo",temperature=0.4)

def format_docs(d):
    return str(d.input)

def load_chain(question):
    template: str = """/
        You are a helpful otc medicine expert / 
        You provides correct information to the users /
        question: {question}. You assist users with general inquiries based on {context} /
        Additionally, advise the user to consult their doctor if symptoms continue /
    """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        input_variables = ["question", "context"],
        template = "{question}"
    )
    chat_prompt_template = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    
    retriever = AzureAISearchRetriever(index_name="rag-app-vectordb",content_key = "content", top_k = 2)
    
    chain = {
            "context": RunnablePassthrough() | retriever | (lambda docs: str(docs)),
            "question": RunnablePassthrough(),
            "history": RunnableLambda(memory.load_memory_variables) | (lambda x: x.get("history", [])),
        } | chat_prompt_template | model | StrOutputParser()

    return chain.invoke(question)
    

st.markdown(
    """
    <style>
    .box-container {
        background-color: #f5f5f5;
        border: 1px solid #ccc;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .box-container h1, .box-container h2 {
        margin-top: 0;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="box-container">
        <h1>PharmaBot 🤖</h1>
        <h3>Your on-the-go otc medicine assistant in the Philippines 💊</h3>
    </div>
    """,
    unsafe_allow_html=True
)


if "greeting" not in st.session_state:
    st.session_state.greeting = True
if "conversation" not in st.session_state:
    st.session_state.conversation = []

if st.session_state.greeting is not None:
    with st.chat_message("assistant"):
        st.write("Hello! How can I assist you today?")
    st.session_state.greeting = False

for chat in st.session_state.conversation:
    with st.chat_message("user"):
        st.write(chat["query"])
    with st.chat_message("assistant"):
        st.write(chat["answer"])
        
user_input = st.chat_input("What's your question?")
if user_input:
    with st.spinner("Thinking..."):
    
        answer = load_chain(user_input)
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            st.write(answer)
        
        st.session_state.conversation.append({
            "query": user_input,
            "answer": answer
        })

    
