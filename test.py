import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage
import sys

def main():
    load_dotenv()
    if len(sys.argv) > 1:
        user_name = sys.argv[1]

    st.header("Welcome to the Clickatell Chat Bot, How may we assist you " + user_name + "?")

    embeddings = OpenAIEmbeddings()
    
    vectorstore = FAISS.load_local("Info", embeddings)

    llm = ChatOpenAI(max_tokens=200)

    style = f"""
        <style>
            .stTextInput {{
            position: fixed;
            bottom: 3rem;
            }}
        </style>
        """
    st.markdown(style, unsafe_allow_html=True)

    if "chat_history" not in st.session_state:       
        history = ChatMessageHistory()

        st.session_state.memory = ConversationBufferMemory(
            memory_key='chat_history', return_messages=True, chat_memory=history)
        
        general_system_template = r""" 
        You are a helpful assistant for a company called Clickatell, use friendly language. 
        Use the following pieces of context to answer the users question. 
        Do not make up answers that are not provided in the documentation. 
        You must always following these rules:
        - If you dont have an answer for the question and it is not written in the context, say "I'm sorry but I do not have information about that, Would you like to chat with a live agent for further assistance?"
        - If you dont know the answer say "I'm sorry but I do not have information about that, Would you like to chat with a live agent for further assistance?"
        - If the question doesnt relate to your context, say "I'm sorry but I do not have information about that, Would you like to chat with a live agent for further assistance?"
        - If the user asks a question that is not within Clickatell services, say "I'm sorry but I do not have information about that, Would you like to chat with a live agent for further assistance?", if you already asked
          if they want to chat to a live agent it and the user said yes, simply say "Transfer successful" if you already asked if the user want to chat to 
          a live agent it and the user said no, simply say "How else can Clickatell help you?"  
        ----
        {context}
        ----
        """
        
        general_user_template = r"""Follow these rules:
                                        The users name is """ + user_name + """. Use it when you answer questions. Make a 50% chance of whether you will use their name in the chat or not
                                        Just answer the question that is asked, do not state that you are there to help, the user knows that already
                                        - Answer all questions in english 
                                        - If the user does not use question mark at the end of their statement, infer whether the user is asking a question or not
                                        - Do not ask follow up questions
                                        - The chat history has a specific structure, keep it in mind in case you need to know which question was asked at what time
                                        - Consider the chat history when asnwering a question:
                                        Chat History also known as conversation: {chat_history}
                                        Follow up question: {question}"""

        messages = [
            SystemMessagePromptTemplate.from_template(general_system_template),
            HumanMessagePromptTemplate.from_template(general_user_template)
        ]

        qa_prompt = ChatPromptTemplate.from_messages( messages )

        if "conversation" not in st.session_state:
            st.session_state.conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                memory=st.session_state.memory,
                combine_docs_chain_kwargs={'prompt': qa_prompt},
                verbose=True
            )

    user_question = st.text_input("")

    if user_question:
        response = st.session_state.conversation_chain({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        i=0
        for msg in st.session_state.chat_history:
            if isinstance(msg, HumanMessage):
                message(msg.content, is_user=True, key="uniqueHuman" + str(i))
                i=i+1
            elif isinstance(msg, AIMessage):
                message(msg.content, is_user=False, key="uniqueAI" + str(i))
                i=i+1
                contain_word = "live agent" in msg.content and "I do not have information" in msg.content and msg.content.endswith("?")
                if contain_word:
                    if st.button("Transfer", key="uniqueStopButton" + str(i)):
                        i=i+1
                        st.write("Transfer complete")
                        st.stop()
   
if __name__ == '__main__':
    main()