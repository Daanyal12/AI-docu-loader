import os
import sys
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
import sys


load_dotenv('.env')
# if len(sys.argv) > 1:
#         name = sys.argv[1]

# print("Welcome to the Clickatell Chat Bot, How may we assist you " + name + "?")

documents = []
# Create a List of Documents from all of our files in the ./docs folder
for file in os.listdir("docs"):
    if file.endswith(".pdf"):
        pdf_path = "./docs/" + file
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    elif file.endswith('.docx') or file.endswith('.doc'):
        doc_path = "./docs/" + file
        loader = Docx2txtLoader(doc_path)
        documents.extend(loader.load())
    elif file.endswith('.txt'):
        text_path = "./docs/" + file
        loader = TextLoader(text_path)
        documents.extend(loader.load())

# Split the documents into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
documents = text_splitter.split_documents(documents)

# Convert the document chunks to embedding and save them to the vector store
vectordb = Chroma.from_documents(documents, embedding=OpenAIEmbeddings(), persist_directory="./data")
vectordb.persist()
# You are a helpful assistant for a company, use formal language.
#     Provide the user with information about the documentation or any related topics about the document. 
#     Do not make up answers that are not provided in the documentation. 
name = input("enter your name ")
history = ChatMessageHistory()

memory = ConversationBufferMemory(
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
                                        The users name is """ + name + """. Use it when you answer questions. Make a 80% chance of whether you will use their name in the chat or not
                                        if the user asks for their name return the user name
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

conversation_chain = ConversationalRetrievalChain.from_llm(
                ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo'),
                retriever=vectordb.as_retriever(search_kwargs={'k': 6}),
                memory=memory,
                combine_docs_chain_kwargs={'prompt': qa_prompt},
                verbose=False
            )


# messages = [
#     SystemMessagePromptTemplate.from_template(general_system_template),
#     HumanMessagePromptTemplate.from_template(general_user_template)
# ]

# qa_prompt = ChatPromptTemplate.from_messages(messages)


# create our Q&A chain
# pdf_qa = ConversationalRetrievalChain.from_llm(
#     ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo'),
#     retriever=vectordb.as_retriever(search_kwargs={'k': 6}),
#     return_source_documents=True,
#     verbose=False,
#     combine_docs_chain_kwargs= {"prompt" : qa_prompt}
# )

yellow = "\033[0;33m"
green = "\033[0;32m"
white = "\033[0;39m"

# chat_history = []
print(f"{yellow}---------------------------------------------------------------------------------")
print('Welcome to Clickatells DocBot. You are now ready to start interacting with your documents')
print('---------------------------------------------------------------------------------')
while True:
    query = input(f"{green}Prompt: ")
    if query == "exit" or query == "quit" or query == "q" or query == "f":
        print('Exiting')
        sys.exit()
    if query == '':
        continue
    response = conversation_chain({'question': query})
    chat_history = response['chat_history']
    print(f"{white}Answer: " + response["answer"])

    # result = conversation_chain(
    #     {"question": query, "chat_history": chat_history})
    # print(f"{white}Answer: " + result["answer"])
    # chat_history.append((query, result["answer"]))