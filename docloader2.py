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
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from PyPDF2 import PdfReader




load_dotenv()

# if len(sys.argv) != 2:
#     print("Usage: python script.py <username>")
#     sys.exit(1)

# username = sys.argv[1]

#print(f"Hello, {username}! This is your Python application.")

documents = []
# Create a List of Documents from all of our files in the ./docs folder
for file in os.listdir("docs"):
    if file.endswith(".pdf"):
        pdf_path = "./docs/" + file
        text = ""
    
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text += page.extract_text()
    
        # loader = PyPDFLoader(pdf_path)
        # documents.extend(loader.load())

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
text = text_splitter.split_text(text)

# Convert the document chunks to embedding and save them to the vector store
vectordb = Chroma.from_texts(text, embedding=OpenAIEmbeddings(), persist_directory="./data")
vectordb.persist()

memory = ConversationBufferMemory(memory_key='chat_history')


name = input("enter name ")
general_system_template = r""" 
    You are a helpful assistant for a compay called Clickatell, use formal language.
    Provide the user with information about clickatells products for example. 
    Do not make up answers that are not provided in the documentation. 
    You must always follow these rules:
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
                            The users name is """+ name + """ use it when you answer questions. make a 50% chance of whether you will use their name in the chat and answer the question that is asked, 
                            do not state that you are there to help, the user knows that already
                            - answer all questions in english
                            - if the user does not use a question mark at the end of their sentence, infer whether the user is asking a question or not
                            - do not ask follow up questions
                            - the chat history has a specific structure, keep it in mind in case you need to know which question was asked at what time
                            - consider the chat history when answering questions:
                            Chat History also known as conversation: {chat_history}
                            Follow up question: {question}"""

#general_system_template.format(name="Daanyal")
messages = [
    SystemMessagePromptTemplate.from_template(general_system_template),
    HumanMessagePromptTemplate.from_template(general_user_template)
]

#CONDENSE_QUESTION_PROMPT 
qa_prompt = ChatPromptTemplate.from_messages(messages)


# create our Q&A chain
pdf_qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(temperature=0.7,  model_name='gpt-3.5-turbo'),
    retriever=vectordb.as_retriever(search_kwargs={'k': 6}),
    return_source_documents=True,
    verbose=False,
    combine_docs_chain_kwargs= {"prompt" : qa_prompt},
    memory = memory

)

yellow = "\033[0;33m"
green = "\033[0;32m"
white = "\033[0;39m"

chat_history = []
print(f"{yellow}---------------------------------------------------------------------------------")
print('Welcome to Clickbot clickatells first chatbot. Ask me Questions about Clickatell')
print('---------------------------------------------------------------------------------')
#user_name = input("before we begin please enter your name: ")
print("Welcome "+ name +" to clickatells personal assistant")
while True:
    query = input(f"{green}Prompt: ")
    if query == "exit" or query == "quit" or query == "q" or query == "f":
        print('Goodbye '+ name + " have a great day!")
        sys.exit()
    if query == '':
        continue
    result = pdf_qa(
        {"question": query, "chat_history": chat_history })
    
    print(f"{white}Answer: " + result["answer"])
    chat_history.append((query, result["answer"]))