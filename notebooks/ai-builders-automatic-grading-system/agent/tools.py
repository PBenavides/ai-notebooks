#RAG Related Tools
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
import os
import logging
from uuid import uuid4
from dotenv import load_dotenv
load_dotenv()
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import InjectedState
from langchain_openai import ChatOpenAI
from typing import Annotated
import os
import json


from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

#Objects:
embeddings = OpenAIEmbeddings(model="text-embedding-3-large",
                              api_key=os.environ.get('OPENAI_API_KEY'))

#Retrieval function
def create_load_vector_store(lessonid: str, lesson_doc: str):

    #We are setting the document as long as possible for this.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5_000, chunk_overlap=1_500, add_start_index=True
    )

    all_splits = text_splitter.create_documents([lesson_doc])

    if f'{lessonid}_index' not in os.listdir():
        
        uuids = [str(uuid4()) for _ in range(len(all_splits))]
        #define vector store:
        index = faiss.IndexFlatL2(len(embeddings.embed_query("hello berlin ai builders!")))

        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        uuids = [str(uuid4()) for _ in range(len(all_splits))]
        vector_store.add_documents(documents=all_splits, ids=uuids)

        return vector_store

    else:
        vector_store = FAISS.load_local(
            f'{lessonid}_index',
            embeddings,
            allow_dangerous_deserialization=True
        )

        return vector_store
    
def retrieve_result(query:str, vector_store) -> str:

    results = vector_store.similarity_search(
        query,
        k=3,
    )

    return results[0].page_content

@tool()
def query_lesson(question: str, state: Annotated[dict, InjectedState]): #Add a fixed parameter?
    """Ask a specific question regarding the lesson.
    This tool allows you to ask any question regarding the lesson being taught to the student. 
    It retrieves the relevant context from the lesson document and provides an AI-generated answer.
    Args:
        question (str): The question to ask in the context of the lesson.
    Returns:
        dict: A dictionary containing the question and the generated answer in the scratchpad.
    Example:
        >>> query_lesson("What is the main topic covered in this lesson?")
    """
    lesson_document = state['lesson_doc']
    #Retrieval
    vector_store = create_load_vector_store('lesson1', lesson_document)
    context_retrieved = retrieve_result(question, vector_store)
    #Augmented:
    rag_prompt = """
        You are a helpful AI assistant, please respond to the users query to the best of your ability!
        You should response to the user in a way that is clear, concise and is aligned with the context. 
        Remember that all the questions are going to be attached to the given context.
        """
    system_prompt = SystemMessage(rag_prompt)
    human_msg = HumanMessage(
        content=f"""
            Context: {context_retrieved} Question to answer: {question}"
        """
    )
    #Generation:
    model = ChatOpenAI(model="gpt-4o-mini", api_key=os.environ.get('OPENAI_API_KEY'))
    ai_msg = model.invoke([system_prompt, human_msg])
    answer = ai_msg.content
    print(f"The answer: {answer}")
    q_and_a_response = {'messages': f"Question: {question} Answer: {answer}"}
    return q_and_a_response