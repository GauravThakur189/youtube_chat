from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import Optional
import os
from dotenv import load_dotenv
import re
import uuid

app = FastAPI()
load_dotenv()

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# Prompt setup
prompt = PromptTemplate(
    template="""
    You are a helpful assistant.
    Answer ONLY from the provided transcript context.
    If the context is insufficient, just say you don't know.

    {context}

    Question: {question}
    """,
    input_variables=['context', 'question']
)

# In-memory store for vector DBs per video
VECTOR_DB_PATH = "vectorstores"
os.makedirs(VECTOR_DB_PATH, exist_ok=True)
video_vectorstores = {}

class VideoInput(BaseModel):
    youtube_url: str

class QuestionInput(BaseModel):
    video_id: str
    question: str

# Utility functions

def extract_video_id(youtube_url: str) -> Optional[str]:
    match = re.search(r"(?:v=|youtu.be/)([\w-]{11})", youtube_url)
    return match.group(1) if match else None

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

@app.post("/process_video")
def process_video(data: VideoInput):
    video_id = extract_video_id(data.youtube_url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id=video_id, languages=["en"])
    except TranscriptsDisabled:
        raise HTTPException(status_code=404, detail="Transcript not available for this video")

    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    vector_store = FAISS.from_documents(chunks, embedding=embedding_model)
    store_path = os.path.join(VECTOR_DB_PATH, video_id)
    vector_store.save_local(store_path)

    return {"message": "Transcript processed and vector store saved.", "video_id": video_id}

# @app.post("/ask")
# def ask_question(data: QuestionInput):
#     print("Received question for video:", data.video_id)
#     store_path = os.path.join(VECTOR_DB_PATH, data.video_id)
#     if not os.path.exists(store_path):
#         raise HTTPException(status_code=404, detail="Vector store not found for this video")

#     vector_store = FAISS.load_local(store_path, embeddings=embedding_model)
#     retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

#     parallel_chain = RunnableParallel({
#         'context': retriever | RunnableLambda(format_docs),
#         'question': RunnablePassthrough()
#     })

#     main_chain = parallel_chain | prompt | llm | StrOutputParser()
#     answer = main_chain.invoke(data.question)

#     return {"answer": answer, "video_id": data.video_id, "question": data.question}



@app.post("/ask")
def ask_question(data: QuestionInput):
    video_id = data.video_id
    question = data.question

    store_path = f"./vectorstores/{video_id}"
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    # ✅ Add the flag below
    vector_store = FAISS.load_local(
        store_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.
        {context}
        Question: {question}
        """,
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    }) | prompt | llm | StrOutputParser()

    result = chain.invoke(question)
    return {"answer": result}
