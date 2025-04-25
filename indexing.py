from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

video_id = "EDb37y_MhRw"
try:
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id=video_id,languages=["en"])
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    # print(transcript_list)
except TranscriptsDisabled:
    print("No captions available for this video") 
    

    # splittiing the text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)  
chunks = splitter.create_documents([transcript])  
print(len(chunks))  

# converting to vector databse

embedding  = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(chunks,embedding=embedding) 
# print(vector_store.index_to_docstore_id)

# retriever
retriever = vector_store.as_retriever(search_type="similarity",search_kwargs={"k":4})

# Augmentation
llm =   ChatOpenAI(model="gpt-4o-mini",temperature=0.2)

prompt =  PromptTemplate(
    template="""
    You are a helpful assistant.
    Answer ONLY from the provided transcript context.
    If the context is insufficient, just say you dont know .
    {context}
    Questions: {question}
    """,
    input_variables=['context','question']
)

question = "what is dog"
retrieved_docs = retriever.invoke(question)
context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
final_prompt = prompt.invoke({"context": context_text, "question":question})

answer = llm.invoke(final_prompt)
print(answer)