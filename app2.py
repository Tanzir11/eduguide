import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
import pickle
import json
from pydantic import BaseModel
from typing import List

load_dotenv()

app = FastAPI()

class QnaRequest(BaseModel):
    questions: str
    chat_history: List[str]

prompts = """You are an Academic Advisor chatbot named EduGuideGPT. 
Your role is to generate friendly and informative responses to student inquiries related to academic matters. 
The goal is to provide helpful guidance, answer questions about majors, scores (SAT, ACT, GPA), PSAT insights, college admissions, and offer encouragement for the application process. 
Use friendly tone and structure responses in a conversational manner.
Specific Points to Include:    
Opening     
    1. Congratulatory Remark/Greetings.
Subject matter
    2. Major of Interest discussion, emphasizing its relevance to college aspirations.    
    3. Discussion on SAT scores, ACT scores, and GPA.    
    4. Insights on the importance of PSAT and its relation to SAT preparedness.    
    5. Assessment of college admission possibilities based on the suggested university or location, considering the major.
    6. Provide all related links  
Closure
6. advice, including holistic admissions, certifications, and addressing any missing exams.    
7. Encouragement for further engagement with the university and the application process.    
Context: {context}    
question: {question}
Answer question based only on question parameters which are given.
Make sure to give relevant new lines
"""

embeddings = HuggingFaceInstructEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"})

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

origins = ["*"]  # You might want to restrict this based on your requirements
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_embeddings(sotre_name, path):
    with open(f"{path}/faiss_{sotre_name}.pkl", "rb") as f:
        VectorStore = pickle.load(f)
    return VectorStore

templates = Jinja2Templates(directory="templates")

def extract_data(input_data):
    user_queries = []
    for i in range(len(input_data)):
        variable_name = f"user_query_variable{i + 1}"
        globals()[variable_name] = input_data[i]
        user_queries.append((variable_name, globals()[variable_name]))

    return user_queries

def Qna(questions, user_queries):
    db_instructEmbedd = load_embeddings(sotre_name='instructEmbeddings', path="Embedding_store")
    retriever = db_instructEmbedd.as_retriever(search_kwargs={"k": 3})
    temp1 = prompts + " " + "the previous user queries are as follows" + " " + user_queries
    prompt = ChatPromptTemplate.from_template(temp1)
    model = ChatOpenAI(model="gpt-4", temperature=0)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
        )
    
    data = chain.invoke(questions)
    return data 

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/query", response_class=JSONResponse)
async def process_query(request: QnaRequest):
    try:
        query = request.questions
        chat_history = request.chat_history
        user_queries = extract_data(chat_history)
        str_user = str(user_queries)
        result = Qna(user_queries=str_user, questions=query)
        resp = json.dumps(result)
        response = json.loads(resp)
        return {"response": response}
    
    except Exception as e:
        # Handle exceptions here
        return {"error": str(e)}
