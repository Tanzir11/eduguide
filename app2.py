import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate


load_dotenv()

prompt = """You are an Academic Advisor chatbot named EduGuideGPT. 
Your role is to generate friendly and informative responses to student inquiries related to academic matters. 
The goal is to provide helpful guidance, answer questions about majors, scores (SAT, ACT, GPA), PSAT insights, college admissions, and offer encouragement for the application process. 
Use friendly tone and structure responses in a conversational manner.        
Keep the response short nothing more than 50-80 words.   
Specific Points to Include:    
Opening     
    1. Congratulatory Remark/Greetings.
Subject matter
    2. Major of Interest discussion, emphasizing its relevance to college aspirations.    
    3. Discussion on SAT scores, ACT scores, and GPA.    
    4. Insights on the importance of PSAT and its relation to SAT preparedness.    
    5. Assessment of college admission possibilities based on the suggested university or location, considering the major.    
Closure
6. advice, including holistic admissions, certifications, and addressing any missing exams.    
7. Encouragement for further engagement with the university and the application process.    
8. A final prompt indicating the readiness to apply and summarizing the student's academic standing.        
9. Be precise and stick to the point.    
Context: {context}    
question: {question}
Take the necessassory details from the previous conversation if provided Remember these are previous quaries that the model made: {conv_history}
Answer question based only on question parameters which are given.
"""

encode_kwargs = {'normalize_embeddings': True}

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

instructor_embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-base",
    model_kwargs={"device": "cpu"},
)

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

path = "static/DB"
vectorstore = Chroma(persist_directory=path, embedding_function=instructor_embeddings)
print("loaded vectorstore")

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# CORS (Cross-Origin Resource Sharing) middleware
origins = ["*"]  # You might want to restrict this based on your requirements
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def store_string_to_file(file_path, content):
    try:
        with open(file_path, 'a', encoding='utf-8') as file:
            file.write(content + "\n")
        print(f"String successfully stored in {file_path}")
    except Exception as e:
        print(f"Error storing string in {file_path}: {str(e)}")

def extract_content_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        if not content:
            print(f"The file at {file_path} is empty.")
        
        print(f"Content successfully extracted from {file_path}")
        return content if content else ""
    except Exception as e:
        return ""

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/query", response_class=JSONResponse)
async def process_query(data: dict):
    try:
        if 'query' not in data:
            raise HTTPException(status_code=400, detail="Bad Request: 'query' parameter is missing in the request data")

        query = data['query']
        store_string_to_file("conversation.txt", query)
        conv_history = extract_content_from_file("conversation.txt")

        Conv_PROMPT = PromptTemplate(template=prompt, input_variables=["context", "question", "conv_history"])
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        print("loaded retriever")

        qa = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo", max_tokens=300),
            retriever=retriever,
            return_source_documents=False,
            max_tokens_limit=8100,
            combine_docs_chain_kwargs={'prompt': Conv_PROMPT}
        )

        # Perform retrieval using the combined chat history
        answer = qa({"question": query, "conv_history":conv_history, "chat_history":[]})

        return {"response": answer['answer']}

    except Exception as e:
        # Handle exceptions here
        return {"error": str(e)}
