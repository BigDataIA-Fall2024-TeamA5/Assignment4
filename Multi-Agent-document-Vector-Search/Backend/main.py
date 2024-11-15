from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from web_search_agent import WebSearchAgent
from arxiv_agent import ArxivAgent
from rag_agent import RAGAgent
from langgraph import Langgraph
import boto3
import os
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

app = FastAPI()




# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Langraph instance
langraph = Langraph()

# Initialize agents
web_search_agent = WebSearchAgent(langraph=langraph)
arxiv_agent = ArxivAgent(langraph=langraph)
rag_agent = RAGAgent(langraph=langraph)

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI JWT Authentication Application!"}

# WebSearch Agent Endpoint
@app.post("/websearch")
def websearch_agent_endpoint(query: str, token: str = Depends(oauth2_scheme)):
    try:
        result = web_search_agent.process_query(query)
        return {"agent": "websearch", "query": query, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Arxiv Agent Endpoint
@app.post("/arxiv")
def arxiv_agent_endpoint(query: str, token: str = Depends(oauth2_scheme)):
    try:
        result = arxiv_agent.process_query(query)
        return {"agent": "arxiv", "query": query, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# RAG Agent Endpoint
@app.post("/rag")
def rag_agent_endpoint(query: str, pinecone_vectors: dict, token: str = Depends(oauth2_scheme)):
    try:
        result = rag_agent.process_query(query, pinecone_vectors)
        return {"agent": "rag", "query": query, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

