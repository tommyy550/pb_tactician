import os
from typing import TypedDict
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

app = FastAPI(title="Pickleball Strategy Agent")

# Allow local testing / external calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Initialize Gemini Model (uses GOOGLE_API_KEY environment variable)
# Free tier model: gemini-2.5-flash (or gemini-1.5-flash)
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.3
)

# 2. State Schema
class AgentState(TypedDict):
    opponent_notes: str
    scout_analysis: str
    proposed_tactics: str
    critic_feedback: str
    revision_count: int

# 3. Graph Nodes
def scout_node(state: AgentState):
    prompt = f"""
    You are an elite pickleball scout. Analyze these raw opponent notes and extract 
    their key strengths, weaknesses, preferred shots, and tendencies:
    
    Notes: {state['opponent_notes']}
    """
    response = llm.invoke(prompt)
    return {"scout_analysis": response.content}

def tactician_node(state: AgentState):
    feedback_context = f"\nPrevious Critic Feedback: {state['critic_feedback']}" if state['critic_feedback'] else ""
    prompt = f"""
    You are a high-level pickleball tactical coach. Based on this scout report, create a 
    3-point tactical game plan to defeat this opponent.
    
    Scout Report:
    {state['scout_analysis']}
    {feedback_context}
    """
    response = llm.invoke(prompt)
    return {
        "proposed_tactics": response.content,
        "revision_count": state["revision_count"] + 1
    }

def critic_node(state: AgentState):
    prompt = f"""
    You are a high-level pickleball reviewer. Evaluate this tactical plan against the scout report.
    Check for high-risk blunders or tactical holes.
    
    Scout Report: {state['scout_analysis']}
    Proposed Plan: {state['proposed_tactics']}
    
    If the plan is tactically solid, reply with ONLY the word: APPROVED.
    Otherwise, provide 1-2 concise bullet points on what needs to be fixed.
    """
    response = llm.invoke(prompt)
    
    # Extract raw string if response content is returned as a list
    content = response.content
    if isinstance(content, list):
        content = " ".join(
            item if isinstance(item, str) else item.get("text", "") 
            for item in content
        )

    return {"critic_feedback": str(content)}

# 4. Routing Logic
def should_continue(state: AgentState) -> str:
    feedback = state.get("critic_feedback", "")
    
    # Flatten if it's somehow still a list
    if isinstance(feedback, list):
        feedback = " ".join(str(i) for i in feedback)
        
    if "APPROVED" in str(feedback).upper() or state.get("revision_count", 0) >= 3:
        return END
    return "tactician"

# 5. Build Graph
builder = StateGraph(AgentState)
builder.add_node("scout", scout_node)
builder.add_node("tactician", tactician_node)
builder.add_node("critic", critic_node)

builder.set_entry_point("scout")
builder.add_edge("scout", "tactician")
builder.add_edge("tactician", "critic")
builder.add_conditional_edges("critic", should_continue, {"tactician": "tactician", END: END})

graph = builder.compile()

# 6. API Endpoint
class MatchRequest(BaseModel):
    opponent_notes: str

@app.post("/api/analyze")
async def analyze_match(request: MatchRequest):
    try:
        initial_state: AgentState = {
            "opponent_notes": request.opponent_notes,
            "scout_analysis": "",
            "proposed_tactics": "",
            "critic_feedback": "",
            "revision_count": 0
        }
        final_state = graph.invoke(initial_state)
        return {
            "scout_analysis": final_state["scout_analysis"],
            "tactical_plan": final_state["proposed_tactics"],
            "iterations": final_state["revision_count"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 7. Serve Static Frontend Files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")