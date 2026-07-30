import os
from typing import TypedDict, List
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

app = FastAPI(title="Pickleball Strategy Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.2
)

# ==========================================
# 2. Define Pydantic Output Contracts
# ==========================================
class ScoutReportSchema(BaseModel):
    summary: str = Field(description="Executive summary of the opponent's playstyle")
    strengths: List[str] = Field(description="Key strengths observed")
    weaknesses: List[str] = Field(description="Key weaknesses or vulnerabilities to exploit")
    preferred_shots: List[str] = Field(description="Their go-to shots or patterns")

class TacticalPlanSchema(BaseModel):
    headline: str = Field(description="Catchy overall strategic theme for the match")
    tactics: List[str] = Field(description="3 clear, actionable tactical instructions")
    coach_cue: str = Field(description="A short mental reminder phrase for between points")

class ReviewSchema(BaseModel):
    is_approved: bool = Field(description="True if plan is tactically sound, False if blunders exist")
    feedback: str = Field(description="APPROVED if true, or bullet points on what needs fixing")

# Bind models to LLM for guaranteed output structure
scout_llm = llm.with_structured_output(ScoutReportSchema)
tactician_llm = llm.with_structured_output(TacticalPlanSchema)
critic_llm = llm.with_structured_output(ReviewSchema)

# ==========================================
# 3. LangGraph State Schema
# ==========================================
class AgentState(TypedDict):
    opponent_notes: str
    scout_analysis: ScoutReportSchema
    proposed_tactics: TacticalPlanSchema
    critic_review: ReviewSchema
    revision_count: int

# ==========================================
# 4. Graph Nodes
# ==========================================
def scout_node(state: AgentState):
    prompt = f"""
    You are an elite pickleball scout. Analyze these opponent notes and extract structured observations:
    
    Notes: {state['opponent_notes']}
    """
    report: ScoutReportSchema = scout_llm.invoke(prompt)
    return {"scout_analysis": report}

def tactician_node(state: AgentState):
    feedback_context = f"\nPrevious Reviewer Feedback: {state['critic_review'].feedback}" if state.get('critic_review') else ""
    
    prompt = f"""
    You are a high-level pickleball tactical coach. Based on this scout report, create a structured tactical plan.
    
    Scout Summary: {state['scout_analysis'].summary}
    Weaknesses to exploit: {', '.join(state['scout_analysis'].weaknesses)}
    {feedback_context}
    """
    plan: TacticalPlanSchema = tactician_llm.invoke(prompt)
    return {
        "proposed_tactics": plan,
        "revision_count": state["revision_count"] + 1
    }

def critic_node(state: AgentState):
    prompt = f"""
    You are a pickleball strategic reviewer. Evaluate this plan against the scout report.
    Check for high-risk blunders (e.g. leaving the middle wide open, misreading 3rd shot drives vs drops).
    
    Scout Report: {state['scout_analysis'].summary}
    Proposed Tactics: {', '.join(state['proposed_tactics'].tactics)}
    
    If the plan is solid, set is_approved to True and feedback to 'APPROVED'.
    Otherwise set is_approved to False and explain what needs fixing in feedback.
    """
    review: ReviewSchema = critic_llm.invoke(prompt)
    return {"critic_review": review}

# ==========================================
# 5. Routing Logic
# ==========================================
def should_continue(state: AgentState) -> str:
    review = state.get("critic_review")
    if (review and review.is_approved) or state["revision_count"] >= 3:
        return END
    return "tactician"

# ==========================================
# 6. Build Graph
# ==========================================
builder = StateGraph(AgentState)
builder.add_node("scout", scout_node)
builder.add_node("tactician", tactician_node)
builder.add_node("critic", critic_node)

builder.set_entry_point("scout")
builder.add_edge("scout", "tactician")
builder.add_edge("tactician", "critic")
builder.add_conditional_edges("critic", should_continue, {"tactician": "tactician", END: END})

graph = builder.compile()

# ==========================================
# 7. API Endpoint & Static Mount
# ==========================================
class MatchRequest(BaseModel):
    opponent_notes: str

@app.post("/api/analyze")
async def analyze_match(request: MatchRequest):
    try:
        initial_state = {
            "opponent_notes": request.opponent_notes,
            "revision_count": 0
        }
        final_state = graph.invoke(initial_state)
        
        # Pydantic objects convert cleanly to standard dicts for API JSON responses
        return {
            "scout_analysis": final_state["scout_analysis"].model_dump(),
            "tactical_plan": final_state["proposed_tactics"].model_dump(),
            "iterations": final_state["revision_count"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")