import os
from typing import TypedDict, List
from urllib.parse import quote_plus
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

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.2
)

# ==========================================
# 1. New Resource Schemas
# ==========================================
class VideoResource(BaseModel):
    title: str = Field(description="Name of the technique or drill to watch")
    search_query: str = Field(description="Exact Youtube search phrase (e.g. 'pickleball hold and hit drill')")

class ScoutReportSchema(BaseModel):
    summary: str = Field(description="Executive summary of opponent playstyle")
    strengths: List[str] = Field(description="Key strengths observed")
    weaknesses: List[str] = Field(description="Key weaknesses or vulnerabilities")

class TacticalPlanSchema(BaseModel):
    headline: str = Field(description="Overall strategic theme for the match")
    tactics: List[str] = Field(description="3 clear, actionable tactical instructions")
    coach_cue: str = Field(description="Short mental reminder phrase")
    recommended_videos: List[VideoResource] = Field(description="1-2 video topics to study for these tactics")

class ReviewSchema(BaseModel):
    is_approved: bool = Field(description="True if plan is sound, False if blunders exist")
    feedback: str = Field(description="APPROVED or revision notes")

# Bind models
scout_llm = llm.with_structured_output(ScoutReportSchema)
tactician_llm = llm.with_structured_output(TacticalPlanSchema)
critic_llm = llm.with_structured_output(ReviewSchema)

class AgentState(TypedDict):
    opponent_notes: str
    scout_analysis: ScoutReportSchema
    proposed_tactics: TacticalPlanSchema
    critic_review: ReviewSchema
    revision_count: int

def scout_node(state: AgentState):
    prompt = f"Analyze these opponent notes: {state['opponent_notes']}"
    report: ScoutReportSchema = scout_llm.invoke(prompt)
    return {"scout_analysis": report}

def tactician_node(state: AgentState):
    feedback_context = f"\nPrevious Feedback: {state['critic_review'].feedback}" if state.get('critic_review') else ""
    prompt = f"""
    Create a tactical plan and recommend 1-2 video topics to help practice these counters.
    Scout Summary: {state['scout_analysis'].summary}
    Weaknesses: {', '.join(state['scout_analysis'].weaknesses)}
    {feedback_context}
    """
    plan: TacticalPlanSchema = tactician_llm.invoke(prompt)
    return {"proposed_tactics": plan, "revision_count": state["revision_count"] + 1}

def critic_node(state: AgentState):
    prompt = f"""
    Evaluate this plan against the scout report. Check for blunders.
    Scout: {state['scout_analysis'].summary}
    Tactics: {', '.join(state['proposed_tactics'].tactics)}
    """
    review: ReviewSchema = critic_llm.invoke(prompt)
    return {"critic_review": review}

def should_continue(state: AgentState) -> str:
    review = state.get("critic_review")
    if (review and review.is_approved) or state["revision_count"] >= 3:
        return END
    return "tactician"

builder = StateGraph(AgentState)
builder.add_node("scout", scout_node)
builder.add_node("tactician", tactician_node)
builder.add_node("critic", critic_node)
builder.set_entry_point("scout")
builder.add_edge("scout", "tactician")
builder.add_edge("tactician", "critic")
builder.add_conditional_edges("critic", should_continue, {"tactician": "tactician", END: END})

graph = builder.compile()

class MatchRequest(BaseModel):
    opponent_notes: str

@app.post("/api/analyze")
async def analyze_match(request: MatchRequest):
    try:
        initial_state = {"opponent_notes": request.opponent_notes, "revision_count": 0}
        final_state = graph.invoke(initial_state)
        
        plan_dict = final_state["proposed_tactics"].model_dump()
        
        # Helper: Transform queries into dynamic, working YouTube URLs
        for video in plan_dict.get("recommended_videos", []):
            encoded_query = quote_plus(video["search_query"])
            video["url"] = f"https://www.youtube.com/results?search_query={encoded_query}"

        return {
            "scout_analysis": final_state["scout_analysis"].model_dump(),
            "tactical_plan": plan_dict,
            "iterations": final_state["revision_count"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")