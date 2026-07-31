import os
import base64
from typing import TypedDict, List, Optional
from urllib.parse import quote_plus
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

app = FastAPI(title="Pickleball Strategy Agent with Multimodal Vision")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Initialize Gemini with automatic rate-limit retry logic
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.2,
    max_retries=3
)

# ==========================================
# 2. Define Pydantic Output Contracts
# ==========================================
class VideoResource(BaseModel):
    title: str = Field(description="Name of the technique or drill to watch")
    search_query: str = Field(description="Exact Youtube search phrase (e.g. 'pickleball hold and hit drill')")

class ScoutReportSchema(BaseModel):
    summary: str = Field(description="Executive summary of opponent playstyle & video mechanics")
    strengths: List[str] = Field(description="Key strengths observed from notes or video")
    weaknesses: List[str] = Field(description="Key weaknesses or vulnerabilities to exploit")

class TacticalPlanSchema(BaseModel):
    headline: str = Field(description="Overall strategic theme for the match")
    tactics: List[str] = Field(description="3 clear, actionable tactical instructions")
    coach_cue: str = Field(description="Short mental reminder phrase")
    recommended_videos: List[VideoResource] = Field(description="1-2 video topics to study")

class ReviewSchema(BaseModel):
    is_approved: bool = Field(description="True if plan is sound, False if blunders exist")
    feedback: str = Field(description="APPROVED or revision notes")

# Bind structured outputs to model instances
scout_llm = llm.with_structured_output(ScoutReportSchema)
tactician_llm = llm.with_structured_output(TacticalPlanSchema)
critic_llm = llm.with_structured_output(ReviewSchema)

# ==========================================
# 3. LangGraph State Schema
# ==========================================
class AgentState(TypedDict):
    input_notes: str
    video_base64: Optional[str]
    video_mime: Optional[str]
    scout_analysis: ScoutReportSchema
    proposed_tactics: TacticalPlanSchema
    critic_review: ReviewSchema
    revision_count: int

# ==========================================
# 4. Graph Nodes
# ==========================================
def scout_node(state: AgentState):
    """Parses text notes and inline video bytes using Gemini Multimodal Vision."""
    prompt_text = f"Analyze this opponent. Additional text notes provided: {state['input_notes'] or 'None'}"
    
    # If a video was uploaded, pass multimodal block directly in RAM
    if state.get("video_base64"):
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {
                    "type": "media",
                    "mime_type": state["video_mime"],
                    "data": state["video_base64"]
                }
            ]
        )
    else:
        message = HumanMessage(content=prompt_text)
        
    report: ScoutReportSchema = scout_llm.invoke([message])
    return {"scout_analysis": report}

def tactician_node(state: AgentState):
    feedback_context = f"\nPrevious Feedback: {state['critic_review'].feedback}" if state.get('critic_review') else ""
    prompt = f"""
    Create a tactical plan and recommend 1-2 video topics to practice these counters.
    Scout Summary: {state['scout_analysis'].summary}
    Weaknesses: {', '.join(state['scout_analysis'].weaknesses)}
    {feedback_context}
    """
    plan: TacticalPlanSchema = tactician_llm.invoke(prompt)
    return {"proposed_tactics": plan, "revision_count": state["revision_count"] + 1}

def critic_node(state: AgentState):
    prompt = f"""
    Evaluate this plan against the scout report. Check for strategic blunders.
    Scout: {state['scout_analysis'].summary}
    Tactics: {', '.join(state['proposed_tactics'].tactics)}
    """
    review: ReviewSchema = critic_llm.invoke(prompt)
    return {"critic_review": review}

# ==========================================
# 5. Routing Logic & Graph Build
# ==========================================
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

# ==========================================
# 6. Unified API Endpoint (Multipart/Form-Data)
# ==========================================
@app.post("/api/analyze")
async def analyze_match(
    file: Optional[UploadFile] = File(None),
    opponent_notes: str = Form("")
):
    try:
        video_b64 = None
        mime_type = None

        # Read binary video file directly into memory (RAM)
        if file and file.filename:
            file_bytes = await file.read()
            video_b64 = base64.b64encode(file_bytes).decode("utf-8")
            mime_type = file.content_type or "video/mp4"

        initial_state = {
            "input_notes": opponent_notes,
            "video_base64": video_b64,
            "video_mime": mime_type,
            "revision_count": 0
        }
        
        final_state = graph.invoke(initial_state)
        
        plan_dict = final_state["proposed_tactics"].model_dump()
        
        # Format video queries into working YouTube search URLs
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

# Mount frontend files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")