import os
from typing import Literal
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
from state import AgentState

load_dotenv()

app = FastAPI(title="Enterprise Research Agent API")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
search_tool = DuckDuckGoSearchRun()

# --- Agent Nodes ---
def planner_node(state: AgentState) -> dict:
    prompt = f"Break down this task into 3 execution steps: {state['task']}"
    response = llm.invoke(prompt)
    steps = [s.strip() for s in response.content.split("\n") if s.strip()]
    return {"plan": steps, "next_step": "research"}

def researcher_node(state: AgentState) -> dict:
    query = state["task"]
    results = search_tool.run(query)
    return {"research_data": [results], "next_step": "validate"}

def validator_node(state: AgentState) -> dict:
    raw_data = "\n".join(state["research_data"])
    prompt = f"Verify the facts in this text and list verified points:\n{raw_data}"
    response = llm.invoke(prompt)
    return {"validated_facts": [response.content], "next_step": "compile"}

def compiler_node(state: AgentState) -> dict:
    facts = "\n".join(state["validated_facts"])
    prompt = f"Write an executive report based on these facts:\n{facts}"
    response = llm.invoke(prompt)
    return {"final_report": response.content, "next_step": "end"}

# --- Graph Assembly ---
workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("validator", validator_node)
workflow.add_node("compiler", compiler_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "researcher")
workflow.add_edge("researcher", "validator")
workflow.add_edge("validator", "compiler")
workflow.add_edge("compiler", END)

graph = workflow.compile()

# --- API Endpoint ---
class ResearchRequest(BaseModel):
    task: str

@app.post("/research")
async def run_research(request: ResearchRequest):
    try:
        initial_state = {
            "task": request.task,
            "plan": [],
            "research_data": [],
            "validated_facts": [],
            "final_report": "",
            "next_step": "planner"
        }
        result = graph.invoke(initial_state)
        return {"report": result["final_report"], "plan": result["plan"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)