from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from typing import TypedDict, Literal
import datetime

# ============================================================
# 1. LLM
# ============================================================

llm = ChatOllama(model="llama3.2", temperature=0)

# ============================================================
# 2. TOOLS
# ============================================================

search = DuckDuckGoSearchRun()
wiki   = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

@tool
def get_date_time() -> str:
    """Get current date and time."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def save_note(note: str) -> str:
    """Saves a note to local file notes.txt."""
    with open("notes.txt", "a") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp}: {note}\n")
    return "Note saved successfully."

@tool
def read_notes() -> str:
    """Reads all notes from notes.txt."""
    try:
        with open("notes.txt", "r") as f:
            notes = f.read()
        return notes if notes else "No notes saved."
    except FileNotFoundError:
        return "No notes file found."

# ============================================================
# 3. SUB-AGENTS
#    FIX: state_modifier → prompt
# ============================================================

research_agent = create_react_agent(
    model=llm,
    tools=[search, wiki],
    prompt=SystemMessage(content=(
        "You are a research specialist. "
        "Use the search and Wikipedia tools to find accurate information. "
        "Always mention your source. Be concise."
    ))
)

writer_agent = create_react_agent(
    model=llm,
    tools=[get_date_time, save_note, read_notes],
    prompt=SystemMessage(content=(
        "You are a personal assistant specialized in note-taking and file organization. "
        "Use get_date_time to timestamp, save_note to save, read_notes to recall. "
        "Be concise and organized."
    ))
)

# ============================================================
# 4. GRAPH STATE
# ============================================================

class AgentState(TypedDict):
    input:           str
    research_output: str
    notes:           str
    final_output:    str
    route:           str

# ============================================================
# 5. ORCHESTRATOR
#    FIX: f-string was broken — {question} wasn't interpolated
# ============================================================

def orchestrator(state: AgentState) -> AgentState:
    question = state["input"]
    print(f"\n[ORCHESTRATOR] Analyzing: '{question}'")

    routing_prompt = f"""
You are a task router. Given this question, decide which agents are needed.
Respond with ONLY one of these words (no explanation):
- research  → needs web search or factual lookup
- writer    → needs note taking or file management
- both      → needs research AND writing

Question: {question}
Answer:"""

    response = llm.invoke([HumanMessage(content=routing_prompt)])
    route = response.content.strip().lower()

    if "both" in route:
        route = "both"
    elif "writer" in route:
        route = "writer"
    else:
        route = "research"

    print(f"[ORCHESTRATOR] Routing to: {route.upper()}")
    return {**state, "route": route}

# ============================================================
# 6. AGENT NODES
#    FIX: agent.invoke() takes a dict with "messages" key, not a list
# ============================================================

def run_research(state: AgentState) -> AgentState:
    print("\n[RESEARCH AGENT] Working...")
    result = research_agent.invoke({
        "messages": [HumanMessage(content=state["input"])]
    })
    output = result["messages"][-1].content
    print(f"[RESEARCH AGENT] Done: {output[:120]}...")
    return {**state, "research_output": output}

def run_writer(state: AgentState) -> AgentState:
    print("\n[WRITER AGENT] Working...")
    context = ""
    if state.get("research_output"):
        context += f"Research findings:\n{state['research_output']}\n\n"

    writer_input = (
        f"Original question: {state['input']}\n\n"
        f"{context}"
        f"Based on the above, take relevant notes and organize them for the user."
    )
    result = writer_agent.invoke({
        "messages": [HumanMessage(content=writer_input)]
    })
    notes = result["messages"][-1].content
    print(f"[WRITER AGENT] Done.")
    return {**state, "notes": notes}

# ============================================================
# 7. ROUTING FUNCTION
#    FIX: removed broken add_edge with dict (not valid),
#         removed duplicate node registration
# ============================================================

def route_decision(state: AgentState) -> Literal["research", "writer", "both_research"]:
    route = state.get("route", "research")
    if route == "both":
        return "both_research"
    return route

# ============================================================
# 8. BUILD GRAPH
# ============================================================

graph = StateGraph(AgentState)

graph.add_node("orchestrator",  orchestrator)
graph.add_node("research",      run_research)
graph.add_node("both_research", run_research)   # same function, separate node
graph.add_node("writer",        run_writer)

graph.set_entry_point("orchestrator")

graph.add_conditional_edges("orchestrator", route_decision, {
    "research":      "research",
    "writer":        "writer",
    "both_research": "both_research",
})

graph.add_edge("research",      "writer")
graph.add_edge("both_research", "writer")
graph.add_edge("writer",        END)

# ============================================================
# 9. COMPILE + RUN
# ============================================================

agent = graph.compile()

def ask(question: str):
    result = agent.invoke({
        "input":           question,
        "research_output": "",
        "notes":           "",
        "final_output":    "",
        "route":           "",
    })
    print(f"\n{'='*60}")
    print(f"QUESTION:         {question}")
    print(f"RESEARCH OUTPUT:  {result.get('research_output', '')[:300]}")
    print(f"NOTES:            {result.get('notes', '')[:300]}")
    print(f"{'='*60}\n")
    return result

if __name__ == "__main__":
    question = input("Ask a question: ")
    ask(question)