from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
import datetime


#=============================================================
#1 real tool tools . uses the @tool decorator as tool description and metadata
#=============================================================

search = DuckDuckGoSearchRun()
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())


#=============================================================
#custom tools
#=============================================================
@tool
def get_date_time()->str:
    "get current date and time"
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")    

@tool
def save_note(note:str)->str:
    """saves a notes to local file called notes.txt"""
    with open('notes.txt','a') as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp}: {note}\n")
    return "Note saved successfully."

@tool
def read_notes()->str:
    """ reads the notes from notes.txt"""
    try:
        with open('notes.txt','r') as f:
            notes = f.read()
        return notes if notes else 'no notes saved '
    
    except FileNotFoundError:
        return "No notes file found.    "
    
#=============================================================  
#2connect to llm +memory
#=============================================================
llm  = ChatOllama(model="llama3.2", temperature=0)
memory = MemorySaver()
tools=[search,wiki,get_date_time,save_note,read_notes]

agent = create_react_agent(model=llm, tools=tools,checkpointer=memory)
#=============================================================
#3 ask agent
#=============================================================
def ask (question:str, thread_id:str = "default"):
    config ={"configurable":{"thread_id":thread_id}}
    result = agent.invoke({
        "messages": [HumanMessage(content=question)]
    }, config=config)
    final = result["messages"][-1].content
    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print(f"ANSWER: {final}")
    print(f"{'='*60}\n")
    return final


if __name__ == "__main__":
    print("\n" + "="*60)
    print("TEST 2: Wikipedia")
    print("="*60)
    ask("fill the notes.txt with hello fouzi", thread_id="t2")
