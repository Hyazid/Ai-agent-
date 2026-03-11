from langchain_ollama import ChatOllama
from langgraph.prebuilt import  create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
#from langchain.prompts import PromptTemplate

import json 
#=============================================================
#1 tools . uses the @tool decorator as tool description and metadata
#=============================================================
@tool
def calculator(expression:str)->str:
    '''evalute mathematical expression and return a result'''
    try:
        allowed=set('0123456789/*-(),.+')
        if not all (c in allowed for c in expression):
            return "Error invalid character"
        return str(eval(expression))
    except Exception as e:
        return f"Error evaluating expression: {e}"
    
@tool
def word_counter(expression:str)->str:
    "count the number of word"
    count = len(expression.split())
    return f"'{expression}' contain '{count}' word"

#####################################
#==2.connect to ollama
llm= ChatOllama(
    model="llama3.2",
    temperature=0.7,
    num_predict=512
)



#=============================================================
#4. create agent
#=============================================================
memory = MemorySaver()
tools=[calculator, word_counter]
agent=create_react_agent(model=llm, tools=tools,checkpointer=memory )


#=============================================================
#5. run agent
def ask(question: str, thread_id:str = "default"):
    config= {"conversation_id": thread_id}
    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print(f"{'='*60}")

    result = agent.invoke({
        "messages": [HumanMessage(content=question)]
    })

    # The last message in the list is the final answer
    final = result["messages"][-1].content
    print(f"\n✅ FINAL ANSWER: {final}")

    # Optional: print full trace
    print("\n--- Full message trace ---")
    for msg in result["messages"]:
        print(f"[{msg.__class__.__name__}]: {msg.content[:200]}")

    return final

if __name__=="__main__":
    question = input("Ask a question: ")
    ask(question)