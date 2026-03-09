import ollama
import requests
import json

llm =ollama.chat(model='llama3.2',
                 messages=[{"role": "user", "content": "What is the capital of France?"}])

print(llm['message']['content'])
#======+++============================
#tools- function that Ai can use
#======+++============================
def calculator(expression:str)-> str:
#     """A simple calculator function that evaluates a mathematical expression.""
    try:
        allowed=  set("0123456789+*-/().,")
        if not all(c in allowed for c in expression):
            return "Invalid characters in expression."
        return  str(eval(expression))
    except Exception as e:
        return f"Error evaluating expression: {e}"
    
#======+++============================
def string_reverse(s:str)-> str:
    """A function that reverses a given string."""
    return s[::-1]

#======+++============================
#map tool name to function
TOOLS={
    "calculator": calculator,
    "string_reverse": string_reverse
}
# ============================================================
# SYSTEM PROMPT — teaches the LLM how to behave as an agent
# ============================================================
SYSTEM_PROMPT = """"you are an AI agent. you can use tools to answer questions. you have access to the following tools:

AVAILABLE TOOLS:
- calculator(expression): Evaluates math. e.g. {"tool": "calculator", "args": {"expression": "15 * 7"}}

- string_reverse(text): Reverses a string. e.g. {"tool": "string_reverse", "args": {"text": "hello"}}

RULES — you MUST follow these strictly:
1. Always respond with a single valid JSON object. Nothing else.
2. To use a tool: {"tool": "tool_name", "args": {"arg_name": "value"}}
3. When you have the final answer: {"answer": "your answer here"}
4. Never add explanation outside the JSON.
5. Never call a tool you already called with the same arguments.

"""
# ============================================================
# AGENT LOOP
# ============================================================
def run_agent(user_input:str, max_steps:int =6):
    print(f"\n={'='*50}")
    print(f"USER: {user_input}")
    print(f"{'='*50}\n")
    messages = [
        {"role":"system", "content": SYSTEM_PROMPT},
        {"role":"user", "content": user_input}
    ]
    for step in range(max_steps):
        print(f"\n--- Step {step + 1} ---")
        response = ollama.chat(model='llama3.2', messages=messages)
        raw = response['message']['content'].strip()
        print(f"LLM RESPONSE: {raw}")
        if raw.startswith("```"):
            raw = raw.strip("`").strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            print("Invalid JSON. Ending.")
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": "Your response was not valid JSON. Respond ONLY with a JSON object."})
            continue
        if "answer" in parsed:
            print(f"FINAL ANSWER: {parsed['answer']}")
            return parsed['answer']
        
        if "tool" in parsed:
            tool_name =parsed.get("tool")
            args = parsed.get("args", {})
            if tool_name not in TOOLS:
                print(f"Unknown tool: {tool_name}. Ending.")
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": f"Unknown tool: {tool_name}. Respond ONLY with a valid tool."})
                continue

            print(f"Calling tool: {tool_name} with args: {args}")
            result = TOOLS[tool_name](**args)
            print(f"Tool result: {result}")
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"Tool result: {result}"})
        else:
            print("No tool called. Ending.")
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": "You must call a tool or provide a final answer. Respond ONLY with a JSON object."})
            continue
    
    print("\n⛔ Max steps reached without a final answer.")
    return None
# ============================================================
# TEST RUNS
# ============================================================
if __name__ == "__main__":
    # Test 1: Pure math
    question  = input("ask ....")
    run_agent(question)
