from typing import Annotated, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

############## Components ##############
llm = ChatGoogleGenerativeAI(
    model="gemini-1.0-pro",
)

class State(TypedDict):
    messages: Annotated[list, add_messages]
    number_of_words: int

def call_llm(state: State) -> State:
    response = llm.invoke(state['messages'])
    state['messages'].append(response)
    return state

def count_words(state: State) -> State:
    state["number_of_words"] = len(state['messages'][-1].content.split())
    return state

def upper_everything(state: State):
    state['messages'] = [message.content.upper() for message in state['messages']]
    return state

def lower_everything(state: State):
    state['messages'] = [message.content.lower() for message in state['messages']]
    return state

def router(state: State):
    return "more_then_ten" if state['number_of_words'] >= 10 else "less_then_ten"

############## Graph Definition ##############
graph = StateGraph(State)

graph.add_node("Agent", call_llm)
graph.add_node("Count Words", count_words)
graph.add_node("Upper Everything", upper_everything)
graph.add_node("Lower Everything", lower_everything)

graph.set_entry_point("Agent")

graph.add_edge("Agent", "Count Words")
graph.add_edge("Upper Everything", "__end__")
graph.add_edge("Lower Everything", "__end__")

graph.add_conditional_edges(
    "Count Words",
    router,
    {
        "less_then_ten": "Upper Everything",
        "more_then_ten": "Lower Everything"
    }
    )

compiled = graph.compile()

############## Execution ##############
response = compiled.invoke({"messages": "What is AI? Explain in 8 words to 12 words"})

print(response.keys())
print(response["messages"][-1].content)
print(response["number_of_words"])


compiled.get_graph().draw_mermaid_png(output_file_path="output_graph.png")
