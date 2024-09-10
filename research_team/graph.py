from typing import Annotated, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages


from agents.researcher.nodes import build_researcher
from agents.reviewer.nodes import build_reviewer
from agents.summarizer.nodes import build_summarizer


############## Components ##############
llm = ChatGoogleGenerativeAI(
    model="gemini-1.0-pro",
)

researcher = build_researcher()
reviewer = build_reviewer()
summarizer = build_summarizer()
