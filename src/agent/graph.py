from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.agent.nodes import extract_memory, perceive, reason_act, respond, retrieve_memory
from src.agent.state import AgentState
from src.tools.tools import AGENT_TOOLS


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("perceive", perceive)
    workflow.add_node("retrieve_memory", retrieve_memory)
    workflow.add_node("reason_act", reason_act)
    workflow.add_node("tools", ToolNode(AGENT_TOOLS))
    workflow.add_node("respond", respond)
    workflow.add_node("extract_memory", extract_memory)

    workflow.set_entry_point("perceive")
    workflow.add_edge("perceive", "retrieve_memory")
    workflow.add_edge("retrieve_memory", "reason_act")
    workflow.add_conditional_edges("reason_act", tools_condition, {"tools": "tools", END: "respond"})
    workflow.add_edge("tools", "reason_act")
    workflow.add_edge("respond", "extract_memory")
    workflow.add_edge("extract_memory", END)

    return workflow.compile()
