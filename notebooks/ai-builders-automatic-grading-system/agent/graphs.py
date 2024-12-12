import uuid
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .nodes import (
    basic_rag_call_evaluator,
    one_shot_call_evaluator,
    call_reflection,
    should_continue_eval,
    should_continue_reflection,
    format_cks,
    tool_node_eval,
    supervisor,
    evaluator_with_feedback,
    format_cks_reflection
    )

from .state import OverallState

def basic_rag_graph():
    memory = MemorySaver()
    workflow = StateGraph(OverallState)
    workflow.add_node("call_evaluator", basic_rag_call_evaluator)
    workflow.add_node("eval_tools", tool_node_eval)
    workflow.add_node("format_cks", format_cks)

    workflow.add_edge(START, "call_evaluator")
    workflow.add_conditional_edges("call_evaluator", should_continue_eval, ["eval_tools", "format_cks"])
    workflow.add_edge("eval_tools", "call_evaluator")
    workflow.add_edge("format_cks", END)
    return workflow.compile(checkpointer=memory)

def one_shot_graph():
    memory = MemorySaver()
    workflow = StateGraph(OverallState)
    workflow.add_node("call_evaluator", one_shot_call_evaluator)
    workflow.add_node("format_cks", format_cks)

    workflow.add_edge(START, "call_evaluator")
    workflow.add_edge("call_evaluator", "format_cks")
    workflow.add_edge("format_cks", END)

    return workflow.compile(checkpointer=memory)

def one_shot_with_reflection_graph():
    memory = MemorySaver()
    workflow = StateGraph(OverallState)
    workflow.add_node("call_evaluator", one_shot_call_evaluator)
    workflow.add_node("format_cks", format_cks_reflection)
    workflow.add_node("call_reflection", call_reflection)
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("evaluator_with_feedback", evaluator_with_feedback)

    workflow.add_edge(START, "call_evaluator")
    workflow.add_edge("call_evaluator", "supervisor")
    workflow.add_conditional_edges("supervisor", should_continue_reflection, ["evaluator_with_feedback","call_reflection","format_cks"])
    workflow.add_edge("evaluator_with_feedback", "supervisor")
    workflow.add_edge("call_reflection", "supervisor")
    workflow.add_edge("format_cks", END)
    return workflow.compile(checkpointer=memory)

def basic_rag_with_reflection_graph():
    return NotImplementedError

def execute_graph(exec_state, graph):
    i = 0
    unique_id = uuid.uuid4()

    config = {
            "configurable": { "thread_id": unique_id, "run_name": "AIBuildersDemo"}, 
            "recursion_limit": 10
            }

    for chunk in graph.stream(exec_state , config, stream_mode="values"):
        try:
            chunk["messages"][-1].pretty_print()
        except:
            pass
        i += 1
        if i == 10:
            break

    final_state = graph.get_state(config).values
    knowledge_state = final_state['current_knowledge_state']
    
    return knowledge_state