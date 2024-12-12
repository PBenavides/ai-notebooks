import os
import json
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from typing import List, Literal
from langgraph.prebuilt import ToolNode
from agent.tools import query_lesson
from agent.prompts import (
    EVALUATOR_SYSTEM_PROMPT_BASIC_RAG,
    EVALUATOR_SYSTEM_PROMPT_ONE_SHOT,
    EVALUATOR_USER_PROMPT,
    SINGLE_QUESTIONS_FORMATTER_SYSTEM_PROMPT,
    EVALUATOR_REFLECTION_PROMPT,
    SUPERVISOR_PROMPT,
    EVALUATOR_SYSTEM_PROMPT_ONE_SHOT_FEEDBACK
)
from agent.tools import query_lesson

tool_node_eval = ToolNode([query_lesson])

def basic_rag_call_evaluator(state):
    """Use RAG to review if the content is correct
    """
    model = ChatOpenAI(model="gpt-4o-mini", api_key=os.environ.get('OPENAI_API_KEY'))
    evaluator_tools = [query_lesson]
    messages = [
        SystemMessage(content=EVALUATOR_SYSTEM_PROMPT_BASIC_RAG.format(
            blooms_state=state['blooms_state'],
            concepts_to_evaluate=', '.join(state['concepts_to_evaluate'])
            )),
        HumanMessage(content=EVALUATOR_USER_PROMPT.format(
            open_questions=json.dumps(state['user_input']['open_questions'])
            ))
    ]
    model = model.bind_tools(evaluator_tools)
    llm_response = model.invoke(messages + state['messages'])
    return {'messages': llm_response, 'first_submission': llm_response.content}

def one_shot_call_evaluator(state):
    model = ChatOpenAI(model="gpt-4o", api_key=os.environ.get('OPENAI_API_KEY'))

    messages = [
        SystemMessage(content=EVALUATOR_SYSTEM_PROMPT_ONE_SHOT.format(
            blooms_state=state['blooms_state'],
            concepts_to_evaluate=', '.join(state['concepts_to_evaluate'])
            )),
        HumanMessage(content=EVALUATOR_USER_PROMPT.format(
            open_questions=json.dumps(state['user_input']['open_questions'])
            ))
    ]

    llm_response = model.invoke(messages + state['messages'])
    return {'messages': llm_response, 'first_submission': llm_response.content}

def should_continue_eval(state):
    last_message = state['messages'][-1]
    #Parse if contains END as Next Action
    if last_message.tool_calls:
        return "eval_tools"
    else:
        return "format_cks"

def format_cks(state):
    #Based on the last messages let's give a format to the output.
    class SingleQuestionFormat(BaseModel):
        question: str
        score: str
        reason: str
        cited_paragraph: str
    class SingleQuestionDeck(BaseModel):
        evaluated_questions: List[SingleQuestionFormat]

    last_message = state['messages'][-1]
    messages = [SystemMessage(content=SINGLE_QUESTIONS_FORMATTER_SYSTEM_PROMPT),
                HumanMessage(content=f"""Message to format: {last_message}""")]

    gpt_mini_model = ChatOpenAI(model="gpt-4o-mini", api_key=os.environ.get('OPENAI_API_KEY'))
    structured_model = gpt_mini_model.with_structured_output(SingleQuestionDeck)
    response = structured_model.invoke(messages)

    return {'current_knowledge_state': response.dict()}

def format_cks_reflection(state):
    class SingleQuestionFormat(BaseModel):
        question: str
        score: str
        reason: str
        cited_paragraph: str
    class SingleQuestionDeck(BaseModel):
        evaluated_questions: List[SingleQuestionFormat]

    last_message = state['messages'][-1].content
    
    to_format = last_message + state['first_submission']
    messages = [SystemMessage(content=SINGLE_QUESTIONS_FORMATTER_SYSTEM_PROMPT),
                HumanMessage(content=f"""Message to format: {to_format}""")]

    gpt_mini_model = ChatOpenAI(model="gpt-4o-mini", api_key=os.environ.get('OPENAI_API_KEY'))
    structured_model = gpt_mini_model.with_structured_output(SingleQuestionDeck)
    response = structured_model.invoke(messages)

    return {'current_knowledge_state': response.dict()}

def call_reflection(state):
    new_step = state['reflection_steps'] + 1
    model = ChatOpenAI(model="gpt-4o-mini", api_key=os.environ.get('OPENAI_API_KEY'))
    last_message = state['messages'][-1].content
    messages = [
        SystemMessage(content=EVALUATOR_REFLECTION_PROMPT),
        HumanMessage(content=f"The evaluation is the following: {last_message}")
        ]
    llm_response = model.invoke(messages)
    return {'messages': llm_response, 'reflection_steps': new_step}

def supervisor(state):
    class RouteResponse(BaseModel):
        next: Literal["evaluator_with_feedback","call_reflection", "format_cks"]

    model = ChatOpenAI(model='gpt-4o-mini',  api_key=os.environ.get('OPENAI_API_KEY'))
    structured_model = model.with_structured_output(RouteResponse)
    messages = [SystemMessage(content=SUPERVISOR_PROMPT)]
    response = structured_model.invoke(messages + state['messages'])
    final_reponse = response.dict()['next']
    print(final_reponse)
    return {'next': final_reponse}

def evaluator_with_feedback(state):
    model = ChatOpenAI(model="gpt-4o-mini", api_key=os.environ.get('OPENAI_API_KEY'))
    evaluator_tools = [query_lesson]

    last_message = state['messages'][-1].content

    messages = [
        SystemMessage(content=EVALUATOR_SYSTEM_PROMPT_ONE_SHOT_FEEDBACK.format(
            blooms_state=state['blooms_state'],
            concepts_to_evaluate=', '.join(state['concepts_to_evaluate'],
            first_submission=state['first_submission'])
            )),
        HumanMessage(content=last_message)
    ]
    
    model = model.bind_tools(evaluator_tools)
    llm_response = model.invoke(messages)

    return {'messages': llm_response}

def should_continue_reflection(state):
    if state['next'] == 'format_cks' or state['reflection_steps'] > 2:
        return 'format_cks'
    else:
        return state['next']