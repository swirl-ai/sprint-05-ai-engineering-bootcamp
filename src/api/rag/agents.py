from pydantic import BaseModel, Field
from typing import List
import instructor
from openai import OpenAI
from langsmith import traceable, get_current_run_tree
from langchain_core.messages import AIMessage

from api.rag.utils.utils import lc_messages_to_regular_messages, format_ai_message
from api.rag.utils.utils import prompt_template_config
from api.core.config import config


client = instructor.from_openai(OpenAI(api_key=config.OPENAI_API_KEY))


class MCPToolCall(BaseModel):
    name: str
    arguments: dict
    server: str

class ToolCall(BaseModel):
    name: str
    arguments: dict


class RAGUsedContext(BaseModel):
    id: str
    description: str


class ProductQAAgentResponse(BaseModel):
    answer: str
    tool_calls: List[MCPToolCall] = Field(default_factory=list)
    final_answer: bool = Field(default=False)
    retrieved_context_ids: List[RAGUsedContext]


class IntentRouterAgentResponse(BaseModel):
    user_intent: str
    answer: str


class ShoppingCartAgentResponse(BaseModel):
    answer: str
    tool_calls: List[ToolCall] = Field(default_factory=list)
    final_answer: bool = Field(default=False)


### Product QA Agent ###

@traceable(
    name="product_qa_agent",
    run_type="llm",
    metadata={"ls_provider": config.GENERATION_MODEL_PROVIDER, "ls_model_name": config.GENERATION_MODEL}
)
def product_qa_agent_node(state):

    prompt_template = prompt_template_config(config.RAG_PROMPT_TEMPLATE_PATH, "product_qa_agent")

    prompt = prompt_template.render(
        available_tools=state.product_qa_available_tools
    )

    messages = state.messages

    conversation = []

    for msg in messages:
        conversation.append(lc_messages_to_regular_messages(msg))

    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1",
        response_model=ProductQAAgentResponse,
        messages=[{"role": "system", "content": prompt}, *conversation],
        temperature=0,
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens,
        }

    ai_message = format_ai_message(response)

    return {
        "messages": [ai_message],
        "mcp_tool_calls": response.tool_calls,
        "product_qa_iteration": state.product_qa_iteration + 1,
        "answer": response.answer,
        "final_answer": response.final_answer,
        "retrieved_context_ids": response.retrieved_context_ids,
    }

### Intent Router Agent ###

@traceable(
    name="intent_router_agent",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1"}
)
def intent_router_agent_node(state) -> dict:

    prompt_template = prompt_template_config(config.RAG_PROMPT_TEMPLATE_PATH, "intent_router_agent")

    prompt = prompt_template.render()

    messages = state.messages

    conversation = []

    for msg in messages:
        conversation.append(lc_messages_to_regular_messages(msg))

    client = instructor.from_openai(OpenAI())

    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1",
        response_model=IntentRouterAgentResponse,
        messages=[{"role": "system", "content": prompt}, *conversation],
        temperature=0,
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens,
        }
        trace_id = str(getattr(current_run, "trace_id", current_run.id))

    if response.user_intent == "product_qa":
        ai_message = []
    else:
        ai_message = [AIMessage(
            content=response.answer,
        )]

    return {
        "messages": ai_message,
        "answer": response.answer,
        "user_intent": response.user_intent,
        "trace_id": trace_id
    }


### Shopping Cart Agent ###

@traceable(
    name="shopping_cart_agent",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1"}
)
def shopping_cart_agent_node(state) -> dict:

    prompt_template = prompt_template_config(config.RAG_PROMPT_TEMPLATE_PATH, "shopping_cart_agent")

    prompt = prompt_template.render(
        available_tools=state.shopping_cart_available_tools,
        user_id=state.user_id,
        cart_id=state.cart_id
    )

    messages = state.messages

    conversation = []

    for msg in messages:
        conversation.append(lc_messages_to_regular_messages(msg))

    client = instructor.from_openai(OpenAI())

    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1",
        response_model=ShoppingCartAgentResponse,
        messages=[{"role": "system", "content": prompt}, *conversation],
        temperature=0,
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens,
        }

        ai_message = format_ai_message(response)

    return {
        "messages": [ai_message],
        "tool_calls": response.tool_calls,
        "shopping_cart_iteration": state.shopping_cart_iteration + 1,
        "answer": response.answer,
        "final_answer": response.final_answer
    }
