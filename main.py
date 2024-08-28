"""Main entrypoint for the app."""
import asyncio
import os
os.environ["LANGSMITH_TRACING"] = "false"
from datetime import datetime
from operator import itemgetter
from typing import List, Optional, Sequence, Tuple, Union

from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate
)
# from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain.retrievers import (
    ContextualCompressionRetriever,
    TavilySearchAPIRetriever,
)

from langchain.schema.document import Document
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.retriever import BaseRetriever

from langchain.schema.runnable import (
    ConfigurableField,
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnableMap,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Backup
from langserve import add_routes
from pydantic import BaseModel, Field
from uuid import UUID

from langchain_community.chat_models import ChatTongyi
import yaml


script_dir = os.path.dirname(os.path.realpath(__file__))
# Load the YAML file
with open(os.path.join(script_dir, 'secrets', 'api_keys.yaml'), 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
os.environ["DASHSCOPE_API_KEY"] = config['DASHSCOPE_API_KEY']
os.environ["TAVILY_API_KEY"] = config['TAVILY_API_KEY']


RESPONSE_TEMPLATE = """\
You are an expert researcher and writer, tasked with answering any question.

Generate a comprehensive and informative, yet concise answer of 250 words or less for the \
given question based solely on the provided search results (URL and content). You must \
only use information from the provided search results. Use an unbiased and \
journalistic tone. Combine search results together into a coherent answer. Do not \
repeat text. Cite search results using [${{number}}] notation. Only cite the most \
relevant results that answer the question accurately. Place these citations at the end \
of the sentence or paragraph that reference them - do not put them all at the end. If \
different results refer to different entities within the same name, write separate \
answers for each entity. If you want to cite multiple results for the same sentence, \
format it as `[${{number1}}] [${{number2}}]`. However, you should NEVER do this with the \
same number - if you want to cite `number1` multiple times for a sentence, only do \
`[${{number1}}]` not `[${{number1}}] [${{number1}}]`

You should use bullet points in your answer for readability. Put citations where they apply \
rather than putting them all at the end.

If there is nothing in the context relevant to the question at hand, just say "Hmm, \
I'm not sure." Don't try to make up an answer.

Anything between the following `context` html blocks is retrieved from a knowledge \
bank, not part of the conversation with the user.

<context>
    {context}
<context/>

REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm \
not sure." Don't try to make up an answer. Anything between the preceding 'context' \
html blocks is retrieved from a knowledge bank, not part of the conversation with the \
user. The current date is {current_date}.
"""

REPHRASE_TEMPLATE = """\
Given the following conversation and a follow up question, rephrase the follow up \
question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone Question:"""



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    chat_history: List[Tuple[str, str]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "question", "output": "answer"}},
    )


def create_retriever_chain(
    llm: BaseLanguageModel, retriever: BaseRetriever
) -> Runnable:
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(REPHRASE_TEMPLATE)
    condense_question_chain = (
        CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()
    ).with_config(
        run_name="CondenseQuestion",
    )
    conversation_chain = condense_question_chain | retriever
    return RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),
            conversation_chain.with_config(run_name="RetrievalChainWithHistory"),
        ),
        (
            RunnableLambda(itemgetter("question")).with_config(
                run_name="Itemgetter:question"
            )
            | retriever
        ).with_config(run_name="RetrievalChainWithNoHistory"),
    ).with_config(run_name="RouteDependingOnChatHistory")


def serialize_history(request: ChatRequest):
    chat_history = request.get("chat_history", [])
    converted_chat_history = []
    for message in chat_history:
        if message[0] == "human":
            converted_chat_history.append(HumanMessage(content=message[1]))
        elif message[0] == "ai":
            converted_chat_history.append(AIMessage(content=message[1]))
    return converted_chat_history


def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


def create_chain(
    llm: BaseLanguageModel,
    retriever: BaseRetriever,
) -> Runnable:
    retriever_chain = create_retriever_chain(llm, retriever) | RunnableLambda(
        format_docs
    ).with_config(run_name="FormatDocumentChunks")
    _context = RunnableMap(
        {
            "context": retriever_chain.with_config(run_name="RetrievalChain"),
            "question": RunnableLambda(itemgetter("question")).with_config(
                run_name="Itemgetter:question"
            ),
            "chat_history": RunnableLambda(itemgetter("chat_history")).with_config(
                run_name="Itemgetter:chat_history"
            ),
        }
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RESPONSE_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    ).partial(current_date=datetime.now().isoformat())

    response_synthesizer = (prompt | llm | StrOutputParser()).with_config(
        run_name="GenerateResponse",
    )
    return (
        {
            "question": RunnableLambda(itemgetter("question")).with_config(
                run_name="Itemgetter:question"
            ),
            "chat_history": RunnableLambda(serialize_history).with_config(
                run_name="SerializeHistory"
            ),
        }
        | _context
        | response_synthesizer
    )


def get_retriever():
    base_tavily_retriever = TavilySearchAPIRetriever(
        k=6, include_raw_content=True, include_images=True
    )
    return base_tavily_retriever


llm = ChatTongyi(
    temperature=0.1,
    model_name='qwen-plus',
    max_retries=10,
    top_p=0.8,
    max_tokens=2000,
    streaming=True,
)

retriever = get_retriever()

chain = create_chain(llm, retriever)

add_routes(
    app, chain, path="/chat", input_type=ChatRequest, config_keys=["configurable"]
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
