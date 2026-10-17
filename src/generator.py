"""Step 6: build the prompt and call the LLM."""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import config

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer questions about the supplied company document using only the "
            "provided context. Read all supplied excerpts before deciding that an "
            "answer is missing. Tables may have broken line order; combine related "
            "labels, periods, and values when the context supports the connection. "
            "Keep the response concise: answer only what was asked in at most "
            "two short sentences. For numeric questions, give only the requested "
            "metric, period, and unit or currency; do not list unrelated figures "
            "or summarize the surrounding table. "
            "If the context does not contain enough evidence to answer, say "
            "\"I couldn't find this in the retrieved document text.\" Do not guess. "
            "For every answer, cite the page number(s) shown with the supporting "
            "context. If the context is empty, say that no text was retrieved.",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


def get_llm():
    if config.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)
    if config.LLM_PROVIDER == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE
        )
    raise ValueError(f"Unknown LLM_PROVIDER: {config.LLM_PROVIDER}")


def format_context(docs) -> str:
    return "\n\n".join(
        f"[Page {d.metadata.get('page')}]\n{d.page_content}" for d in docs
    )


def generate_answer(question: str, docs) -> str:
    chain = PROMPT | get_llm() | StrOutputParser()
    return chain.invoke({"context": format_context(docs), "question": question})
