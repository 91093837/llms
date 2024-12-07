import re
import tiktoken
import numpy as np
import datetime as dt
from typing import List
from utils import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    DATABASE_NAME,
    create_model,
    load_jinja_prompt,
    get_current_timestamp,
    hash_string,
    build_portfolios,
    JSONFile,
    load_tickers,
    ignore_exception,
)
import logging
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.messages.ai import AIMessage


def model() -> List[JSONFile]:
    pass


def model_1(llm, tickers: List[dict], parser) -> List[JSONFile]:
    """
    - one-shot
    """

    prompt_name = "ranking"
    symbols = [l["symbol"] for l in tickers]
    symbols_str = str(symbols).replace("'", "")
    values = {
        "date": dt.datetime.now().strftime("%Y-%m-%d"),
        "symbols": symbols_str,
    }
    PROMPT = load_jinja_prompt(f"prompts/{prompt_name}.jinja", values)

    message = HumanMessagePromptTemplate.from_template(template=PROMPT)
    chat_prompt = ChatPromptTemplate.from_messages([message])
    chat_prompt_with_values = chat_prompt.format_prompt()

    # send prompt to llm
    output = llm(chat_prompt_with_values.to_messages())

    enc = tiktoken.encoding_for_model(OPENAI_MODEL)
    tokens = enc.encode(chat_prompt_with_values.to_string())

    # dump raw
    ts = get_current_timestamp()
    raw_output = (
        chat_prompt.model_dump()
        | output.model_dump()
        | {"prompt_hash": hash_string(PROMPT)}
        | {"token_size": len(tokens)}
        | {"execution_ts": ts}
    )
    JSONFile(
        data=[raw_output],
        path=f"{DATABASE_NAME}/2-model_output/model_dump.json",
        extend=True,
    )

    # dump parsed
    content = re.sub(r"[^a-zA-Z0-9\s]", "", output.content).split()
    content = {content[i]: (len(content) - i) for i in range(len(content))}
    diff = set(content.keys()).symmetric_difference(symbols)
    if len(diff) > 0:
        logging.warning(f"failed to maintain structure, new {diff}")
    content = content | {k: 0 for k in diff}
    content = {k: v for k, v in content.items() if k in symbols}
    raw_portfolio = parser.pydantic_object(**content)

    name = f"{OPENAI_MODEL}/{prompt_name}"
    JSONFile(
        data=[
            {
                "name": name,
                "portfolio": raw_portfolio.model_dump(),
                "date": dt.datetime.now().strftime("%Y-%m-%d"),
                "execution_ts": get_current_timestamp(),
            }
        ],
        path=f"{DATABASE_NAME}/2-model_output/model_raw_portfolio.json",
        extend=True,
    )
    JSONFile(
        data=build_portfolios(raw_portfolio, name=name),
        path=f"{DATABASE_NAME}/3-reporting/portfolio.json",
        extend=True,
    )
    return None


@ignore_exception
def run(tickers: dict = None):
    if not tickers:
        tickers = load_tickers()

    # building portfolio model
    symbols = [l["symbol"] for l in tickers]
    fields = {s: float for s in symbols}
    description = {
        l["symbol"]: f"portfolio weight in {l['companyName']}" for l in tickers
    }
    Portfolio = create_model("Portfolio", fields, description)
    parser = PydanticOutputParser(pydantic_object=Portfolio)

    class DummyChatOpenAI(ChatOpenAI):
        def __init__(self, **kwargs):
            super(ChatOpenAI, self).__init__()

        def __call__(self, *args, **kwargs):
            output = np.random.choice(symbols, size=len(symbols), replace=False)
            content = f"```\n{output}\n```"
            return AIMessage(content)

    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL, temperature=0
    )

    model_1(llm, tickers, parser)
    return None


if __name__ == "__main__":
    run()
