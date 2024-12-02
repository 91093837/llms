import tiktoken
import datetime as dt
from typing import List
from utils import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    create_model,
    load_prompt,
    load_jinja_prompt,
    get_current_timestamp,
    load_json,
    upload_json,
    hash_string,
)
from pydantic import BaseModel
from langchain.prompts.chat import (
    ChatPromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.messages.ai import AIMessage


def load_tickers():
    data = load_json("database/nasdaq.json")
    ts = data[0]["ts"]
    data = [r for r in data if r["ts"] == ts]
    return data


def model_1(llm, tickers: List[dict], parser):
    """
    - one-shot
    - uses pydantic_object to format output
    """

    # asks llm
    values = {"date": dt.datetime.now().strftime("%Y-%m-%d")}
    prompt_template = load_jinja_prompt("prompts/per-market-fundamental.jinja", values)

    PROMPT = """{prompt_template}
    {format_instructions}
    """

    message = HumanMessagePromptTemplate.from_template(template=PROMPT)
    chat_prompt = ChatPromptTemplate.from_messages([message])
    chat_prompt_with_values = chat_prompt.format_prompt(
        prompt_template=prompt_template,
        format_instructions=parser.get_format_instructions(),
    )

    # send prompt to llm
    output = llm(chat_prompt_with_values.to_messages())

    enc = tiktoken.encoding_for_model(OPENAI_MODEL)
    tokens = enc.encode(chat_prompt_with_values.to_string())

    # dump raw
    ts = get_current_timestamp()

    raw_output = (
        chat_prompt_with_values.model_dump()
        | output.model_dump()
        | {"prompt_hash": hash_string(PROMPT)}
        | {"token_size": len(tokens)}
        | {"execution_ts": ts}
    )
    upload_json(data=[raw_output], path="database/model_dump.json", extend=True)

    # dump parsed
    portfolio = parser.parse(output.content)
    portfolio = portfolio.model_dump()

    output = {}
    output["date"] = dt.datetime.now().strftime("%Y-%m-%d")
    output["execution_ts"] = ts
    output["name"] = "model_1"
    output["portfolio"] = portfolio
    upload_json(data=[output], path="database/model_portfolio.json", extend=True)
    return None


def model_2(llm, tickers: List[dict], parser):
    """
    - one-shot
    - asks the model to return a json-object
    """
    symbols = [l["symbol"] for l in tickers]
    values = {"date": dt.datetime.now().strftime("%Y-%m-%d"), "stock_list": symbols}
    prompt_template = load_jinja_prompt("prompts/per-market-fundamental.jinja", values)
    PROMPT = """{prompt_template}"""

    # send prompt to llm
    output = llm(PROMPT)

    enc = tiktoken.encoding_for_model(OPENAI_MODEL)
    tokens = enc.encode(PROMPT)

    # dump raw
    ts = get_current_timestamp()

    raw_output = (
        {"prompt": PROMPT}
        | output.model_dump()
        | {"prompt_hash": hash_string(PROMPT)}
        | {"token_size": len(tokens)}
        | {"execution_ts": ts}
    )
    upload_json(data=[raw_output], path="database/model_dump.json", extend=True)

    # dump parsed
    portfolio = parser.parse(output.content)
    portfolio = portfolio.model_dump()

    output = {}
    output["date"] = dt.datetime.now().strftime("%Y-%m-%d")
    output["execution_ts"] = ts
    output["name"] = "model_1"
    output["portfolio"] = portfolio
    upload_json(data=[output], path="database/model_portfolio.json", extend=True)

    return None


def model_3():
    """
    - two-shots (first asks a general overview, then asks the portfolio)
    """
    return None


def run(tickers: dict):
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
            output = Portfolio(**{k: 0 for k in fields}).model_dump().__repr__()
            output = output.replace("'", '"')
            content = f"```\n{output}\n```"
            return AIMessage(content)

    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL, temperature=0
    )
    model_1(llm, tickers, parser)

    return None


if __name__ == "__main__":
    run()
