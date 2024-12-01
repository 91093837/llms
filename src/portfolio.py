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
    load_yaml,
    upload_yaml,
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
    data = load_yaml("database/nasdaq.yaml")
    ts = data[0]["ts"]
    data = [r for r in data if r["ts"] == ts]
    return data


def build_portfolio(tickers: List[dict] = None):
    if tickers is None:
        tickers = load_tickers()

    # building portfolio model
    symbols = [l["symbol"] for l in tickers]
    fields = {s: float for s in symbols}
    description = {
        l["symbol"]: f"portfolio weight in {l['companyName']}" for l in tickers
    }
    Portfolio = create_model("Portfolio", fields, description)

    class DummyChatOpenAI(ChatOpenAI):
        def __init__(self, **kwargs):
            super(ChatOpenAI, self).__init__()

        def __call__(self, *args, **kwargs):
            output = Portfolio(**{k: 0 for k in fields}).model_dump().__repr__()
            output = output.replace("'", '"')
            content = f"```\n{output}\n```"
            return AIMessage(content)

    # asks llm
    values = {"date": dt.datetime.now().strftime("%Y-%m-%d")}
    prompt_template = load_jinja_prompt("prompts/per-market-fundamental.jinja", values)

    PROMPT = """{prompt_template}
    {format_instructions}
    """

    parser = PydanticOutputParser(pydantic_object=Portfolio)
    llm = DummyChatOpenAI(
        openai_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL, temperature=0
    )
    message = HumanMessagePromptTemplate.from_template(template=PROMPT)
    chat_prompt = ChatPromptTemplate.from_messages([message])
    chat_prompt_with_values = chat_prompt.format_prompt(
        prompt_template=prompt_template,
        format_instructions=parser.get_format_instructions(),
    )
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
    upload_yaml(data=raw_output, path="database/model_dump.yaml")

    # dump parsed
    portfolio = parser.parse(output.content)
    portfolio = portfolio.model_dump()
    portfolio["date"] = dt.datetime.now().strftime("%Y-%m-%d")

    portfolio["execution_ts"] = ts
    upload_yaml(data=portfolio, path="database/model_portfolio.yaml")

    return None


if __name__ == "__main__":
    build_portfolio()
