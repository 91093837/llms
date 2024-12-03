import tiktoken
import numpy as np
import datetime as dt
from typing import List
from utils import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    create_model,
    load_jinja_prompt,
    get_current_timestamp,
    hash_string,
    build_portfolios,
    JSONFile,
    load_tickers,
)
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
    - asks directly to return a portfolio with all nasdaq stocks
    - with format_instructions
    """

    # asks llm
    values = {"date": dt.datetime.now().strftime("%Y-%m-%d")}
    prompt_name = "portfolio"
    prompt_template = load_jinja_prompt(f"prompts/{prompt_name}.jinja", values)

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
    JSONFile(
        data=[raw_output], path="database/2-model_output/model_dump.json", extend=True
    )

    # dump parsed
    raw_portfolio = parser.parse(output.content)

    JSONFile(
        data=[
            {
                "name": prompt_name,
                "portfolio": raw_portfolio.model_dump(),
                "date": dt.datetime.now().strftime("%Y-%m-%d"),
                "execution_ts": get_current_timestamp(),
            }
        ],
        path="database/2-model_output/model_raw_portfolio.json",
        extend=True,
    )
    JSONFile(
        data=build_portfolios(raw_portfolio, name=prompt_name),
        path="database/3-reporting/portfolio.json",
        extend=True,
    )
    return None


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
            output = (
                Portfolio(**{k: np.random.uniform(0, 1) for k in fields})
                .model_dump()
                .__repr__()
            )
            output = output.replace("'", '"')
            content = f"```\n{output}\n```"
            return AIMessage(content)

    llm = DummyChatOpenAI(
        openai_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL, temperature=0
    )

    model_1(llm, tickers, parser)
    return None


if __name__ == "__main__":
    run()
