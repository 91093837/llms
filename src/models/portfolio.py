import tiktoken
import datetime as dt

from tiktoken import MODEL_PREFIX_TO_ENCODING
from typing import List
from utils import (
    MODELS,
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
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.output_parsers import PydanticOutputParser


def model() -> List[JSONFile]:
    pass


def model_1(llm, config) -> List[JSONFile]:
    """
    - one-shot
    - asks directly to return a portfolio with all nasdaq stocks
    - with format_instructions
    """

    tickers, parser = config["tickers"], config["parser"]

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

    # dump raw
    ts = get_current_timestamp()
    raw_output = (
        chat_prompt_with_values.model_dump()
        | output.model_dump()
        | {"prompt_hash": hash_string(PROMPT)}
        | {"execution_ts": ts}
    )

    if config["model_name"] in MODEL_PREFIX_TO_ENCODING:
        enc = tiktoken.encoding_for_model(config["model_name"])
        tokens = enc.encode(chat_prompt_with_values.to_string())
        raw_output = raw_output | {"token_size": len(tokens)}

    JSONFile(
        data=[raw_output],
        path=f"{DATABASE_NAME}/2-model_output/model_dump.json",
        extend=True,
    )

    # dump parsed
    raw_portfolio = parser.parse(output.content)

    name = f"{config['model_name']}/{prompt_name}"
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

    for name, conf in MODELS.items():
        llm = conf["chat"]
        for model_name in conf["models"]:
            llm(model_name=model_name, temperature=0)
            model_1(
                llm,
                config={"tickers": tickers, "parser": parser, "model_name": model_name},
            )

    return None


if __name__ == "__main__":
    run()
