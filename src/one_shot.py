import datetime as dt

from utils import create_model, load_prompt, OPENAI_API_KEY, OPENAI_MODEL, upload_yaml
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI


def one_short():
    symbols = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOG", "META"]
    fields = {s: float for s in symbols}
    Score = create_model("Score", fields)

    prompt_template = load_prompt("prompts/one-shot-fundamental.txt")

    PROMPT = """{prompt_template}
    {format_instructions}
    """

    parser = PydanticOutputParser(pydantic_object=Score)
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL, temperature=0
    )
    message = HumanMessagePromptTemplate.from_template(template=PROMPT)
    chat_prompt = ChatPromptTemplate.from_messages([message])
    chat_prompt_with_values = chat_prompt.format_prompt(
        prompt_template=prompt_template,
        format_instructions=parser.get_format_instructions(),
    )
    output = llm(chat_prompt_with_values.to_messages())

    # dump raw
    raw_output = chat_prompt_with_values.model_dump() | output.model_dump()
    upload_yaml(data=raw_output, path="database/model_dump")

    # dump parsed
    portfolio = parser.parse(output.content)
    portfolio = portfolio.model_dump()
    portfolio["ts"] = int(dt.datetime.now().timestamp())
    upload_yaml(data=portfolio, path="database/model_score")

    pass


if __name__ == "__main__":
    one_short()
