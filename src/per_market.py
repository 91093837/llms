import datetime as dt

from utils import (
    create_model,
    load_prompt,
    load_jinja_prompt,
    OPENAI_API_KEY,
    OPENAI_MODEL,
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

symbols = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOG", "META"]
fields = {s: float for s in symbols}
Score = create_model("Score", fields)


class DummyModel(BaseModel):
    pass


class DummyChatOpenAI(ChatOpenAI):
    def __init__(self, **kwargs):
        super(ChatOpenAI, self).__init__()

    def __call__(self, *args, **kwargs):
        output = Score(**{k: 0 for k in fields}).model_dump().__repr__()
        output = output.replace("'", '"')
        content = f"```\n{output}\n```"
        return AIMessage(content)


def one_short():

    values = {"date": dt.datetime.now().strftime("%Y-%m-%d")}
    prompt_template = load_jinja_prompt("prompts/per-market-fundamental.jinja", values)

    PROMPT = """{prompt_template}
    {format_instructions}
    """

    parser = PydanticOutputParser(pydantic_object=Score)
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

    # dump raw
    raw_output = (
        chat_prompt_with_values.model_dump()
        | output.model_dump()
        | {"prompt_hash": hash_string(PROMPT)}
    )
    upload_yaml(data=raw_output, path="database/model_dump")

    # dump parsed
    portfolio = parser.parse(output.content)
    portfolio = portfolio.model_dump()
    portfolio["ts"] = int(dt.datetime.now().timestamp())
    upload_yaml(data=portfolio, path="database/model_score")

    pass


if __name__ == "__main__":
    one_short()
