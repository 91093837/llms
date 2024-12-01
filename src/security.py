import datetime as dt
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from utils import OPENAI_MODEL, upload_json, load_prompt


class Score(BaseModel):
    """investing quality score for nasdaq-listed stock provided."""

    risk_score: float = Field(
        description="future uncertainty score for the stock. from 0 to 1."
    )
    overall_score: float = Field(
        description="overall investing quality score for the stock. from -1 to 1."
    )


def security_call():
    # to-do: add prompt input to the database
    llm = ChatOpenAI(model=OPENAI_MODEL)
    structured_llm = llm.with_structured_output(Score, include_raw=True)

    prompt_template = load_prompt(
        "prompts/per-market-fundamental.jinja",
    )
    stocks = ["apple"]  # ...
    output = {}
    for stock in stocks:
        query = [
            ("system", prompt_template),
            ("human", f"what are the expectations for {stock}?"),
        ]

        output_msg = structured_llm.invoke(query)
        output[stock] = output_msg

    # dump raw
    model_dump = {k: v["raw"].model_dump() for k, v in output.items()}
    upload_json(model_dump, path="database/model_dump")

    # dump parsed
    score = {k: v["parsed"].model_dump() for k, v in output.items()}
    ts = int(dt.datetime.now().timestamp())
    for k, v in score.items():
        v.update({"timestamp": ts})

    upload_json(score, path="database/model_score")

    return None


if __name__ == "__main__":
    security_call()
