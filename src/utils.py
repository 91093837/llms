from pydantic import BaseModel
from typing import Type, Dict, Any

import os
import yaml
import datetime as dt
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-3.5-turbo"


def create_model(name: str, fields: Dict[str, Any]) -> Type[BaseModel]:
    class_body = {"__annotations__": fields}
    model = type(name, (BaseModel,), class_body)
    return model


def load_prompt(path: str):
    prompt = open(path, "r").read().replace("\n", "")
    return prompt


def upload_yaml(data: dict, path: str, name: str = None):
    # if name is None:
    now = dt.datetime.now().strftime("%Y%m%d")
    path += f"/{now}.yaml"

    with open(path, "w") as yaml_file:
        yaml.safe_dump(data, yaml_file, sort_keys=False)
