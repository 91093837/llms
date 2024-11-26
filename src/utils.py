from pydantic import BaseModel
from typing import Type, Dict, Any

import os
import yaml
import jinja2
import hashlib
import datetime as dt
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-3.5-turbo"


def create_model(name: str, fields: Dict[str, Any]) -> Type[BaseModel]:
    class_body = {"__annotations__": fields}
    model = type(name, (BaseModel,), class_body)
    return model


def hash_string(str):
    return hashlib.md5(str.encode()).hexdigest()


def load_jinja_prompt(path: str, values: dict):
    loader = jinja2.FileSystemLoader(searchpath="./")
    template = jinja2.Environment(loader=loader, autoescape=True)
    template = template.get_template(path)
    output = template.render(**values)
    return output


def load_prompt(path: str):
    prompt = open(path, "r").read().replace("\n", "")
    return prompt


def upload_yaml(data: dict, path: str, name: str = None):
    # if name is None:
    now = dt.datetime.now().strftime("%Y%m%d")
    path += f"/{now}.yaml"

    with open(path, "w") as yaml_file:
        yaml.safe_dump(data, yaml_file, sort_keys=False)
