from pydantic import BaseModel, Field
from typing import Type, Dict, Any

import os
import json
import jinja2
import hashlib
import datetime as dt
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-3.5-turbo"


def get_current_timestamp():
    return int(dt.datetime.now().timestamp())


def create_model(
    name: str, fields: Dict[str, Any], descriptions: Dict[str, str] = None
) -> Type[BaseModel]:
    if descriptions is None:
        descriptions = {}

    class_body = {}
    annotations = {}
    for field_name, field_type in fields.items():
        annotations[field_name] = field_type
        description = descriptions.get(field_name, None)
        if description:
            class_body[field_name] = Field(..., description=description)
        else:
            class_body[field_name] = Field(...)

    class_body["__annotations__"] = annotations
    model = type(name, (BaseModel,), class_body)
    return model


# def create_model(name: str, fields: Dict[str, Any]) -> Type[BaseModel]:
#     class_body = {"__annotations__": fields}
#     model = type(name, (BaseModel,), class_body)
#     return model


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


def load_json(path):
    assert ".json" in path

    if os.path.exists(path):
        with open(path, "r") as file:
            data = json.load(file)
    return data


def upload_json(data: list, path: str, extend=False):
    assert ".json" in path

    if extend and os.path.exists(path):
        with open(path, "r") as f:
            existing_data = json.load(f) or []
        data = existing_data + data

    with open(path, "w") as f:
        json.dump(data, f, sort_keys=False)

    print(f"Data successfully appended to {path}")
    return None
