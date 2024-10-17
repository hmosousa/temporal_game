from typing import TypedDict, Literal

class Relation(TypedDict):
    source: str
    target: str
    type: Literal["<", ">", "=", "-"]