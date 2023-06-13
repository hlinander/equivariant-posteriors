import dataclasses
import json
import hashlib


def json_default(data):
    try:
        return {"__type": type(data).__name__, **dataclasses.asdict(data)}
    except:
        try:
            return data.__name__
        except:
            return type(data).__name__


def json_dumps_dataclass(data_class):
    return json.dumps(
        data_class,
        default=json_default,
        ensure_ascii=False,
        sort_keys=True,
        indent=None,
        separators=(",", ":"),
    ).encode("utf-8")


def stable_hash(data_class):
    return hashlib.md5(json_dumps_dataclass(data_class)).digest().hex()
