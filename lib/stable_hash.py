import dataclasses
import json
import hashlib


def json_default(data):
    try:
        return {"__type": type(data).__name__, **dataclasses.asdict(data)}
    except:
        return type(data).__name__


def stable_hash(data_class):
    return (
        hashlib.md5(
            json.dumps(
                data_class,
                default=json_default,
                ensure_ascii=False,
                sort_keys=True,
                indent=None,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        .digest()
        .hex()
    )
