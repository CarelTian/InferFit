from typing import Any

def _positive(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(f"config.{key} must be a positive integer")
    return value
