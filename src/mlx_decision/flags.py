"""Flag values that hold lists (``a,b,c``) and pairs (``a=x,b=y``), and command lines.

A comma inside a value is written ``\\,`` (a backslash itself ``\\\\``), so
any text can travel through ``--options``, ``--label-map`` and
``--levels``; the menus print the command they stand for with the same
escapes, so it can be run again in a script.
"""

import shlex


def split_list(value: str) -> list[str]:
    """``a,b\\,c`` -> ``["a", "b,c"]``, each part stripped."""
    parts, current, escaped = [], [], False
    for char in value:
        if escaped:
            current.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == ",":
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if escaped:
        current.append("\\")
    parts.append("".join(current).strip())
    return parts


def join_list(values: list[str]) -> str:
    """The inverse of ``split_list``."""
    return ",".join(str(value).replace("\\", "\\\\").replace(",", "\\,") for value in values)


def split_pairs(value: str) -> dict[str, str]:
    """``a=x,b=y`` -> ``{"a": "x", "b": "y"}`` (the first ``=`` splits a part);
    ``ValueError`` naming a part without one."""
    result = {}
    for part in split_list(value):
        key, sep, rest = part.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"expected KEY=VALUE,... but got {part!r}")
        result[key.strip()] = rest.strip()
    return result


def join_pairs(values: dict[str, str]) -> str:
    """The inverse of ``split_pairs``."""
    return join_list([f"{key}={value}" for key, value in values.items()])


def command_line(words: list[str]) -> str:
    """Words quoted for a POSIX shell."""
    return " ".join(shlex.quote(str(word)) for word in words)
