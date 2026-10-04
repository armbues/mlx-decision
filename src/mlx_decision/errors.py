"""Errors raised for requests that cannot be answered."""


class DecisionError(ValueError):
    """An invalid or unsupported request.

    ``param`` is the dotted path of the offending field, for example
    ``questions.team.criteria``, or ``None`` when no single field is at fault.
    """

    def __init__(self, message: str, param: str | None = None, code: str | None = None):
        super().__init__(message)
        self.message = message
        self.param = param
        self.code = code

    def __str__(self) -> str:
        return f"{self.param}: {self.message}" if self.param else self.message
