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


class PlatformError(RuntimeError):
    """MLX cannot run the models here: they need an Apple Silicon Mac (a Metal GPU)."""


def require_metal() -> None:
    import mlx.core as mx

    if not mx.metal.is_available():
        raise PlatformError(
            "mlx-decision runs models on Apple Silicon Macs; MLX finds no Metal GPU here"
        )
