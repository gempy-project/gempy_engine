import re

#: Mesher errors are ``ValueError("<code>: <detail>")``; the code is a snake_case word with at least one underscore.
_MESH_CODE = re.compile(r"^([a-z][a-z0-9]*(?:_[a-z0-9]+)+): ")


class GemPyEngineInputError(Exception):
    """Exception raised for errors in the input data validation of GemPy Engine.

    Attributes:
        message: The human-readable message (unchanged from earlier versions, callers match on it).
        reason: A stable machine-readable code, e.g. ``missing_orientations``. ``inconsistent_input`` marks
            arrays that disagree with each other, which is a bug in whatever built them, not in the model.
        context: Engine indices and counts for the failure, e.g. ``{"stack_index": 0}``.
    """

    def __init__(self, message: str, reason: str = "inconsistent_input", context: dict | None = None):
        self.message = message
        self.reason = reason
        self.context = context or {}
        super().__init__(self.message)


class MeshExtractionError(ValueError):
    """A mesher error, raised once at the extraction entry point from the ``ValueError`` of the step that failed.

    ``str()`` is the original message, so existing ``ValueError`` handlers and message matches still work.
    ``code`` is its leading code, e.g. ``unsupported_sampled_junction``.
    """

    reason = "mesh_extraction_failed"

    def __init__(self, message: str, code: str):
        super().__init__(message)
        self.code = code
        self.context = {"code": code}

    @classmethod
    def from_error(cls, error: BaseException) -> "MeshExtractionError | None":
        """The ``MeshExtractionError`` for a coded ``ValueError``, or None for any other error."""
        if isinstance(error, cls) or not isinstance(error, ValueError):
            return None
        match = _MESH_CODE.match(str(error))
        return cls(str(error), match.group(1)) if match else None
