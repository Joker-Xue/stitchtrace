class StitchTraceError(Exception):
    """Stable coded error at the external boundary."""
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f'{code}: {message}')


def require(condition, code, message):
    if not condition:
        raise StitchTraceError(code, message)
