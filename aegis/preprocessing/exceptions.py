"""
Exceptions used by the AEGIS preprocessing layer.

Version: 0.6.0
"""


class PreprocessingError(Exception):
    """Base preprocessing exception."""


class InvalidSampleError(PreprocessingError):
    """Raised when a sample cannot be converted safely."""


class UnsupportedLabelError(PreprocessingError):
    """Raised when an unsupported information-integrity label is used."""