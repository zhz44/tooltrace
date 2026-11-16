"""tooltrace: record and summarise the shell commands a coding agent runs."""

from .classifier import classify_tool
from .hook import Tracer, install, install_from_env

__version__ = "0.1.0"

__all__ = ["Tracer", "classify_tool", "install", "install_from_env", "__version__"]
