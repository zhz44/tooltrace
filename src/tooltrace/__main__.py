"""Allows ``python -m tooltrace``."""

import sys

from .cli import main

sys.exit(main())
