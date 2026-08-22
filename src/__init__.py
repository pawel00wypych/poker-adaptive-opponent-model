"""Project package root.

The interpreter version is verified here because every entry point - the
`python -m src.experiments....` commands and the test suite alike - imports
this package first. An unsupported interpreter has already caused behaviour
that differed between versions, so the mismatch is rejected up front instead
of surfacing later as a confusing failure.
"""

import sys

SUPPORTED_PYTHON_VERSION = (3, 11)


def format_python_version(version_info: tuple[int, ...]) -> str:
    return ".".join(str(part) for part in version_info[:2])


def validate_python_version(
    version_info: tuple[int, ...] | None = None,
) -> None:
    """Raise when running on an interpreter the project does not support."""
    running_version = tuple(
        sys.version_info[:2] if version_info is None else version_info[:2]
    )

    if running_version == SUPPORTED_PYTHON_VERSION:
        return

    required = format_python_version(SUPPORTED_PYTHON_VERSION)
    found = format_python_version(running_version)

    raise RuntimeError(
        f"This project requires Python {required}, but it is running on "
        f"Python {found}. Recreate the virtual environment with the "
        f"supported interpreter:\n"
        f"    py -{required} -m venv .venv\n"
        f"    .venv\\Scripts\\python.exe -m pip install -r requirements.txt"
    )


validate_python_version()
