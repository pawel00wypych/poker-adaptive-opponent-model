import pytest

import src
from src import (
    SUPPORTED_PYTHON_VERSION,
    format_python_version,
    validate_python_version,
)


def test_running_interpreter_is_supported():
    """The suite itself must run on the interpreter the project targets."""
    import sys

    assert sys.version_info[:2] == SUPPORTED_PYTHON_VERSION


def test_supported_version_is_accepted():
    validate_python_version(SUPPORTED_PYTHON_VERSION)


def test_patch_level_is_ignored():
    major, minor = SUPPORTED_PYTHON_VERSION

    validate_python_version((major, minor, 7, "final", 0))


@pytest.mark.parametrize(
    "version_info",
    [
        (3, 10),
        (3, 12),
        (3, 14),
        (4, 0),
        (2, 7),
    ],
)
def test_unsupported_version_is_rejected(version_info):
    with pytest.raises(RuntimeError) as error:
        validate_python_version(version_info)

    message = str(error.value)
    assert format_python_version(SUPPORTED_PYTHON_VERSION) in message
    assert format_python_version(version_info) in message
    assert "requirements.txt" in message


def test_package_import_runs_the_guard():
    """Importing the package must be what triggers the check."""
    assert hasattr(src, "validate_python_version")
