"""Every experiment CLI must be able to render its own --help output.

argparse applies %-formatting to help strings. An unescaped '%' therefore
raises at --help time on Python 3.11 and, since Python 3.14, already when the
argument is declared. These tests render the help of every discovered CLI so
the whole class of defect is caught for any supported interpreter.
"""

import importlib
import pkgutil
import sys

import pytest

import src.experiments


def _discover_cli_modules() -> list[str]:
    module_names = []

    for module_info in pkgutil.walk_packages(
        src.experiments.__path__,
        prefix="src.experiments.",
    ):
        module = importlib.import_module(module_info.name)

        if hasattr(module, "parse_args"):
            module_names.append(module_info.name)

    return sorted(module_names)


CLI_MODULES = _discover_cli_modules()


def test_experiment_cli_modules_are_discovered():
    assert len(CLI_MODULES) >= 20


@pytest.mark.parametrize("module_name", CLI_MODULES)
def test_cli_help_renders_without_error(module_name, monkeypatch, capsys):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(sys, "argv", [module_name, "--help"])

    with pytest.raises(SystemExit) as exit_info:
        module.parse_args()

    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip()
