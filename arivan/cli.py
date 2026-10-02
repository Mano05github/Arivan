"""Console entry point for Arivan.

The native runtime is inherited from Colibri while Arivan's shared pager and
agent layer are developed. This adapter keeps legacy COLI_* variables working
and gives new installations stable ARIVAN_* names.
"""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys

from ._version import __version__
from .memory import get_profile, render_profile


_ENV_ALIASES = {
    "ARIVAN_MODEL": "COLI_MODEL",
    "ARIVAN_MODEL_MIRROR": "COLI_MODEL_MIRROR",
    "ARIVAN_API_KEY": "COLI_API_KEY",
    "ARIVAN_COLOR": "COLI_COLOR",
}


def _configure_output() -> None:
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass


def _engine_script() -> Path:
    return Path(__file__).resolve().parent.parent / "c" / "coli"


def _apply_environment_aliases() -> None:
    for new_name, legacy_name in _ENV_ALIASES.items():
        value = os.environ.get(new_name)
        if value is not None:
            os.environ.setdefault(legacy_name, value)


def _selected_model(arguments: list[str]) -> str | None:
    for index, value in enumerate(arguments):
        if value == "--model" and index + 1 < len(arguments):
            return arguments[index + 1]
        if value.startswith("--model="):
            return value.split("=", 1)[1]
    return os.environ.get("ARIVAN_MODEL") or os.environ.get("COLI_MODEL")


def _model_arch(arguments: list[str]) -> str:
    model = _selected_model(arguments)
    if not model:
        return "generic"
    engine_dir = _engine_script().parent
    sys.path.insert(0, str(engine_dir))
    try:
        from family_registry import resolve_model

        return resolve_model(model).descriptor.id
    except (OSError, ValueError, KeyError, ModuleNotFoundError):
        # The inherited CLI owns the detailed checkpoint error. A profile must
        # not hide it behind a second, less useful parser failure.
        return "generic"


def _extract_profile(arguments: list[str]) -> tuple[list[str], str | None]:
    result: list[str] = []
    selected = None
    index = 0
    while index < len(arguments):
        value = arguments[index]
        if value == "--memory-profile":
            if index + 1 >= len(arguments):
                raise SystemExit("--memory-profile requires 8gb, 16gb, or 32gb")
            selected = arguments[index + 1]
            index += 2
            continue
        if value.startswith("--memory-profile="):
            selected = value.split("=", 1)[1]
            index += 1
            continue
        result.append(value)
        index += 1
    return result, selected


def _profile_command(arguments: list[str]) -> int:
    name = "8gb"
    json_output = False
    for value in arguments:
        if value == "--json":
            json_output = True
        elif not value.startswith("-"):
            name = value
        else:
            raise SystemExit(f"unknown profile option: {value}")
    print(render_profile(get_profile(name), json_output=json_output))
    return 0


def _welcome_without_model() -> int:
    print()
    print("அறிவன் — Think. Code. Solve.")
    print()
    print("Ready to explore your codebase.")
    print()
    print("No model is configured yet.")
    print("Set ARIVAN_MODEL to a checkpoint directory, then run arivan again.")
    print("Use `arivan profile 8gb` to inspect the experimental low-memory contract.")
    return 2


def main() -> int:
    _configure_output()
    arguments = sys.argv[1:]
    if arguments[:1] == ["profile"]:
        return _profile_command(arguments[1:])
    if arguments[:1] == ["--version"]:
        print(f"arivan {__version__}")
        return 0

    arguments, selected_profile = _extract_profile(arguments)
    _apply_environment_aliases()
    if selected_profile:
        profile = get_profile(selected_profile)
        os.environ.update(profile.engine_environment(_model_arch(arguments)))
        os.environ["ARIVAN_PROFILE_STATUS"] = profile.status

    if not arguments:
        if not os.environ.get("COLI_MODEL"):
            return _welcome_without_model()
        arguments = ["chat"]

    script = _engine_script()
    if not script.exists():
        raise SystemExit(
            "Arivan engine source was not found. Install from a complete source checkout."
        )

    os.environ["ARIVAN_CLI"] = "1"
    os.environ["ARIVAN_VERSION"] = __version__
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script), *arguments]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
