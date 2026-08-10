"""Validate config.yaml structure and paths before any runner is started.

Call validate_config() immediately after load_config() in run_proteobench.py.
Returns a list of human-readable error strings; an empty list means all checks passed.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

# Import registries so enzyme/mod names are validated against the same source of truth.
# Guard the import so the validator can still be imported standalone for testing.
try:
    from runners.base import ENZYME_MAP, MOD_REGISTRY
except ImportError:
    ENZYME_MAP: dict = {}
    MOD_REGISTRY: dict = {}

VALID_FORMATS = {"raw", "mzml", "d", "wiff", "mgf"}
VALID_ACQUISITIONS = {"DDA", "DIA"}

# Fields that must be 0 < value <= 1
_FDR_FIELDS = ("fdr_psm", "fdr_peptide", "fdr_protein")
# Fields that must be positive numbers, or exactly 0 meaning "automatic"
# (see BaseRunner.auto_tolerance: each tool then enables its own mass
# calibration, or falls back to its own default tolerance).
_TOL_FIELDS = ("precursor_mass_tolerance_ppm", "fragment_mass_tolerance_ppm")


def _suggest_path(path_str: str) -> str:
    """If path_str doesn't exist but its basename is on PATH, suggest it."""
    name = Path(path_str).name
    found = shutil.which(name)
    if found:
        return f" ('{name}' found on PATH at {found} — update config.yaml to use this path)"
    return ""


def docker_setup_errors(cfg: dict) -> list[str]:
    """Check only the docker/tool-installation side of config.yaml: is docker
    itself available, and does every *enabled* tool version have its image
    pulled and its tool-specific extras (FragPipe JARs, in-container paths)
    in place? Deliberately excludes dataset/search_params checks — this is
    used by proteobench.nf to decide whether setup.nf needs to run again,
    and a missing dataset path is not something setup can fix.
    """
    errors: list[str] = []
    if not shutil.which("docker"):
        errors.append("docker is not installed or not on PATH.")
        return errors

    for tool_name, tool_cfg in (cfg.get("tools") or {}).items():
        if not isinstance(tool_cfg, dict):
            continue
        for i, ver in enumerate(tool_cfg.get("versions", [])):
            if not isinstance(ver, dict) or not ver.get("enabled", False):
                continue
            ver_prefix = f"tools > {tool_name} > id: {ver.get('id', f'index {i}')}"
            _validate_tool_docker(tool_name, ver, ver_prefix, errors)
    return errors


def incomplete_docker_tools(cfg: dict) -> list[str]:
    """Names of tools whose docker setup is incomplete (at least one enabled
    version has a missing image / FragPipe JAR / in-container path). Used by
    setup.nf to redo and re-prompt only the tools that actually need it,
    leaving already-complete tools untouched. If docker is unavailable, every
    configured tool is reported (nothing can be verified).
    """
    if not shutil.which("docker"):
        return list((cfg.get("tools") or {}).keys())

    incomplete: list[str] = []
    for tool_name, tool_cfg in (cfg.get("tools") or {}).items():
        if not isinstance(tool_cfg, dict):
            continue
        tool_errors: list[str] = []
        for i, ver in enumerate(tool_cfg.get("versions", [])):
            if not isinstance(ver, dict) or not ver.get("enabled", False):
                continue
            ver_prefix = f"tools > {tool_name} > id: {ver.get('id', f'index {i}')}"
            _validate_tool_docker(tool_name, ver, ver_prefix, tool_errors)
        if tool_errors:
            incomplete.append(tool_name)
    return incomplete


def validate_config(cfg: dict, config_path: Path) -> list[str]:
    errors: list[str] = []

    # 1. Required top-level sections
    for section in ("global", "search_params", "datasets", "tools"):
        if cfg.get(section) is None:
            errors.append(
                f"Missing or empty required section '{section}' in {config_path}. "
                f"Check that config.yaml has a '{section}:' block with content under it."
            )
    if errors:
        # Cannot safely continue without the basic structure
        return errors

    _validate_global(cfg["global"], config_path, errors)
    _validate_search_params(cfg["search_params"], config_path, errors)
    _validate_datasets(cfg["datasets"], config_path, errors)
    _validate_tools(cfg["tools"], cfg["datasets"], config_path, errors)

    return errors


# ── Section validators ────────────────────────────────────────────────────────

def _validate_global(g: dict, config_path: Path, errors: list[str]) -> None:
    output_dir = g.get("output_dir", "")
    if not output_dir:
        errors.append(
            "global.output_dir is empty. Set it to the directory where results should be written."
        )
    elif "CHANGE_ME" in str(output_dir):
        errors.append(
            f"global.output_dir still contains 'CHANGE_ME': {output_dir!r}. "
            "Replace it with a real path on your system."
        )

    if not shutil.which("docker"):
        errors.append(
            "docker is not installed or not on PATH. Every tool now runs in a docker "
            "container — install Docker before running this pipeline."
        )

    for key in ("max_parallel_jobs", "threads_per_job"):
        val = g.get(key)
        if val is not None and (not isinstance(val, int) or val < 1):
            errors.append(f"global.{key} must be a positive integer (got {val!r}).")


def _validate_search_params(sp: dict, config_path: Path, errors: list[str]) -> None:
    enzyme = sp.get("enzyme", "")
    if enzyme and ENZYME_MAP and enzyme not in ENZYME_MAP:
        known = ", ".join(sorted(ENZYME_MAP))
        errors.append(
            f"search_params.enzyme: unknown value {enzyme!r}. "
            f"Supported enzymes: {known}."
        )

    for field in _FDR_FIELDS:
        val = sp.get(field)
        if val is not None:
            try:
                fval = float(val)
                if not (0 < fval <= 1):
                    errors.append(
                        f"search_params.{field} must be between 0 and 1 "
                        f"(e.g. 0.01 for 1% FDR); got {val!r}."
                    )
            except (TypeError, ValueError):
                errors.append(f"search_params.{field} must be a number; got {val!r}.")

    for field in _TOL_FIELDS:
        val = sp.get(field)
        if val is not None:
            try:
                if float(val) < 0:
                    errors.append(
                        f"search_params.{field} must be a positive number, "
                        f"or 0 for automatic calibration; got {val!r}."
                    )
            except (TypeError, ValueError):
                errors.append(f"search_params.{field} must be a number; got {val!r}.")

    if MOD_REGISTRY:
        for mod_key in ("fixed_mods", "variable_mods"):
            for mod in sp.get(mod_key, []):
                if mod not in MOD_REGISTRY:
                    known_mods = ", ".join(sorted(MOD_REGISTRY))
                    errors.append(
                        f"search_params.{mod_key}: unknown modification {mod!r}. "
                        f"Supported modifications: {known_mods}."
                    )


def _validate_datasets(datasets: dict, config_path: Path, errors: list[str]) -> None:
    for ds_name, ds in datasets.items():
        if not isinstance(ds, dict):
            errors.append(f"datasets.{ds_name}: expected a mapping, got {type(ds).__name__}.")
            continue
        prefix = f"datasets > {ds_name}"

        # Required keys
        for key in ("path", "fasta"):
            val = ds.get(key, "")
            if not val:
                errors.append(
                    f"{prefix}: '{key}' is missing or empty. "
                    f"Set it in config.yaml under datasets > {ds_name}."
                )
            elif "CHANGE_ME" in str(val):
                errors.append(
                    f"{prefix}: '{key}' still contains 'CHANGE_ME': {val!r}. "
                    "Replace it with a real path."
                )
            elif Path(val).is_absolute() and not Path(val).exists():
                errors.append(
                    f"{prefix}: '{key}' path does not exist: {val}. "
                    "Is the data directory mounted? Check 'path:' in config.yaml."
                    if key == "path" else
                    f"{prefix}: FASTA file does not exist: {val}. "
                    f"Check 'fasta:' under datasets > {ds_name} in config.yaml."
                )

        fmt = ds.get("format", "")
        if fmt and fmt not in VALID_FORMATS:
            errors.append(
                f"{prefix}: 'format' is {fmt!r} but must be one of: "
                f"{', '.join(sorted(VALID_FORMATS))}."
            )

        acq = ds.get("acquisition", "").upper()
        if ds.get("acquisition") and acq not in VALID_ACQUISITIONS:
            errors.append(
                f"{prefix}: 'acquisition' is {ds['acquisition']!r} but must be 'DDA' or 'DIA'."
            )


def _validate_extra_args(value, where: str, errors: list[str]) -> None:
    """extra_args is passed to the tool verbatim, so the only thing worth checking
    is that it can be shell-split at all (a stray quote would otherwise fail the
    job long after the run started)."""
    if value is None or isinstance(value, (list, tuple)):
        return
    if not isinstance(value, str):
        errors.append(f"{where}: should be a string or a list of arguments, got {value!r}.")
        return
    try:
        shlex.split(value)
    except ValueError as exc:
        errors.append(f"{where}: cannot be parsed as command-line arguments ({exc}): {value!r}")


def _validate_tools(
    tools: dict, datasets: dict, config_path: Path, errors: list[str]
) -> None:
    dataset_names = set(datasets.keys())

    for tool_name, tool_cfg in tools.items():
        if not isinstance(tool_cfg, dict):
            continue
        prefix = f"tools > {tool_name}"

        # extra_args is free-form and never interpreted, but it is shell-split, so
        # catch an unbalanced quote here rather than mid-run.
        _validate_extra_args((tool_cfg.get("extra") or {}).get("extra_args"),
                             f"{prefix} > extra > extra_args", errors)

        # Cross-check dataset names
        for ds_name in tool_cfg.get("datasets", []):
            if ds_name not in dataset_names:
                errors.append(
                    f"{prefix}: dataset '{ds_name}' is listed but not defined in the "
                    "datasets section. Check for a typo or add the dataset definition."
                )

        for i, ver in enumerate(tool_cfg.get("versions", [])):
            if not isinstance(ver, dict):
                continue
            ver_id = ver.get("id", f"index {i}")
            ver_prefix = f"{prefix} > id: {ver_id}"

            if "id" not in ver:
                errors.append(f"{prefix}: version entry at index {i} is missing 'id'.")

            _validate_extra_args(ver.get("extra_args"), f"{ver_prefix} > extra_args", errors)

            enabled = ver.get("enabled")
            if enabled is not None and not isinstance(enabled, bool):
                errors.append(
                    f"{ver_prefix}: 'enabled' should be true or false (boolean), "
                    f"got {enabled!r}. Remove quotes if you used a string."
                )

            if not ver.get("enabled", False):
                continue  # skip path checks for disabled versions

            _validate_tool_docker(tool_name, ver, ver_prefix, errors)


def _docker_image_present(image: str) -> bool:
    r = subprocess.run(["docker", "image", "inspect", image], capture_output=True)
    return r.returncode == 0


def _validate_tool_docker(
    tool_name: str, ver: dict, ver_prefix: str, errors: list[str]
) -> None:
    """Check that the docker image (and any tool-specific extras) for an enabled version exist."""

    image = ver.get("image", "")
    if not image:
        errors.append(
            f"{ver_prefix}: 'image' is missing. Run: nextflow run setup.nf   to pull it."
        )
        return
    if "CHANGE_ME" in image:
        errors.append(f"{ver_prefix}: 'image' still contains 'CHANGE_ME'.")
        return
    if shutil.which("docker") and not _docker_image_present(image):
        errors.append(
            f"{ver_prefix}: docker image not pulled locally: {image}. "
            "Run: nextflow run setup.nf   to pull it."
        )

    if tool_name == "diann":
        diann_bin = ver.get("diann_bin", "")
        if not diann_bin or "CHANGE_ME" in diann_bin:
            errors.append(
                f"{ver_prefix}: 'diann_bin' is missing or still 'CHANGE_ME'. "
                "Run: nextflow run setup.nf   to detect the in-container binary path."
            )

    elif tool_name == "fragpipe":
        fragpipe_root = ver.get("fragpipe_root", "")
        if not fragpipe_root or "CHANGE_ME" in fragpipe_root:
            errors.append(
                f"{ver_prefix}: 'fragpipe_root' is missing or still 'CHANGE_ME'. "
                "Run: nextflow run setup.nf   to detect the in-container FragPipe path."
            )
        jars_dir = ver.get("jars_dir", "")
        if not jars_dir or "CHANGE_ME" in jars_dir:
            errors.append(
                f"{ver_prefix}: 'jars_dir' is missing or still 'CHANGE_ME'. "
                "Run: nextflow run setup.nf   to collect the licensed MSFragger/IonQuant/diaTracer JARs."
            )
        else:
            found = {p.name.lower() for p in Path(jars_dir).glob("*.jar")} if Path(jars_dir).is_dir() else set()
            for label, needle in (("MSFragger", "msfragger"), ("IonQuant", "ionquant"), ("diaTracer", "diatracer")):
                if not any(needle in n for n in found):
                    errors.append(
                        f"{ver_prefix}: {label} JAR not found in jars_dir ({jars_dir}). "
                        "Run: nextflow run setup.nf   to add it (or disable FragPipe)."
                    )


# ── CLI: used by proteobench.nf to decide whether setup.nf needs to run ──────

if __name__ == "__main__":
    import argparse
    import sys

    import yaml

    parser = argparse.ArgumentParser(description="Check config.yaml completeness.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--check-docker-setup", action="store_true",
        help="Only check docker/tool installation (images, FragPipe JARs), not datasets or search_params.",
    )
    parser.add_argument(
        "--list-incomplete-tools", action="store_true",
        help="Print (one per line) the tools whose docker setup is incomplete; used by setup.nf.",
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Config file not found: {args.config}")
        sys.exit(1)

    with open(args.config) as f:
        loaded_cfg = yaml.safe_load(f)

    if args.list_incomplete_tools:
        for tool in incomplete_docker_tools(loaded_cfg):
            print(tool)
        sys.exit(0)

    found_errors = docker_setup_errors(loaded_cfg) if args.check_docker_setup else validate_config(loaded_cfg, args.config)
    for e in found_errors:
        print(e)
    sys.exit(1 if found_errors else 0)
