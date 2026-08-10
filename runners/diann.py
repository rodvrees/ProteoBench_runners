"""DIA-NN runner.

Runs inside the quantms DIA-NN docker images (set up by setup.nf): 1.8.1 is
pulled from biocontainers/diann (public), while 2.x images are built locally
from the bigbio/quantms-containers recipes, since the DIA-NN license forbids
redistributing them. Any number of versions can be configured side by side;
the in-container binary path is recorded per version in 'diann_bin'.

Supported search_params keys (others ignored):
  fdr_psm, fdr_protein, match_between_runs, normalize,
  precursor_mass_tolerance_ppm, fragment_mass_tolerance_ppm,
  missed_cleavages, min_peptide_length, max_peptide_length,
  fixed_mods, variable_mods, max_mods_per_peptide,
  min_charge, max_charge, precursor_mz_range, fragment_mz_range

Not mapped (DIA-NN has no separate peptide-level FDR):
  fdr_peptide
"""

from __future__ import annotations

import logging
import signal
import subprocess
from pathlib import Path

from .base import DDA, DIA, ENZYME_MAP, BaseRunner

logger = logging.getLogger(__name__)

# Cache smoke-test results per image+binary so we don't re-run for every dataset.
_binary_smoke_cache: dict[str, str | None] = {}  # "image|binary" → error string or None


def _smoke_test_binary(image: str, binary: str) -> str | None:
    """Return an error string if the binary crashes on startup, else None."""
    cache_key = f"{image}|{binary}"
    if cache_key in _binary_smoke_cache:
        return _binary_smoke_cache[cache_key]
    try:
        r = subprocess.run(["docker", "run", "--rm", image, binary], capture_output=True, timeout=30)
        if r.returncode < 0:
            sig = -r.returncode
            try:
                sig_name = signal.Signals(sig).name
            except ValueError:
                sig_name = str(sig)
            err = (
                f"DIA-NN binary crashes on startup (signal {sig_name}); "
                "likely a library or CPU-instruction incompatibility on this system."
            )
        else:
            err = None
    except subprocess.TimeoutExpired:
        err = None  # binary is running — not a startup crash
    except Exception as exc:
        err = f"Could not smoke-test DIA-NN binary: {exc}"
    _binary_smoke_cache[cache_key] = err
    return err

# Carbamidomethyl (C) as a fixed mod has a built-in shortcut flag; DIA-NN requires
# explicit mass+residue specs for everything else via --fixed-mod / --var-mod.
_FIXED_MOD_SHORTCUT: dict[str, str] = {
    "Carbamidomethyl (C)": "--unimod4",
}

# UniMod spec strings for --fixed-mod / --var-mod.
# Multi-residue mods are stored as a list (one entry per residue).
_MOD_SPECS: dict[str, list[str]] = {
    "Carbamidomethyl (C)": ["UniMod:4,57.021464,C"],
    "Oxidation (M)":       ["UniMod:35,15.994915,M"],
    "Phospho (STY)":       ["UniMod:21,79.966331,S",
                             "UniMod:21,79.966331,T",
                             "UniMod:21,79.966331,Y"],
    "Acetyl (Protein N-term)": ["UniMod:1,42.010565,*n"],
    "Deamidation (NQ)":    ["UniMod:7,0.984016,N",
                             "UniMod:7,0.984016,Q"],
}


class DIANNRunner(BaseRunner):
    SUPPORTED_ACQUISITIONS = (DDA, DIA)

    def _major_version(self) -> int | None:
        """Best-effort parse of major version from version_id (e.g. '2.5.0' -> 2)."""
        try:
            major_str = str(self.version_id).split(".", 1)[0]
            return int(major_str)
        except Exception:
            return None

    def requires_mzml(self) -> bool:
        """DIA-NN < 2 cannot process Thermo .raw directly on Linux; use mzML instead."""
        major = self._major_version()
        return major is not None and major < 2

    @property
    def tool_name(self) -> str:
        return "diann"

    def is_compatible(self) -> bool:
        if not super().is_compatible():
            return False
        if self.acquisition == DDA and not self.version_cfg.get("supports_dda", False):
            return False
        return True

    def preflight_check(self) -> list[str]:
        errors = super().preflight_check()
        errors += self.docker_preflight()
        if errors:
            return errors

        image = self.docker_image()
        binary = self.version_cfg.get("diann_bin", "")
        if not binary:
            errors.append(
                f"'diann_bin' is not set under tools > diann > versions > id: {self.version_id} in config.yaml. "
                "Run: nextflow run setup.nf   to detect it automatically."
            )
            return errors
        smoke_err = _smoke_test_binary(image, binary)
        if smoke_err:
            errors.append(smoke_err)
        return errors

    def map_params(self) -> dict:
        sp = self.search_params
        enzyme_key = sp.get("enzyme", "trypsin")
        enzyme_info = ENZYME_MAP.get(enzyme_key, ENZYME_MAP["trypsin"])
        precursor_mz = sp.get("precursor_mz_range", [400.0, 1200.0])
        fragment_mz  = sp.get("fragment_mz_range",  [200.0, 2000.0])

        return {
            "enzyme":           enzyme_info["diann"],
            "missed_cleavages": sp.get("missed_cleavages", 2),
            "min_pep_length":   sp.get("min_peptide_length", 7),
            "max_pep_length":   sp.get("max_peptide_length", 30),
            "mass_acc":         sp.get("fragment_mass_tolerance_ppm", 20),
            "mass_acc_ms1":     sp.get("precursor_mass_tolerance_ppm", 20),
            "fdr_psm":          sp.get("fdr_psm", 0.01),
            "fdr_protein":      sp.get("fdr_protein", 0.01),
            "fixed_mods":       list(sp.get("fixed_mods", [])),
            "var_mods":         list(sp.get("variable_mods", [])),
            "max_mods":         sp.get("max_mods_per_peptide", 3),
            "min_charge":       sp.get("min_charge", 2),
            "max_charge":       sp.get("max_charge", 4),
            "min_pr_mz":        precursor_mz[0],
            "max_pr_mz":        precursor_mz[1],
            "min_fr_mz":        fragment_mz[0],
            "max_fr_mz":        fragment_mz[1],
            "mbr":              sp.get("match_between_runs", False),
            "normalize":        sp.get("normalize", True),
        }

    def _fixed_mod_args(self, mod_names: list[str]) -> list[str]:
        """Return CLI args for fixed modifications."""
        args: list[str] = []
        for m in mod_names:
            if m in _FIXED_MOD_SHORTCUT:
                args.append(_FIXED_MOD_SHORTCUT[m])
            else:
                for spec in _MOD_SPECS.get(m, []):
                    args += ["--fixed-mod", spec]
        return args

    def _var_mod_args(self, mod_names: list[str]) -> list[str]:
        """Return CLI args for variable modifications."""
        args: list[str] = []
        for m in mod_names:
            for spec in _MOD_SPECS.get(m, []):
                args += ["--var-mod", spec]
        return args

    def build_command(self, input_files: list[Path], fasta: Path, output_dir: Path) -> list[str]:
        binary  = self.version_cfg.get("diann_bin", "diann")
        p       = self.map_params()
        threads = self.global_cfg.get("threads_per_job", 16)

        cmd = self.docker_run_prefix(self.docker_image())
        cmd.append(binary)

        for f in input_files:
            cmd += ["--f", str(f)]

        cmd += [
            "--fasta", str(fasta),
            "--out",   str(output_dir / "report.tsv"),
            "--temp",  str(output_dir),
            "--threads",          str(threads),
            "--missed-cleavages", str(p["missed_cleavages"]),
            "--min-pep-len",      str(p["min_pep_length"]),
            "--max-pep-len",      str(p["max_pep_length"]),
            "--qvalue",           str(p["fdr_psm"]),
            "--protein-qvalue",   str(p["fdr_protein"]),
            "--min-pr-charge",    str(p["min_charge"]),
            "--max-pr-charge",    str(p["max_charge"]),
            "--min-pr-mz",        str(int(p["min_pr_mz"])),
            "--max-pr-mz",        str(int(p["max_pr_mz"])),
            "--min-fr-mz",        str(int(p["min_fr_mz"])),
            "--max-fr-mz",        str(int(p["max_fr_mz"])),
            "--cut",              str(p["enzyme"]),
            "--gen-spec-lib",
            "--predictor"
        ]

        # Tolerance 0 = automatic: DIA-NN determines mass accuracy itself from the
        # first run when the flag is absent, so leave it off rather than passing 0.
        if not self.auto_tolerance("fragment"):
            cmd += ["--mass-acc", str(p["mass_acc"])]
        if not self.auto_tolerance("precursor"):
            cmd += ["--mass-acc-ms1", str(p["mass_acc_ms1"])]

        cmd += self._fixed_mod_args(p["fixed_mods"])
        cmd += self._var_mod_args(p["var_mods"])
        cmd += ["--var-mods", str(p["max_mods"])]

        library = (self.extra or {}).get("library", "")
        if library:
            cmd += ["--lib", str(library)]
        else:
            cmd.append("--gen-spec-lib")

        cmd.append("--fasta-search")

        if self.acquisition == DDA:
            cmd.append("--dda")

        if p["mbr"]:
            cmd.append("--reanalyse")

        if not p["normalize"]:
            cmd.append("--no-norm")

        return cmd
