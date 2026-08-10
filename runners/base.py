"""Base runner class and shared utilities for all search engine runners."""

from __future__ import annotations

import gc
import logging
import os
import re
import shlex
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_image_present_cache: dict[str, bool] = {}


def docker_image_present(image: str) -> bool:
    """Return True if a docker image is already pulled locally."""
    if image not in _image_present_cache:
        r = subprocess.run(["docker", "image", "inspect", image], capture_output=True)
        _image_present_cache[image] = r.returncode == 0
    return _image_present_cache[image]

# Acquisition types recognised throughout the pipeline
DDA = "DDA"
DIA = "DIA"


def infer_acquisition(dataset_name: str, dataset_cfg: dict) -> str:
    """Return 'DDA' if dataset name contains 'DDA', else 'DIA'.
    An explicit 'acquisition' key in dataset_cfg always takes precedence.
    """
    explicit = dataset_cfg.get("acquisition", "").upper()
    if explicit in (DDA, DIA):
        return explicit
    return DDA if "DDA" in dataset_name.upper() else DIA


_CONDITION_RE = re.compile(r"Condition_([A-Za-z0-9]+)")


def infer_condition(filename: str) -> str | None:
    """Extract the "Condition_X" label embedded in every ProteoBench HYE-style
    filename (e.g. "LFQ_Orbitrap_DDA_Condition_A_Sample_Alpha_01.raw" -> "A").
    Verified against every dataset under configs/*.yaml, DDA and DIA,
    Orbitrap and Astral alike. Returns None if the filename doesn't match.
    """
    m = _CONDITION_RE.search(filename)
    return m.group(1) if m else None


# Maps human-readable mod names (MaxQuant convention) to tool-specific representations.
# Keys must match what users write in config.yaml search_params.fixed_mods / variable_mods.
MOD_REGISTRY: dict[str, dict[str, Any]] = {
    "Carbamidomethyl (C)": {
        "maxquant": "Carbamidomethyl (C)",
        "metamorpheus_fixed": "Carbamidomethyl on C",
        "alphadia": "Carbamidomethyl@C",
        "diann": "Carbamidomethyl",          # DIA-NN recognises common mod names
        "sage_residues": ["C"],
        "sage_mass": 57.021464,
        "unimod_id": 4,
    },
    "Oxidation (M)": {
        "maxquant": "Oxidation (M)",
        "metamorpheus_variable": "Oxidation on M",
        "alphadia": "Oxidation@M",
        "diann": "Oxidation",
        "sage_residues": ["M"],
        "sage_mass": 15.994915,
        "unimod_id": 35,
    },
    "Phospho (STY)": {
        "maxquant": "Phospho (STY)",
        "metamorpheus_variable": "Phosphorylation on S",    # MetaMorpheus uses per-residue entries
        "alphadia": "Phospho@S;Phospho@T;Phospho@Y",
        "diann": "Phospho",
        "sage_residues": ["S", "T", "Y"],
        "sage_mass": 79.966331,
        "unimod_id": 21,
    },
    "Acetyl (Protein N-term)": {
        "maxquant": "Acetyl (Protein N-term)",
        "metamorpheus_variable": "Acetylation on X",
        "alphadia": "Acetyl@^",
        "diann": "Acetylation",
        "sage_residues": ["["],           # Sage uses "[" for N-term
        "sage_mass": 42.010565,
        "unimod_id": 1,
    },
    "Deamidation (NQ)": {
        "maxquant": "Deamidation (NQ)",
        "metamorpheus_variable": "Deamidation on N",
        "alphadia": "Deamidation@N;Deamidation@Q",
        "diann": "Deamidation",
        "sage_residues": ["N", "Q"],
        "sage_mass": 0.984016,
        "unimod_id": 7,
    },
}

ENZYME_MAP = {
    # sage_c_terminal: True = enzyme cuts C-terminally (default for trypsin, lysc, etc.)
    #                  False = enzyme cuts N-terminally (aspn)
    # metamorpheus names verified empirically against the actual smithchemwisc/metamorpheus:latest
    # image (its shipped ProteolyticDigestion/proteases.tsv, not mzLib's newer embedded resource
    # naming scheme, which this build predates). Its built-in "trypsin" has no proline-exclusion
    # rule at all (PSI-MS name is literally "Trypsin/P") -- METAMORPHEUS_CUSTOM_PROTEASES below adds
    # the missing restricted variant so results are comparable to the other tools' "trypsin".
    "trypsin":      {"diann": "K*,R*,!*P", "alphadia": "trypsin_not_p", "sage_cleave_at": "KR", "sage_restrict": "P",
                     "sage_c_terminal": True, "maxquant": "Trypsin", "metamorpheus": "trypsin (don't cleave before proline)"},
    "trypsin/p":    {"diann": "K*,R*",  "alphadia": "trypsin/p", "sage_cleave_at": "KR", "sage_restrict": None,
                     "sage_c_terminal": True, "maxquant": "Trypsin/P", "metamorpheus": "trypsin"},
    "lysc":         {"diann": "K*",    "alphadia": "lys-c",    "sage_cleave_at": "K",  "sage_restrict": None,
                     "sage_c_terminal": True, "maxquant": "LysC", "metamorpheus": "Lys-C (cleave before proline)"},
    "gluc":         {"diann": "E*",    "alphadia": "glu-c",    "sage_cleave_at": "E", "sage_restrict": None,
                     "sage_c_terminal": True, "maxquant": "GluC", "metamorpheus": "Glu-C"},
    "chymotrypsin": {"diann": "F*,Y*,W*,M*,L*,!*P", "alphadia": "chymotrypsin", "sage_cleave_at": "FWYLM", "sage_restrict": "P",
                     "sage_c_terminal": True, "maxquant": "Chymotrypsin+", "metamorpheus": "chymotrypsin (don't cleave before proline)"},
    "aspn":         {"diann": "*D",    "alphadia": "asp-n",    "sage_cleave_at": "D",  "sage_restrict": None,
                     "sage_c_terminal": False, "maxquant": "AspN", "metamorpheus": "Asp-N"},
    "argc":         {"diann": "R*",    "alphadia": "arg-c",    "sage_cleave_at": "R",  "sage_restrict": None,
                     "sage_c_terminal": True, "maxquant": "ArgC", "metamorpheus": "Arg-C"},
    "non-specific": {"diann": "**",     "alphadia": "non-specific", "sage_cleave_at": "", "sage_restrict": None,
                     "sage_c_terminal": True, "maxquant": "unspecific", "metamorpheus": "non-specific"},
    # No additional in-silico digestion: each FASTA entry is used as-is (e.g. a
    # pre-digested/peptide-level FASTA, such as the Entrapment module's entrapment
    # peptide database). diann: "" (empty --cut rule) verified empirically to return
    # each entry unsplit, as long as the entry contains only standard amino acids —
    # an ambiguous residue (X/Z/B/J/U/O) anywhere in an entry silently drops it.
    "no-cleave":    {"diann": "",         "alphadia": "no-cleave",    "sage_cleave_at": "$", "sage_restrict": None,
                     "sage_c_terminal": True, "maxquant": None, "metamorpheus": "peptidomics"},
}


@dataclass
class RunResult:
    tool: str
    version: str
    dataset: str
    success: bool
    runtime_s: float
    output_dir: Path
    error_msg: str = ""
    skipped: bool = False
    stdout_log: Path | None = None
    stderr_log: Path | None = None


class BaseRunner(ABC):
    """Abstract base for all search engine runners."""

    # Override in subclasses to restrict to one acquisition type
    SUPPORTED_ACQUISITIONS: tuple[str, ...] = (DDA, DIA)

    def __init__(
        self,
        tool_cfg: dict,
        dataset_name: str,
        dataset_cfg: dict,
        version_cfg: dict,
        global_cfg: dict,
        search_params: dict,
    ) -> None:
        self.tool_cfg = tool_cfg
        self.dataset_name = dataset_name
        self.dataset_cfg = dataset_cfg
        self.version_cfg = version_cfg
        self.global_cfg = global_cfg
        self.search_params = search_params
        self.extra = tool_cfg.get("extra", {}) or {}

        # Entrapment datasets ship a pre-digested, peptide-level FASTA. Any real
        # enzyme would cut those peptides further and invalidate the entrapment
        # results, so force 'no-cleave' regardless of the global enzyme setting.
        # This lets users keep their enzyme of choice for other datasets in the
        # same batch. Copy first so the shared search_params dict is not mutated
        # for the other (non-entrapment) datasets' runners.
        # ponytail: name-based, like infer_acquisition; add an explicit dataset
        # flag if a non-entrapment dataset ever needs forced no-cleave.
        if "entrapment" in dataset_name.lower() and self.search_params.get("enzyme") != "no-cleave":
            logger.info(
                "[%s] entrapment dataset: forcing enzyme 'no-cleave' (was %r) — "
                "FASTA is pre-digested; global enzyme ignored for this dataset.",
                dataset_name, self.search_params.get("enzyme"),
            )
            self.search_params = {**self.search_params, "enzyme": "no-cleave"}

    @property
    @abstractmethod
    def tool_name(self) -> str:
        pass

    @property
    def version_id(self) -> str:
        return self.version_cfg["id"]

    @property
    def acquisition(self) -> str:
        return infer_acquisition(self.dataset_name, self.dataset_cfg)

    def is_compatible(self) -> bool:
        """Return False if this tool/version cannot handle the dataset at all.
        Incompatible jobs are silently skipped at build time, never queued.
        Override in subclasses for version-specific checks.
        """
        return self.acquisition in self.SUPPORTED_ACQUISITIONS

    def extra_args(self) -> list[str]:
        """Free-form CLI arguments appended verbatim to the built command.

        Escape hatch for tool flags this pipeline does not model, so they need no
        schema entry of their own — e.g. DIA-NN's decoy-generation options:

            tools:
              diann:
                extra:
                  extra_args: "--dg-keep-nterm 2 --dg-min-mut 20.0"

        Also readable per version, which is appended after the tool-level value,
        for flags that only exist in some versions:

                versions:
                  - id: "2.5.1"
                    extra_args: "--dg-max-mut 60.0"

        Accepts a string (split with shell quoting rules) or an already-split
        list. Nothing is validated or translated: whatever is written here
        reaches the tool as-is, and it is the caller's job to know the flag
        exists in that version.

        ponytail: appended at the end of the command only. Tools whose CLI needs
        a flag before a positional argument are not supported; add an insertion
        point if one ever does.
        """
        def split(raw) -> list[str]:
            if not raw:
                return []
            if isinstance(raw, (list, tuple)):
                return [str(a) for a in raw]
            return shlex.split(str(raw))

        return split((self.extra or {}).get("extra_args")) + split(self.version_cfg.get("extra_args"))

    def auto_tolerance(self, which: str) -> bool:
        """True when search_params sets this mass tolerance to 0, which means
        "let the tool work it out" rather than a literal 0 ppm window.

        0 is never a usable tolerance, so it is an unambiguous opt-in marker.
        What each runner does with it depends on what the tool actually offers:

          * a real automatic mass calibration → switch it on and stop pinning
            the tolerance (DIA-NN omits --mass-acc/--mass-acc-ms1; MSFragger
            gets calibrate_mass=2; AlphaDIA drops its target_ms*_tolerance)
          * no such switch → fall back to the tool's own default tolerance
            (MaxQuant, MetaMorpheus, Sage)

        'which' is "precursor" or "fragment". The two are independent: only the
        one set to 0 is made automatic.
        """
        val = self.search_params.get(f"{which}_mass_tolerance_ppm")
        try:
            return val is not None and float(val) == 0
        except (TypeError, ValueError):
            return False

    def requires_mzml(self) -> bool:
        """Override to return True when this tool/version cannot read the
        dataset's native format and needs mzML instead (e.g. Sage always;
        DIA-NN < 2.0 for Thermo .raw). mzML is not a separate dataset entry:
        get_input_files() then looks for it in the "<dataset>_mzml" sibling
        directory next to the dataset's configured path.
        """
        return False

    def _mzml_search_dirs(self) -> list[Path]:
        """Candidate directories to search for mzML when requires_mzml() is True
        and the dataset isn't already configured as mzML.

        Convention: if the dataset path is /path/DATASET, prefer a sibling
        /path/DATASET_mzml directory (populated by setup.nf from a downloaded
        dataset's mzml/ subfolder, or placed there by hand).
        """
        dataset_path = Path(self.dataset_cfg["path"])
        redirected = dataset_path.parent / f"{dataset_path.name}_mzml"
        if redirected == dataset_path:
            return [dataset_path]
        return [redirected, dataset_path]

    def _find_mzml_files(self, dataset_dir: Path) -> list[Path]:
        return sorted(dataset_dir.glob("*.mzML")) or sorted(dataset_dir.glob("*.mzml"))

    def _no_cleave_fasta_sanity_check(self, fasta: Path) -> list[str]:
        """'no-cleave' searches each FASTA entry as-is (see the entrapment
        override in __init__ above), so entries longer than max_peptide_length
        can never be matched. Pairing 'no-cleave' with a normal, full-length
        protein FASTA silently collapses the candidate pool to near-zero
        instead of raising an error, and the run only fails hours later
        downstream (near-0 identifications -> spectral library step crashes on
        a file that was never written). Catch that mismatch up front instead.
        """
        max_len = self.search_params.get("max_peptide_length", 30)
        sample_size = 500
        lengths: list[int] = []
        seq: list[str] = []
        with open(fasta) as fh:
            for line in fh:
                if line.startswith(">"):
                    if seq:
                        lengths.append(sum(len(s) for s in seq))
                        seq = []
                    if len(lengths) >= sample_size:
                        break
                else:
                    seq.append(line.strip())
            else:
                if seq:
                    lengths.append(sum(len(s) for s in seq))

        if not lengths:
            return []
        too_long = sum(1 for length in lengths if length > max_len)
        if too_long / len(lengths) > 0.5:
            return [
                f"enzyme is 'no-cleave' but {too_long}/{len(lengths)} sampled entries in "
                f"{fasta} exceed max_peptide_length ({max_len}) — 'no-cleave' searches each "
                "entry whole, so this looks like a normal full-length-protein FASTA rather "
                "than a pre-digested peptide-level one. Point 'fasta:' at the pre-digested "
                "FASTA instead (or set search_params.enzyme explicitly if this is intentional)."
            ]
        return []

    def preflight_check(self) -> list[str]:
        """Return list of error strings; empty list means all checks passed.
        Called only on jobs that passed is_compatible().
        """
        errors: list[str] = []

        dataset_path = Path(self.dataset_cfg["path"])
        if not dataset_path.exists():
            errors.append(
                f"Dataset path not found: {dataset_path}. "
                f"Check 'path:' under datasets > {self.dataset_name} in config.yaml "
                "(is the data directory mounted?)"
            )
        fasta = Path(self.dataset_cfg["fasta"])
        if not fasta.exists():
            errors.append(
                f"FASTA file not found: {fasta}. "
                f"Check 'fasta:' under datasets > {self.dataset_name} in config.yaml."
            )
        elif self.search_params.get("enzyme") == "no-cleave":
            errors += self._no_cleave_fasta_sanity_check(fasta)
        if not self.get_input_files():
            fmt = self.dataset_cfg.get("format", "?")
            if self.requires_mzml() and fmt != "mzml":
                candidates = ", ".join(str(p) for p in self._mzml_search_dirs())
                errors.append(
                    f"{self.tool_name} requires mzML input; dataset format is {fmt!r} and "
                    f"no mzML files found in candidate folder(s): {candidates}. "
                    "Convert RAW/.d files to mzML first (e.g. with ThermoRawFileParser or msconvert)."
                )
            else:
                errors.append(
                    f"No {fmt!r} files found in {dataset_path}. "
                    "Verify that MS data files are present and that 'format:' matches "
                    "the actual file type (raw / mzml / d / wiff / mgf)."
                )
        return errors

    def get_input_files(self) -> list[Path]:
        """Return list of MS data file paths based on dataset format.

        When requires_mzml() is True and the dataset isn't already mzML, redirect
        to the "<dataset>_mzml" sibling directory instead (see _mzml_search_dirs).
        """
        if self.requires_mzml() and self.dataset_cfg.get("format") != "mzml":
            for d in self._mzml_search_dirs():
                if d.exists():
                    mzmls = self._find_mzml_files(d)
                    if mzmls:
                        return mzmls
            # Tool cannot use the dataset's native format at all — do not fall
            # through to it below; preflight_check() reports this clearly.
            return []

        dataset_path = Path(self.dataset_cfg["path"])
        fmt = self.dataset_cfg["format"]
        if fmt == "raw":
            return sorted(dataset_path.glob("*.raw"))
        if fmt == "mzml":
            return sorted(dataset_path.glob("*.mzML")) or sorted(dataset_path.glob("*.mzml"))
        if fmt == "d":
            # Bruker .d directories
            return sorted(p for p in dataset_path.iterdir() if p.is_dir() and p.suffix == ".d")
        if fmt == "wiff":
            # SCIEX: .wiff (older) and .wiff2 (ZenoTOF); companion .wiff.scan
            # files are read by the tool alongside and are not listed here.
            return sorted(dataset_path.glob("*.wiff")) + sorted(dataset_path.glob("*.wiff2"))
        if fmt == "mgf":
            return sorted(dataset_path.glob("*.mgf")) + sorted(dataset_path.glob("*.mgf.gz"))
        return []

    @abstractmethod
    def map_params(self) -> dict:
        """Translate self.search_params to tool-specific parameter dict."""

    @abstractmethod
    def build_command(self, input_files: list[Path], fasta: Path, output_dir: Path) -> list[str]:
        pass

    def subprocess_stdin(self) -> bytes | None:
        """Override to supply bytes written to the subprocess stdin."""
        return None

    def extra_env(self) -> dict[str, str]:
        """Override to inject extra environment variables into the subprocess."""
        return {}

    # ── Docker execution helpers ──────────────────────────────────────────
    # Every tool now runs inside its own docker image (pulled by setup.nf).
    # We bind-mount every host path a job touches at the *same* absolute path
    # inside the container, so tool-specific config files (mqpar.xml,
    # sage_config.json, ...) can keep using host paths unmodified.

    def docker_image(self) -> str:
        image = self.version_cfg.get("image")
        if not image:
            raise RuntimeError(
                f"tools > {self.tool_name} > versions > id: {self.version_id} has no 'image:' set. "
                "Run: nextflow run setup.nf   to pull the required docker image."
            )
        return image

    def _docker_host_dirs(self) -> list[str]:
        """Host directories referenced by this job (dataset, fasta(s), output).

        Includes each input file's own parent dir, not just the configured
        dataset path — get_input_files() may redirect to a "<dataset>_mzml"
        sibling directory (see requires_mzml()), which otherwise would never
        get bind-mounted and the tool would fail to open its input files.
        """
        dirs = {
            str(Path(self.dataset_cfg["path"]).resolve()),
            str(Path(self.dataset_cfg["fasta"]).resolve().parent),
            str(self._output_dir_path().resolve()),
        }
        if self.dataset_cfg.get("fasta_decoy"):
            dirs.add(str(Path(self.dataset_cfg["fasta_decoy"]).resolve().parent))
        for f in self.get_input_files():
            dirs.add(str(f.resolve().parent))
        return sorted(dirs)

    def docker_run_prefix(
        self,
        image: str,
        extra_mounts: list[tuple[str, str]] | None = None,
        env: dict[str, str] | None = None,
        gpu: bool = False,
    ) -> list[str]:
        """Build the `docker run ...` prefix; append the in-container command after this."""
        cmd = ["docker", "run", "--rm", "-i", "-u", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp"]
        if gpu:
            cmd += ["--gpus", "all"]
        for d in self._docker_host_dirs():
            cmd += ["-v", f"{d}:{d}"]
        for host, container in extra_mounts or []:
            cmd += ["-v", f"{host}:{container}:ro"]
        for k, v in (env or {}).items():
            cmd += ["-e", f"{k}={v}"]
        cmd.append(image)
        return cmd

    def docker_preflight(self) -> list[str]:
        """Standard preflight check: docker CLI present, image pulled locally."""
        errors: list[str] = []
        import shutil as _shutil
        if not _shutil.which("docker"):
            errors.append("docker is not installed or not on PATH. Install Docker before running this pipeline.")
            return errors
        image = self.version_cfg.get("image", "")
        if not image:
            errors.append(
                f"tools > {self.tool_name} > versions > id: {self.version_id} has no 'image:' set in config.yaml."
            )
        elif not docker_image_present(image):
            errors.append(
                f"Docker image not found locally: {image}. Run: nextflow run setup.nf   to pull it."
            )
        return errors

    def pre_run_hook(self, input_files: list[Path]) -> None:
        """Called once just before the subprocess is launched. Override for pre-run cleanup."""

    def post_run_hook(
        self, input_files: list[Path], output_dir: Path, success: bool, error_msg: str
    ) -> tuple[bool, str]:
        """Called after the subprocess exits. Return (success, error_msg), possibly overriding them."""
        return success, error_msg

    def _output_dir_path(self) -> Path:
        return (
            Path(self.global_cfg["output_dir"])
            / self.dataset_name
            / f"{self.tool_name}_v{self.version_id}"
        )

    def make_output_dir(self) -> Path:
        out = self._output_dir_path()
        out.mkdir(parents=True, exist_ok=True)
        return out

    def run(self) -> RunResult:
        output_dir = self._output_dir_path()
        done_marker = output_dir / ".done"
        overwrite = self.global_cfg.get("overwrite", False)

        if done_marker.exists() and not overwrite:
            logger.info("[SKIP] %s v%s / %s — previous successful run found; set overwrite: true to rerun",
                        self.tool_name, self.version_id, self.dataset_name)
            return RunResult(
                tool=self.tool_name, version=self.version_id, dataset=self.dataset_name,
                success=True, runtime_s=0.0, output_dir=output_dir,
                skipped=True,
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        stdout_log = output_dir / "stdout.log"
        stderr_log = output_dir / "stderr.log"
        input_files = self.get_input_files()
        fasta = Path(self.dataset_cfg["fasta"])

        self.pre_run_hook(input_files)

        try:
            cmd = list(self.build_command(input_files, fasta, output_dir))
            # Free-form passthrough flags, last so they can override anything the
            # runner already set (most CLIs let a later occurrence win). Inside the
            # try because extra_args() raises on unbalanced quotes in user input.
            passthrough = self.extra_args()
        except Exception as exc:
            return RunResult(
                tool=self.tool_name, version=self.version_id, dataset=self.dataset_name,
                success=False, runtime_s=0.0, output_dir=output_dir,
                error_msg=f"build_command failed: {exc}",
            )

        if passthrough:
            logger.info("[%s v%s / %s] extra_args: %s", self.tool_name, self.version_id,
                        self.dataset_name, shlex.join(passthrough))
            cmd += passthrough

        logger.info("[%s v%s / %s] starting: %s", self.tool_name, self.version_id, self.dataset_name,
                    shlex.join(str(c) for c in cmd))
        t0 = time.monotonic()
        try:
            env = {**os.environ, **self.extra_env()}
            with open(stdout_log, "w") as fout, open(stderr_log, "w") as ferr:
                proc = subprocess.run(
                    [str(c) for c in cmd],
                    stdout=fout,
                    stderr=ferr,
                    input=self.subprocess_stdin(),
                    env=env,
                    check=False,
                )
            runtime = time.monotonic() - t0
            success = proc.returncode == 0
            error_msg = "" if success else f"exit code {proc.returncode} — check log: {stderr_log}"
        except Exception as exc:
            runtime = time.monotonic() - t0
            success = False
            error_msg = str(exc)
        finally:
            gc.collect()

        success, error_msg = self.post_run_hook(input_files, output_dir, success, error_msg)

        if success:
            done_marker.write_text(datetime.now().isoformat() + "\n")

        level = logging.INFO if success else logging.ERROR
        logger.log(level, "[%s v%s / %s] finished in %.1fs success=%s%s",
                   self.tool_name, self.version_id, self.dataset_name,
                   runtime, success, f" ({error_msg})" if error_msg else "")
        return RunResult(
            tool=self.tool_name, version=self.version_id, dataset=self.dataset_name,
            success=success, runtime_s=runtime, output_dir=output_dir,
            error_msg=error_msg, stdout_log=stdout_log, stderr_log=stderr_log,
        )
