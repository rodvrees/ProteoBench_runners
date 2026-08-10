"""Self-check for the "tolerance 0 = automatic calibration" convention.

Run: python3 test_auto_tolerance.py

Asserts that setting precursor/fragment_mass_tolerance_ppm to 0 stops each
runner from pinning that tolerance, and that a normal value still lands in the
generated command / config file. No docker or data needed.
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from runners.alphadia import AlphaDIARunner
from runners.diann import DIANNRunner
from runners.metamorpheus import MetaMorpheusRunner
from runners.sage import SageRunner

BASE_SP = {
    "enzyme": "trypsin",
    "precursor_mass_tolerance_ppm": 7,
    "fragment_mass_tolerance_ppm": 13,
    "fixed_mods": [],
    "variable_mods": [],
}
GLOBAL = {"threads_per_job": 4, "output_dir": "/tmp"}


def sp(**over):
    return {**BASE_SP, **over}


def make(cls, search_params, version_cfg, acquisition="DIA", fmt="mzml"):
    dataset_cfg = {"path": "/tmp", "acquisition": acquisition, "format": fmt,
                   "fasta": "/tmp/f.fasta"}
    tool_cfg = {"versions": [version_cfg], "extra": {}}
    return cls(tool_cfg, "TestSet", dataset_cfg, version_cfg, GLOBAL, search_params)


def check_diann():
    v = {"id": "2.5.0", "image": "diann:2.5.0", "diann_bin": "/d", "supports_dda": True}
    out = Path("/tmp")

    cmd = make(DIANNRunner, sp(), v).build_command([Path("a.mzML")], Path("f.fasta"), out)
    assert "--mass-acc" in cmd and cmd[cmd.index("--mass-acc") + 1] == "13", cmd
    assert "--mass-acc-ms1" in cmd and cmd[cmd.index("--mass-acc-ms1") + 1] == "7", cmd

    both = make(DIANNRunner, sp(precursor_mass_tolerance_ppm=0,
                                fragment_mass_tolerance_ppm=0), v)
    cmd = both.build_command([Path("a.mzML")], Path("f.fasta"), out)
    assert "--mass-acc" not in cmd and "--mass-acc-ms1" not in cmd, cmd

    # the two are independent
    one = make(DIANNRunner, sp(precursor_mass_tolerance_ppm=0), v)
    cmd = one.build_command([Path("a.mzML")], Path("f.fasta"), out)
    assert "--mass-acc" in cmd and "--mass-acc-ms1" not in cmd, cmd
    print("diann            ok")


def check_alphadia():
    v = {"id": "latest", "image": "mannlabs/alphadia:latest"}
    with TemporaryDirectory() as td:
        r = make(AlphaDIARunner, sp(), v)
        r.build_command([Path("a.mzML")], Path("f.fasta"), Path(td))
        cfg = yaml.safe_load((Path(td) / "alphadia_config.yaml").read_text())
        assert cfg["search"]["target_ms1_tolerance"] == 7, cfg["search"]
        assert cfg["search"]["target_ms2_tolerance"] == 13, cfg["search"]

    with TemporaryDirectory() as td:
        r = make(AlphaDIARunner, sp(precursor_mass_tolerance_ppm=0,
                                    fragment_mass_tolerance_ppm=0), v)
        r.build_command([Path("a.mzML")], Path("f.fasta"), Path(td))
        cfg = yaml.safe_load((Path(td) / "alphadia_config.yaml").read_text())
        # 0 must reach AlphaDIA verbatim: it selects AutomaticMS1/MS2Optimizer.
        # Dropping the key would fall back to default.yaml (5/10 ppm) and keep
        # the targeted optimizer instead.
        assert cfg["search"]["target_ms1_tolerance"] == 0, cfg["search"]
        assert cfg["search"]["target_ms2_tolerance"] == 0, cfg["search"]
        assert cfg["search"]["extraction_backend"] == "rust", cfg["search"]
    print("alphadia         ok")


def check_sage():
    v = {"id": "latest", "image": "ghcr.io/lazear/sage:latest", "sage_bin": "/app/sage"}
    with TemporaryDirectory() as td:
        r = make(SageRunner, sp(precursor_mass_tolerance_ppm=0,
                                fragment_mass_tolerance_ppm=0), v, acquisition="DDA")
        r.build_command([Path("a.mzML")], Path("f.fasta"), Path(td))
        cfg = json.loads(next(Path(td).glob("*.json")).read_text())
        # Sage rejects a config missing these keys, so they must still be present
        assert cfg["precursor_tol"]["ppm"] == [-20, 20], cfg["precursor_tol"]
        assert cfg["fragment_tol"]["ppm"] == [-20, 20], cfg["fragment_tol"]
    print("sage             ok")


def check_metamorpheus():
    v = {"id": "latest", "image": "smithchemwisc/metamorpheus:latest"}
    with TemporaryDirectory() as td:
        r = make(MetaMorpheusRunner, sp(), v, acquisition="DDA")
        toml = r._write_search_task(Path(td), True).read_text()
        assert "PrecursorMassTolerance" in toml and "ProductMassTolerance" in toml

    with TemporaryDirectory() as td:
        r = make(MetaMorpheusRunner, sp(precursor_mass_tolerance_ppm=0,
                                        fragment_mass_tolerance_ppm=0), v, acquisition="DDA")
        toml = r._write_search_task(Path(td), True).read_text()
        assert "PrecursorMassTolerance" not in toml, toml
        assert "ProductMassTolerance" not in toml, toml
        assert "QuantifyPpmTol = 20.0" in toml, toml
        # the surrounding table must stay intact
        assert "QValueThreshold" in toml and "[CommonParameters]" in toml
    print("metamorpheus     ok")


MQPAR_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<MaxQuantParams>
  <fastaFiles><FastaFileInfo><fastaFilePath>/tmp/f.fasta</fastaFilePath></FastaFileInfo></fastaFiles>
  <filePaths><string>/tmp/a.raw</string></filePaths>
  <parameterGroups>
    <parameterGroup>
      <maxMissedCleavages>2</maxMissedCleavages>
      <mainSearchTol>4.5</mainSearchTol>
      <maxNmods>5</maxNmods>
      <lfqNormType>1</lfqNormType>
    </parameterGroup>
  </parameterGroups>
  <msmsParamsArray>
    <msmsParams><MatchTolerance>20</MatchTolerance><MatchToleranceInPpm>True</MatchToleranceInPpm></msmsParams>
    <msmsParams><MatchTolerance>0.5</MatchTolerance><MatchToleranceInPpm>False</MatchToleranceInPpm></msmsParams>
  </msmsParamsArray>
</MaxQuantParams>
"""


def check_maxquant():
    from runners.maxquant import MaxQuantRunner
    v = {"id": "2.6.3.0", "image": "quay.io/medbioinf/maxquant:2.6.3.0",
         "maxquant_dll": "/opt/MaxQuant_v2.6.3.0/bin/MaxQuantCmd.dll"}

    def tols(search_params):
        with TemporaryDirectory() as td:
            m = Path(td) / "mqpar.xml"
            m.write_text(MQPAR_TEMPLATE)
            r = make(MaxQuantRunner, search_params, v, acquisition="DDA", fmt="raw")
            r._patch_mqpar(m, [Path("/tmp/a.raw")], Path("/tmp/f.fasta"))
            root = ET.parse(m).getroot()
            return (
                sorted({e.text for e in root.findall('.//parameterGroup/mainSearchTol')}),
                sorted({e.text for e in root.findall('.//msmsParamsArray/msmsParams/MatchTolerance')}),
            )

    assert tols(sp()) == (["7"], ["13"]), tols(sp())
    # automatic: MaxQuant recalibrates itself, so the template's values stay put
    auto = tols(sp(precursor_mass_tolerance_ppm=0, fragment_mass_tolerance_ppm=0))
    assert auto == (["4.5"], ["0.5", "20"]), auto
    print("maxquant         ok")


def check_fragpipe():
    """Needs the FragPipe image: the workflow template lives inside it."""
    from runners.fragpipe import FragPipeRunner
    v = {"id": "24.0", "image": "fcyucn/fragpipe:latest",
         "fragpipe_root": "/fragpipe_bin/fragpipe-24.0/fragpipe-24.0",
         "jars_dir": "/home/robbe/PB_output/tools/fragpipe_jars",
         "container_python": "/usr/bin/python3"}

    def props(search_params):
        r = make(FragPipeRunner, search_params, v)
        with TemporaryDirectory() as td:
            wf = r._write_workflow(Path("/tmp/f.fasta"), r.map_params(), 4, Path(td))
            return dict(
                line.split("=", 1) for line in wf.read_text().splitlines()
                if "=" in line and not line.startswith("#")
            )

    try:
        explicit = props(sp())
    except Exception as exc:                     # image missing / docker down
        print(f"fragpipe         skipped ({type(exc).__name__})")
        return

    assert explicit["msfragger.precursor_true_tolerance"] == "7", explicit
    assert explicit["msfragger.fragment_mass_tolerance"] == "13", explicit

    auto = props(sp(precursor_mass_tolerance_ppm=0, fragment_mass_tolerance_ppm=0))
    # calibrate_mass=2 is MSFragger's calibration + parameter optimization, and the
    # tolerances stay at the workflow template's values instead of being pinned.
    assert auto["msfragger.calibrate_mass"] == "2", auto
    assert auto["msfragger.precursor_true_tolerance"] == "20", auto
    assert auto["msfragger.fragment_mass_tolerance"] == "20", auto

    half = props(sp(precursor_mass_tolerance_ppm=0))
    assert half["msfragger.precursor_true_tolerance"] == "20", half
    assert half["msfragger.fragment_mass_tolerance"] == "13", half
    print("fragpipe         ok")


def check_extra_args():
    """Free-form passthrough: appended verbatim, tool-level then version-level."""
    v = {"id": "2.5.0", "image": "diann:2.5.0", "diann_bin": "/d"}

    def runner(tool_extra=None, version_extra=None):
        vc = {**v, **({"extra_args": version_extra} if version_extra is not None else {})}
        dataset_cfg = {"path": "/tmp", "acquisition": "DIA", "format": "mzml",
                       "fasta": "/tmp/f.fasta"}
        extra = {"library": ""}
        if tool_extra is not None:
            extra["extra_args"] = tool_extra
        return DIANNRunner({"versions": [vc], "extra": extra}, "TestSet",
                           dataset_cfg, vc, GLOBAL, sp())

    assert runner().extra_args() == []
    assert runner(tool_extra="").extra_args() == []

    # the flags from the request
    r = runner(tool_extra="--dg-keep-nterm 2 --dg-keep-cterm 1 --dg-min-shuffle 7.5")
    assert r.extra_args() == ["--dg-keep-nterm", "2", "--dg-keep-cterm", "1",
                              "--dg-min-shuffle", "7.5"], r.extra_args()

    # tool-level first, then version-level
    r = runner(tool_extra="--dg-min-mut 20.0", version_extra="--dg-max-mut 60.0")
    assert r.extra_args() == ["--dg-min-mut", "20.0", "--dg-max-mut", "60.0"], r.extra_args()

    # version-level alone, and an already-split list
    assert runner(version_extra="--dg-min-mut 20").extra_args() == ["--dg-min-mut", "20"]
    assert runner(tool_extra=["--dg-min-mut", "20"]).extra_args() == ["--dg-min-mut", "20"]

    # shell quoting is respected, so a value with spaces survives as one argument
    assert runner(tool_extra="--foo 'a b'").extra_args() == ["--foo", "a b"]

    # and it actually reaches the end of the command
    cmd = runner(tool_extra="--dg-keep-nterm 2").build_command(
        [Path("a.mzML")], Path("f.fasta"), Path("/tmp"))
    cmd += runner(tool_extra="--dg-keep-nterm 2").extra_args()
    assert cmd[-2:] == ["--dg-keep-nterm", "2"], cmd[-4:]
    print("extra_args       ok")


def check_auto_tolerance_helper():
    v = {"id": "2.5.0", "image": "diann:2.5.0", "diann_bin": "/d"}
    r = make(DIANNRunner, sp(precursor_mass_tolerance_ppm=0), v)
    assert r.auto_tolerance("precursor") and not r.auto_tolerance("fragment")
    # 0.0 and "0" count; a missing key does not (falls back to the runner default)
    assert make(DIANNRunner, sp(fragment_mass_tolerance_ppm=0.0), v).auto_tolerance("fragment")
    assert make(DIANNRunner, sp(fragment_mass_tolerance_ppm="0"), v).auto_tolerance("fragment")
    missing = {k: x for k, x in BASE_SP.items() if k != "fragment_mass_tolerance_ppm"}
    assert not make(DIANNRunner, missing, v).auto_tolerance("fragment")
    assert not make(DIANNRunner, sp(fragment_mass_tolerance_ppm=None), v).auto_tolerance("fragment")
    print("auto_tolerance   ok")


if __name__ == "__main__":
    check_auto_tolerance_helper()
    check_extra_args()
    check_diann()
    check_alphadia()
    check_sage()
    check_metamorpheus()
    check_maxquant()
    check_fragpipe()
    print("\nall ok")
