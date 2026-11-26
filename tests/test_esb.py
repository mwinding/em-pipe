"""pipeline/esb.py: every raw file matched to its EsB file, the naming detected, missing sets by day."""

import numpy as np

import synth
from pipeline import esb


def touch(raw, *names, shape=(2, 8, 8)):
    for name in names:
        (raw / name).parent.mkdir(parents=True, exist_ok=True)
        synth.write_imagej(raw / name, np.zeros(shape, np.uint16), [f"s{i}" for i in range(shape[0])])


def raws(days, tiles=("0-0", "0-1"), folder=""):
    return [f"{folder}M{m:02d}_D{d:02d}_tile{t}.tif" for m, d in days for t in tiles]


def as_esb(files, token="_EsB", folder="Backscatter/", flat=False):
    out = []
    for f in files:
        d, _, name = f.rpartition("/")
        out.append(("" if flat or not d else d + "/") + folder + name[:7] + token + name[7:])
    return out


def test_detects_the_namings_met_so_far(tmp_path):
    cases = {
        "next": (raws([(10, 21), (10, 22)]), dict(), r"\1Backscatter/\2_EsB\3"),
        "caps": (raws([(10, 21), (10, 22)]), dict(token="_ESB"), r"\1Backscatter/\2_ESB\3"),
        "approach": (raws([(9, 18)], folder="Approach/") + raws([(9, 22)]), dict(), r"\1Backscatter/\2_EsB\3"),
        "flat": (raws([(9, 18)], folder="Approach/") + raws([(9, 22)]), dict(token="_ESB", flat=True),
                 r"Backscatter/\2_ESB\3"),
    }
    for name, (files, kw, expect) in cases.items():
        raw = tmp_path / name
        touch(raw, *files, *as_esb(files, **kw))
        rep = esb.check(raw, files)
        assert rep.ok and rep.reference_to == expect, (name, rep.lines())
        assert len(rep.present) == len(files) and f"{len(files)} of {len(files)} EsB files present" in rep.summary()
        # the detected pattern is what destreak's reference regex then uses
        assert all(esb.reference_of(f, esb.REFERENCE_FROM, rep.reference_to) == e
                   for f, e in zip(files, as_esb(files, **kw)))


def test_missing_sets_are_reported_by_day_and_folder(tmp_path):
    """As one of our volumes: raw files in Approach/ and at the top, EsB in one flat folder, five days absent."""
    approach = raws([(9, d) for d in range(17, 22)], folder="Approach/", tiles=("0-0", "0-1", "1-0"))
    top = raws([(9, 22), (9, 23), (9, 30), (10, 1)], tiles=("0-0", "0-1", "1-0"))
    have = approach[:3] + top[3:]                       # M09_D17 and M09_D23 on
    raw = tmp_path / "raw"
    touch(raw, *approach, *top, *as_esb(have, token="_ESB", flat=True))
    rep = esb.check(raw, approach + top)
    assert not rep.ok and rep.reference_to == r"Backscatter/\2_ESB\3"
    assert rep.summary() == ("12 of 27 EsB files present; 15 missing: Approach/M09_D18-M09_D21 (12 files), "
                             "M09_D22 (3 files)")
    assert len(rep.missing) == 15 and all(f.startswith(("Approach/M09_D18", "Approach/M09_D19", "Approach/M09_D20",
                                                         "Approach/M09_D21", "M09_D22")) for f in rep.missing)


def test_days_across_a_month_end_are_one_run(tmp_path):
    files = raws([(9, 29), (9, 30), (10, 1), (10, 3)], tiles=("0-0",))
    raw = tmp_path / "raw"
    touch(raw, *files)
    rep = esb.check(raw, files)
    assert rep.reference_to is None and len(rep.missing) == 4
    assert "0 of 4 EsB files present; 4 missing: M09_D29-M10_D01 (3 files), M10_D03 (1 file)" == rep.summary()
    assert "no EsB file was found for any raw file" in "\n".join(rep.lines())


def test_slice_count_and_frame_size_must_match(tmp_path):
    files = raws([(10, 21), (10, 22)], tiles=("0-0",))
    raw = tmp_path / "raw"
    touch(raw, *files, shape=(3, 8, 8))
    e = as_esb(files)
    touch(raw, e[0], shape=(2, 8, 8))     # a partial copy: fewer slices
    touch(raw, e[1], shape=(3, 8, 6))     # another frame size
    rep = esb.check(raw, files)
    assert not rep.ok and not rep.missing and len(rep.mismatched) == 2 and not rep.present
    text = "\n".join(rep.lines())
    assert "2 slices of 8 x 8 px; its raw file M10_D21_tile0-0.tif has 3 slices of 8 x 8 px" in text
    assert "3 slices of 8 x 6 px" in text
    assert esb.check(raw, files, headers=False).ok       # names only: not seen


def test_joined_multi_day_stacks_are_named_with_the_tool_that_splits_them(tmp_path):
    files = raws([(11, 12), (11, 13)], tiles=("0-0",))
    raw = tmp_path / "raw"
    touch(raw, *files, "Backscatter/M11_D12D13_ESB_tile0-0.tif")
    rep = esb.check(raw, files)
    assert not rep.ok and rep.joined == ["Backscatter/M11_D12D13_ESB_tile0-0.tif"]
    assert "tools/split_stack.py" in "\n".join(rep.lines())


def test_mixed_naming_is_reported_not_guessed(tmp_path):
    files = raws([(10, 21), (10, 22), (10, 23)], tiles=("0-0",))
    e = as_esb(files[:2]) + as_esb(files[2:], token="_ESB")
    raw = tmp_path / "raw"
    touch(raw, *files, *e)
    rep = esb.check(raw, files)
    assert not rep.ok and rep.reference_to == r"\1Backscatter/\2_EsB\3" and len(rep.missing) == 1
    assert "named another way" in "\n".join(rep.lines())


def test_a_given_pattern_is_checked_as_given(tmp_path):
    files = raws([(10, 21)], tiles=("0-0",))
    raw = tmp_path / "raw"
    touch(raw, *files, *as_esb(files, token="_ESB"))
    assert not esb.check(raw, files, r"\1Backscatter/\2_EsB\3").ok      # the config says EsB, the files ESB
    assert esb.check(raw, files, r"\1Backscatter/\2_ESB\3").ok


def test_command_line(tmp_path, capsys):
    files = raws([(10, 21), (10, 22)], tiles=("0-0",))
    raw = tmp_path / "raw"
    touch(raw, *files, *as_esb(files[:1]))
    assert esb.main([str(raw)]) == 1
    out = capsys.readouterr().out
    assert "1 of 2 EsB files present; 1 missing: M10_D22 (1 file)" in out and "--no-esb" in out
    touch(raw, *as_esb(files[1:]))
    assert esb.main([str(raw)]) == 0
