"""Small, recorded changes made to the recovered dev scripts after sanitising (so every difference from
what originally ran is visible here):

1. masked_stats.py rewrites <test>/work/preview/stats.csv by design (it built the masked-normalisation
   test dataset): it now refuses to run when <test> is <main>, so a dataset's own preview stats can't
   be overwritten by mistake.
2. Shell and text references to $OUT_ROOT in braces (written by the sanitiser for redirected qc/
   outputs) are spelled as $DEV_OUT in braces, the variable Python reads for the same purpose.
3. spot_compare.py no longer deletes the previous run's *.tif in its output folder before writing
   (nothing in dev/ deletes files); same-named stacks are overwritten.
4. qc_jumps.py's plot title named the dataset by its facility project number; it now says 35h only.

Safe to run again: each change is applied only once.

usage: python -I patch_dev.py DEV_DIR
"""
import sys
from pathlib import Path

GUARD_OLD = "MAIN, TEST = Path(sys.argv[1]), Path(sys.argv[2])\n"
GUARD_NEW = (GUARD_OLD + "if TEST.resolve() == MAIN.resolve():\n"
             "    sys.exit(\"masked_stats: <test> must be a separate scratch copy, not <main> (it rewrites <test>/work/preview/stats.csv)\")\n")
CLEANUP = 'for old in out.glob("*.tif"):\n    old.unlink()\n'
# built from parts so this file never matches (and rewrites) itself
OLD_VAR, NEW_VAR = "${" + "OUT_ROOT}", "${" + "DEV_OUT}"


def main(root):
    root = Path(root)
    ms = next(root.rglob("masked_stats.py"))
    s = ms.read_text()
    if GUARD_NEW not in s:
        assert s.count(GUARD_OLD) == 1, "masked_stats.py: anchor not found"
        ms.write_text(s.replace(GUARD_OLD, GUARD_NEW))
    sc = next(root.rglob("spot_compare.py"))
    s = sc.read_text()
    if CLEANUP in s:
        sc.write_text(s.replace(CLEANUP, ""))
    qj = next(root.rglob("qc_jumps.py"))
    s = qj.read_text()
    title = 'fig.suptitle(f"P' + '667 35h:'
    if title in s:
        qj.write_text(s.replace(title, 'fig.suptitle(f"35h:'))
    n = 0
    for p in root.rglob("*"):
        if "_assembly" in p.parts or not p.is_file() or p.suffix not in (".py", ".sh", ".sbatch", ".md", ".txt"):
            continue
        t = p.read_text()
        if OLD_VAR in t:
            p.write_text(t.replace(OLD_VAR, NEW_VAR))
            n += 1
    print(f"masked_stats guard and spot_compare clean-up removal checked; {OLD_VAR} -> {NEW_VAR} in {n} files")


if __name__ == "__main__":
    main(sys.argv[1])
