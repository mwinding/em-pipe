"""Undo the shell quoting left in inline scripts that were run inside an ssh '...' or "..." command.

A heredoc nested in a quoted remote command keeps the outer layer's escaping (\\" for ", '\\'' for ')
and can end with the closing quote of the outer command (EOF' or EOF"). For each Python file that does
not compile, try those repairs and keep the result only if it then compiles.

usage: python -I fix_ssh_quoting.py DIR [DIR ...]
"""
import sys
from pathlib import Path


def compiles(src):
    try:
        compile(src, "<inline>", "exec")
        return True
    except SyntaxError:
        return False


def repairs(src):
    lines = src.rstrip("\n").split("\n")
    # the heredoc's end marker may share its line with the rest of the outer command (EOF' && ...)
    end = next((i for i, ln in enumerate(lines) if ln.startswith("EOF")), None)
    if end is not None:
        lines = lines[:end]
    while lines and lines[-1].strip() in ("EOF'", 'EOF"', "EOF", "'", '"', "EOF' 2>&1", "EOF'\""):
        lines.pop()
    base = "\n".join(lines) + "\n"
    yield base
    yield base.replace('\\"', '"')
    yield base.replace("'\\''", "'")
    yield base.replace('\\"', '"').replace("'\\''", "'").replace("\\$", "$")


def main(dirs):
    fixed, broken = 0, []
    for d in dirs:
        for p in sorted(Path(d).rglob("*.py")):
            src = p.read_text()
            if compiles(src):
                continue
            for cand in repairs(src):
                if compiles(cand):
                    p.write_text(cand)
                    fixed += 1
                    break
            else:
                broken.append(p)
    print(f"repaired {fixed}")
    for p in broken:
        print("still does not compile:", p)


if __name__ == "__main__":
    main(sys.argv[1:])
