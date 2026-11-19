#!/usr/bin/env bash
# Run before every push: fails if the working tree, or a commit not yet on the remote (messages and
# diffs), contains site-specific or personal details: paths on your storage, host and cluster names,
# people, collaborators, buckets, facility dataset IDs, email addresses.
#
#   tools/check_private.sh              files and commits this push would add (vs the upstream branch, or origin/main)
#   tools/check_private.sh --all        the whole working tree + the whole history
#   tools/check_private.sh RANGE        files and commits of a revision range, e.g. origin/main..my-branch
#
# Patterns: tools/private_patterns.txt (committed, generic) plus tools/private_patterns.local.txt
# (git-ignored: your cluster, share, host and people names; one extended regex per line).
# Allowed: Python decorators, matrix '@', shell "$@" / [@], the LICENSE copyright line and the
# Co-Authored-By trailer. Commit author emails are metadata: they are listed for review, not checked.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 2
if [ "${1:-}" = --all ]; then
  range=(--all)
elif [ -n "${1:-}" ]; then
  range=("$1")
else
  up=$(git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' 2>/dev/null || echo origin/main)
  range=("$up..HEAD")
fi
pats=$(mktemp)
trap 'rm -f "$pats"' EXIT
cat tools/private_patterns.txt tools/private_patterns.local.txt 2>/dev/null | grep -vE '^[[:space:]]*(#|$)' > "$pats"
if [ ! -f tools/private_patterns.local.txt ]; then
  echo "note: tools/private_patterns.local.txt not found: only the generic patterns are checked"
fi
allow='^[^:]*:[0-9]*:[[:space:]+-]*@[A-Za-z_][A-Za-z0-9_.]*(\(|$)|^[+ -]?[[:space:]]*@[A-Za-z_][A-Za-z0-9_.]*(\(|$)|Copyright \(c\)|Co-Authored-By: Claude|\$@|\[@\]| @ |@='
if [ "${range[0]}" = --all ]; then
  files=$(git ls-files -co --exclude-standard)
else  # what this push would add: files changed in the range, plus uncommitted and untracked ones
  files=$( { git diff --name-only --diff-filter=d "${range[0]%%..*}" 2>/dev/null; git ls-files -mo --exclude-standard; } | sort -u)
fi
tree=$(printf '%s\n' "$files" | grep -vE '^tools/private_patterns|\.local\.|^$' | while read -r f; do [ -f "$f" ] && printf '%s\0' "$f"; done \
       | xargs -0 grep -nIiE -f "$pats" -- 2>/dev/null | grep -vE "$allow")
hist=$(git log -p "${range[@]}" --format='commit %H%n%B' -- . ':(exclude)tools/private_patterns.txt' \
       | grep -niE -f "$pats" | grep -vE "$allow" | grep -vE '^[0-9]+:[+ -]?(/\[|\[)')
status=0
if [ -n "$tree" ]; then echo "== working tree:"; echo "$tree"; status=1; fi
if [ -n "$hist" ]; then echo "== commits (${range[*]}):"; echo "$hist"; status=1; fi
echo "== commit authors / committers in ${range[*]} (metadata, review by eye):"
git log "${range[@]}" --format='%an <%ae> | %cn <%ce>' | sort -u
[ $status -eq 0 ] && echo "clean: no private patterns in the working tree or in ${range[*]}"
exit $status
