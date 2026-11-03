"""Apply a pGM patch stage to the GROMACS source tree: exact, unique string replacements.

    python3 apply.py STAGE.py GROMACS_SRC_ROOT [--check]

A stage file defines EDITS = [(relative_path, old, new), ...].  Every `old` must occur exactly once in
its file (or `new` already be present, or, for an insertion, its first inserted lines: the edit counts as
applied), so the patch is idempotent and fails loudly if upstream code moved.  --check only reports.
"""

import pathlib
import runpy
import sys



def _insertion(old, new):
    """The inserted text if `new` is `old` with text inserted at one place, else None."""
    k = 0
    while k < min(len(old), len(new)) and old[k] == new[k]:
        k += 1
    m = 0
    while m < min(len(old), len(new)) - k and old[len(old) - 1 - m] == new[len(new) - 1 - m]:
        m += 1
    if k + m != len(old):
        return None
    return new[k:len(new) - m]


stage, root = sys.argv[1], pathlib.Path(sys.argv[2])
check = "--check" in sys.argv
edits = runpy.run_path(stage)["EDITS"]
bad = 0
cache = {}
for rel, old, new in edits:
    p = root / rel
    text = cache.get(p) or p.read_text()
    n = text.count(old)
    if new and new in text:
        print(f"already applied: {rel}: {old.strip().splitlines()[0][:70]}")
        cache[p] = text
        continue
    inserted = _insertion(old, new)
    if inserted is not None:
        # an insertion whose text a later stage has extended: applied if its first 160 characters are there
        head = inserted.strip("\n")[:160]
        if len(head.strip()) >= 40 and head in text:
            print(f"already applied (extended later): {rel}: {old.strip().splitlines()[0][:60]}")
            cache[p] = text
            continue
    if n != 1:
        print(f"ERROR {rel}: expected one match, found {n}: {old.strip().splitlines()[0][:90]!r}")
        bad += 1
        continue
    cache[p] = text.replace(old, new)
    print(f"ok {rel}: {old.strip().splitlines()[0][:70]}")
if bad:
    sys.exit(f"{bad} edits failed, nothing written")
if not check:
    for p, text in cache.items():
        p.write_text(text)
    print(f"wrote {len(cache)} files")
