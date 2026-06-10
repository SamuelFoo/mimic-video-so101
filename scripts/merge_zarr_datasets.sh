#!/bin/bash
#
# Merge any number of per-episode .zarr datasets into a single combined
# dataset with sequential episode indices. Episodes are copied so the
# merged dir is independent of the sources.

set -euo pipefail

# ---- Configure ------------------------------------------------------------
SOURCES=(
    "/ephemeral/mimic-video-so101/staging/mimic-video/ex3-1-blue_all-zarr"
    "/ephemeral/mimic-video-so101/staging/mimic-video/ex3-1-orange_all-zarr"
    # add more source dirs here
)
DEST="/ephemeral/mimic-video-so101/data/ex3-1_merged-zarr"
# ---------------------------------------------------------------------------

if (( ${#SOURCES[@]} == 0 )); then
    echo "ERROR: SOURCES is empty. Edit the list at the top of $0." >&2
    exit 2
fi

if [[ -e "$DEST" ]]; then
    echo "ERROR: $DEST already exists. Remove it first or pick a different DEST." >&2
    exit 2
fi

for src in "${SOURCES[@]}"; do
    if [[ ! -d "$src" ]]; then
        echo "ERROR: source dir not found: $src" >&2
        exit 2
    fi
done

mkdir -p "$DEST"

echo "=== Merging ${#SOURCES[@]} zarr datasets ==="
for src in "${SOURCES[@]}"; do
    echo "  - $src ($(ls "$src" | grep -c '^episode_.*\.zarr$') eps)"
done
echo "  -> $DEST"
echo

i=0
for src in "${SOURCES[@]}"; do
    start=$i
    for e in $(ls "$src" | sort); do
        if [[ ! "$e" =~ ^episode_.*\.zarr$ ]]; then
            continue
        fi
        cp -r "$src/$e" "$DEST/episode_$(printf '%06d' $i).zarr"
        i=$((i+1))
    done
    n=$((i - start))
    echo "Copied $n episodes from $(basename "$src") -> indices $start..$((i-1))"
done

echo
echo "=== Done. Merged $i episodes into $DEST ==="
echo "Disk usage:"
du -sh "$DEST"
