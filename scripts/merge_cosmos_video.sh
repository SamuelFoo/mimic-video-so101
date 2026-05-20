#!/bin/bash
set -euo pipefail

SOURCES=(
    "/ephemeral/robot_learning_project/data/ex3-1-blue_all-cosmos-video"
    "/ephemeral/robot_learning_project/data/ex3-2-blue_all-cosmos-video"
    "/ephemeral/robot_learning_project/data/ex3-1-orange_all-cosmos-video"
    "/ephemeral/robot_learning_project/data/ex3-2-orange_all-cosmos-video"
)

DEST="/ephemeral/robot_learning_project/data/ex3_all-cosmos-video"

if [[ -e "$DEST" ]]; then
    echo "ERROR: $DEST already exists. Remove it first or choose another DEST." >&2
    exit 2
fi

for src in "${SOURCES[@]}"; do
    if [[ ! -d "$src" ]]; then
        echo "ERROR: source dir not found: $src" >&2
        exit 2
    fi

    for sub in metas t5_xxl video; do
        if [[ ! -d "$src/$sub" ]]; then
            echo "ERROR: missing folder: $src/$sub" >&2
            exit 2
        fi
    done
done

mkdir -p "$DEST/metas" "$DEST/t5_xxl" "$DEST/video"

echo "=== Merging ${#SOURCES[@]} cosmos-video folders ==="
echo "Destination: $DEST"
echo

i=0

for src in "${SOURCES[@]}"; do
    start=$i

    for meta_file in $(find "$src/metas" -maxdepth 1 -name 'episode_*.txt' -printf '%f\n' | sort -V); do
        old_id="${meta_file#episode_}"
        old_id="${old_id%.txt}"

        src_meta="$src/metas/episode_${old_id}.txt"
        src_t5="$src/t5_xxl/episode_${old_id}.pickle"
        src_video="$src/video/episode_${old_id}.mp4"

        if [[ ! -f "$src_meta" ]]; then
            echo "ERROR: missing $src_meta" >&2
            exit 2
        fi

        if [[ ! -f "$src_t5" ]]; then
            echo "ERROR: missing $src_t5" >&2
            exit 2
        fi

        if [[ ! -f "$src_video" ]]; then
            echo "ERROR: missing $src_video" >&2
            exit 2
        fi

        new_name=$(printf "episode_%06d" "$i")

        cp "$src_meta" "$DEST/metas/${new_name}.txt"
        cp "$src_t5" "$DEST/t5_xxl/${new_name}.pickle"
        cp "$src_video" "$DEST/video/${new_name}.mp4"

        i=$((i + 1))
    done

    n=$((i - start))
    echo "Copied $n episodes from $(basename "$src") -> indices $start..$((i-1))"
done

echo
echo "=== Done. Merged $i episodes into $DEST ==="
echo

echo "Counts:"
echo "metas:  $(find "$DEST/metas" -maxdepth 1 -name 'episode_*.txt' | wc -l)"
echo "t5_xxl: $(find "$DEST/t5_xxl" -maxdepth 1 -name 'episode_*.pickle' | wc -l)"
echo "video:  $(find "$DEST/video" -maxdepth 1 -name 'episode_*.mp4' | wc -l)"

echo
echo "Disk usage:"
du -sh "$DEST"
