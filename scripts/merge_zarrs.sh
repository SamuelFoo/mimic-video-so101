cd /ephemeral/mimic-video-so101/data

mkdir -p ex3_all-zarr

idx=0

for folder in ex3-1-blue_all-zarr ex3-2-blue_all-zarr ex3-1-orange_all-zarr ex3-2-orange_all-zarr; do
    echo "Merging $folder"

    for zarr in $(find "$folder" -maxdepth 1 -name 'episode_*.zarr' | sort -V); do
        target=$(printf "ex3_all-zarr/episode_%06d.zarr" "$idx")

        if [ -e "$target" ]; then
            echo "ERROR: $target already exists"
            exit 1
        fi

        cp -r "$zarr" "$target"
        echo "$zarr -> $target"

        idx=$((idx + 1))
    done
done

echo "Done. Copied $idx episodes into ex3_all-zarr"
