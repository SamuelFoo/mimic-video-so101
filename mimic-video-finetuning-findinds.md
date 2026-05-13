### Finetuning Video model of Mimic-video

## Num_Frames

The number of frames we can define in ```mimic-video/model/cosmos_predict2/configs/defaults/data_video.py``` 

is 1, 5 or 61. To speed finetine we can set it to 5.

## Batch_Size

If our amount of generated videos is smaller than ```bsz``` defined in ```mimic-video/model/cosmos_predict2/configs/experiment/video2world.py``` then it will result in an error. Adjust accordingly

## Encoding of the .mp4 videos

I had an issue that I couldnt stream the LeRobot videos since they are AV1 encoded, thus we need to Re-encode to H.264:

```bash
cd /home/ubuntu/<DATASET_LOCATION>

mkdir -p video_h264

for f in video/*.mp4; do
  base=$(basename "$f")
  ffmpeg -y -i "$f" \
    -map 0:v:0 \
    -c:v libx264 \
    -pix_fmt yuv420p \
    -r 30 \
    -movflags +faststart \
    -an \
    "video_h264/$base"
done

mv video video_av1
mv video_h264 video
```




