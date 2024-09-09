# loop through all .mp4 files in the current directory
for F in *.mp4; do
  # create a directory with the same name as the video file if it does not exist
  DIR_NAME=$(basename "$F" .mp4)
  mkdir -p "$DIR_NAME"
  
  # extract a frame at 7 second and save it in the directory
  ffmpeg -ss 00:00:07 -i "$F" -frames:v 1 "${DIR_NAME}/${DIR_NAME}-frame.jpg"
done
