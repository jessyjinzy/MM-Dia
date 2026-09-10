import subprocess
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm
import argparse

def compress_to_lowp(input_path, output_path, height=360):
    cmd = [
        "/usr/bin/ffmpeg", "-y", "-i", input_path,
        "-vf", f"scale=-2:{height}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
        "-c:a", "aac", "-b:a", "64k",
        "-threads", "12",
        "-loglevel", "error",
        output_path
    ]
    subprocess.run(cmd, check=True)

def compress_to_fps(input_path, output_path, fps=25):
    # ffmpeg -i input.mp4 -r 30 -preset fast -crf 23 output.mp4
    # ffmpeg -y -i %s -qscale:v 2 -threads %d -async 1 -r 25 %s -loglevel panic
    cmd = [
        "/usr/bin/ffmpeg", "-i", input_path,
        "-qscale:v", "2",
        "-threads", "12",
        "-async", "1",
        "-r", str(fps),
        "-loglevel", "panic",
        "-y",  # Overwrite output file without asking
        output_path
    ]
    subprocess.run(cmd, check=True)

def extract_audio(input_path, output_path):
    subprocess.run([
        "/usr/bin/ffmpeg",
        "-y",   
        "-loglevel", "error",
        "-i", input_path,
        "-ar", "24000",
        output_path 
    ], check=True)


def denoise_video(input_video, input_wav, output_path):
    cmd = [
        #/usr/bin/ffmpeg -i input.mp4 -i new_audio.wav -c:v copy -map 0:v:0 -map 1:a:0 -shortest output.mp4
        "/usr/bin/ffmpeg", "-y", "-loglevel", "error",
        "-i", input_video,
        "-i", input_wav,
        "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0",
        "-shortest", output_path
    ]
    subprocess.run(cmd, check=True)


def multiprocess_handler(args):
    function, f_args = args[0], args[1:]
    function(*f_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Processing Script")
    parser.add_argument("--video_root", type=str, required=False, default="video/tv_01_mad_men")
    parser.add_argument("--target_root", type=str, required=False, default="audio/tv_01_mad_men")
    args_parsed = parser.parse_args()

    video_root = args_parsed.video_root
    target_root = args_parsed.target_root

    video_paths = list(Path(video_root).rglob("*.mp4")) + list(Path(video_root).rglob("*.mkv"))

    args = []
    for video_file in tqdm(video_paths, desc="Processing videos"):
        ### option.0 extract_audio
        output_file = Path(target_root) / video_file.relative_to(video_root)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        #extract_audio(str(video_file), str(output_file.with_suffix(".wav")))
        args.append((extract_audio, str(video_file), str(output_file.with_suffix(".wav"))))

        ### option.1 compress_to_360p
        #output_file = Path(target_root) / video_file.relative_to(video_root)
        #output_file.parent.mkdir(parents=True, exist_ok=True)
        #compress_to_360p(str(video_file), str(output_file))
        #args.append((compress_to_lowp, str(video_file), str(output_file), 224))

        ### option.2 denoise_video
        #audio_file = Path(audio_root) / video_file.stem / video_file.relative_to(video_root).with_suffix(".wav").name.replace('.wav', '_orig.wav')
        ##audio_file = Path(audio_root) / (video_file.parts[-2] + '_processed') / video_file.stem / video_file.relative_to(video_root).with_suffix(".wav").name.replace('.wav', '_orig.wav')
        #output_file = Path(target_root) / video_file.relative_to(video_root)
        #output_file.parent.mkdir(parents=True, exist_ok=True)
        #print(f"Compressing {video_file} to {output_file} with audio {audio_file}")
        #if not audio_file.exists():
        #    print(f"Audio file {audio_file} does not exist, skipping {video_file}")
        #    continue
        #if output_file.exists():
        #    print(f"Output file {output_file} already exists, skipping {video_file}")
        #    continue
        ##denoise_video(str(video_file), str(audio_file), str(output_file))
        #args.append((denoise_video, str(video_file), str(audio_file), str(output_file)))
        #raise Exception("Stopping after denoise_video for testing purposes")


        ### option.3 compress_to_fps
        #output_file = Path(target_root) / video_file.relative_to(video_root)
        #output_file.parent.mkdir(parents=True, exist_ok=True)
        #print(f"Compressing {video_file} to {output_file} at 16fps")
        #if output_file.exists():
        #    print(f"Output file {output_file} already exists, skipping {video_file}")
        #    continue
        #compress_to_fps(str(video_file), str(output_file))
        #args.append((compress_to_fps, str(video_file), str(output_file), 16))

    with Pool(processes=12) as pool:
        list(tqdm(pool.imap(multiprocess_handler, args), total=len(args), desc="Processing videos"))
    print("All videos processed.")