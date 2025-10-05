import os
import sys
import json
import shutil
import tempfile
import subprocess
from typing import Optional, List


def ensure_tools_available():
    if shutil.which("yt-dlp") is None:
        raise RuntimeError("yt-dlp is not installed or not in PATH")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not installed or not in PATH")


def try_download_subtitles(url: str, out_dir: str) -> Optional[str]:
    # Attempt to download available subtitles (auto or user-provided)
    # We'll prefer English if available, else accept any
    output_template = os.path.join(out_dir, "%(title)s.%(ext)s")

    base_cmd: List[str] = [
        "yt-dlp",
        "--no-playlist",
        "--restrict-filenames",
        "--geo-bypass",
        "--user-agent",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0 Safari/537.36",
        "--add-header",
        "Referer:https://www.youtube.com",
        "-o",
        output_template,
        # subtitle flags
        "--write-auto-subs",
        "--write-subs",
        "--sub-langs",
        "en.*,live_chat",  # prefer English, allow variants
        "--sub-format",
        "vtt/srt/best",
        "--skip-download",
    ]

    cookies_file = os.path.join(os.getcwd(), "cookies.txt")
    if os.path.exists(cookies_file):
        base_cmd += ["--cookies", cookies_file]

    attempts = [
        base_cmd + ["--extractor-args", "youtube:player_client=web"],
        base_cmd + ["--extractor-args", "youtube:player_client=android"],
        base_cmd + ["--extractor-args", "youtube:player_client=ios"],
        base_cmd,
    ]

    last_error: Optional[Exception] = None
    for cmd in attempts:
        try:
            subprocess.check_call(cmd + [url])
            break
        except subprocess.CalledProcessError as e:
            last_error = e
            continue
    else:
        if last_error:
            raise last_error

    # Pick the newest .vtt or .srt file
    subtitle_candidates = []
    for name in os.listdir(out_dir):
        if name.endswith(".vtt") or name.endswith(".srt"):
            subtitle_candidates.append(os.path.join(out_dir, name))

    if not subtitle_candidates:
        return None

    subtitle_candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return subtitle_candidates[0]


def normalize_audio(input_file: str, output_file: str):
    subprocess.check_call(["ffmpeg", "-y", "-i", input_file, "-ac", "1", "-ar", "16000", output_file])


def download_audio(url: str, out_dir: str) -> str:
    output_template = os.path.join(out_dir, "%(title)s.%(ext)s")
    base_cmd: List[str] = [
        "yt-dlp",
        "-f",
        "bestaudio/best",
        "--no-playlist",
        "--restrict-filenames",
        "--geo-bypass",
        "--user-agent",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0 Safari/537.36",
        "--add-header",
        "Referer:https://www.youtube.com",
        "-o",
        output_template,
    ]

    cookies_file = os.path.join(os.getcwd(), "cookies.txt")
    if os.path.exists(cookies_file):
        base_cmd += ["--cookies", cookies_file]

    attempts = [
        base_cmd + ["--extractor-args", "youtube:player_client=web"],
        base_cmd + ["--extractor-args", "youtube:player_client=android"],
        base_cmd + ["--extractor-args", "youtube:player_client=ios"],
        base_cmd + ["--extractor-args", "youtube:player_client=android"],
        base_cmd,
    ]

    last_error: Optional[Exception] = None
    for cmd in attempts:
        try:
            subprocess.check_call(cmd + [url])
            break
        except subprocess.CalledProcessError as e:
            last_error = e
            continue
    else:
        if last_error:
            raise last_error

    # Find newest file produced
    candidates = [
        os.path.join(out_dir, f)
        for f in os.listdir(out_dir)
        if not (f.endswith(".part") or f.endswith(".ytdl"))
    ]
    if not candidates:
        raise RuntimeError("Download failed: no output files produced")
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def transcribe_with_openai(wav_path: str, api_key: str) -> str:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("openai package not installed. Add it to requirements.txt") from e
    client = OpenAI(api_key=api_key)
    with open(wav_path, "rb") as f:
        resp = client.audio.transcriptions.create(model="whisper-1", file=f)
    return resp.text


def load_api_key_interactive() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key
    try:
        # Prompt user if running interactively
        api_key = input("Enter OpenAI API key (or leave blank to skip Whisper fallback): ").strip()
    except EOFError:
        api_key = ""
    return api_key


def convert_subtitles_to_text(subtitle_path: str) -> str:
    # For .vtt or .srt, yt-dlp produces plain subtitle files already readable.
    # We'll just return raw content so user gets exact captions.
    with open(subtitle_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_captions.py <video_url> [--out <output.txt>]", file=sys.stderr)
        sys.exit(1)

    url = sys.argv[1]
    out_path: Optional[str] = None
    if "--out" in sys.argv:
        try:
            out_index = sys.argv.index("--out")
            out_path = sys.argv[out_index + 1]
        except Exception:
            print("--out flag provided without a following path", file=sys.stderr)
            sys.exit(1)

    ensure_tools_available()

    tmpdir = tempfile.mkdtemp()
    try:
        # 1) Try native subtitles first
        subtitle_file = try_download_subtitles(url, tmpdir)
        if subtitle_file:
            captions_text = convert_subtitles_to_text(subtitle_file)
        else:
            # 2) Fallback: download audio and use Whisper
            print("No subtitles found. Falling back to transcription via Whisper.")
            api_key = load_api_key_interactive()
            if not api_key:
                print("OpenAI API key not provided; cannot transcribe. Exiting.", file=sys.stderr)
                sys.exit(2)

            audio_file = download_audio(url, tmpdir)
            wav_path = os.path.join(tmpdir, "audio.wav")
            normalize_audio(audio_file, wav_path)
            captions_text = transcribe_with_openai(wav_path, api_key)

        # Output handling
        if not out_path:
            # Default file name
            out_path = os.path.join(os.getcwd(), "captions.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(captions_text)
        print(out_path)
    finally:
        # Leave tmpdir for debugging? We can clean automatically.
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()


