# Captions Generator (FastAPI + yt-dlp)

A small FastAPI service and CLI that fetches captions from a video URL using yt-dlp. If subtitles aren't available, it can fall back to Whisper transcription (requires an OpenAI API key).

## Features
- Fetch subtitles via yt-dlp and clean them into plain text
- Optional Whisper transcription fallback when subtitles are missing
- Simple web UI at `/captions-ui`
- CLI script `generate_captions.py` to output captions to a file

## Requirements
- Python 3.10+
- ffmpeg installed and in PATH
- yt-dlp installed (comes from `requirements.txt`)
- Optional: `OPENAI_API_KEY` for Whisper fallback
- Optional: `cookies.txt` in project root for age/region-restricted videos

## Setup
```powershell
cd "C:\\Users\\<you>\\Documents\\v to s"
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install ffmpeg (Windows options):
- winget: `winget install --id Gyan.FFmpeg -e`
- choco: `choco install ffmpeg -y`

Verify:
```powershell
ffmpeg -version
yt-dlp --version
```

## Run the API server
```powershell
$env:OPENAI_API_KEY="your_api_key_here"   # only needed for Whisper fallback
uvicorn app:app --reload --port 8000
```
Open the UI: `http://127.0.0.1:8000/captions-ui`

### API
- POST `/captions`
  - Body: `{ "url": "https://www.youtube.com/watch?v=..." }`
  - Response: `{ "captions": "...", "source": "subtitles|whisper" }`

- POST `/summarize` (existing feature)
  - Body: `{ "url": "...", "desired_sentences": 6 }`

- POST `/summarize_upload` (existing feature)
  - Multipart form with file upload

## CLI usage
Generate captions into a file:
```powershell
python generate_captions.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --out captions.txt
```
Notes:
- The script auto-uses `cookies.txt` if present.
- If no subtitles, it'll prompt for `OPENAI_API_KEY` if not set.



