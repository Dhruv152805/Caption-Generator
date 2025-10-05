import os
import tempfile
import subprocess
from typing import Optional, List, Tuple
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, HttpUrl, Field
from openai import OpenAI

# Create OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# ----------------------
# Request & Response Models
# ----------------------
class SummarizeRequest(BaseModel):
    url: HttpUrl
    desired_sentences: int = Field(default=6, ge=1, le=15)

class SummaryResponse(BaseModel):
    title: Optional[str]
    summary: str
    chunk_summaries: List[str]

class CaptionsRequest(BaseModel):
    url: HttpUrl

class CaptionsResponse(BaseModel):
    captions: str
    source: str = Field(description="where captions came from: 'subtitles' or 'whisper'")
# ----------------------
# Helper Functions
# ----------------------
def ensure_tools_available():
    if shutil.which("yt-dlp") is None:
        raise RuntimeError("yt-dlp is not installed or not in PATH")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not installed or not in PATH")


def download_audio(url: str, out_dir: str) -> Tuple[str, str]:
    out_template = os.path.join(out_dir, "%(title)s.%(ext)s")

    base_cmd = [
        "yt-dlp",
        "-f", "bestaudio/best",
        "--no-playlist",
        "--restrict-filenames",
        "--geo-bypass",
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0 Safari/537.36",
        "--add-header", "Referer:https://www.youtube.com",
        "-o", out_template,
    ]

    # Determine cookie options
    browser = os.getenv("YTDLP_COOKIES_FROM_BROWSER")
    cookies_file = os.getenv("YTDLP_COOKIES_FILE")

    attempts: List[List[str]] = []

    def with_cookies(cmd: List[str]) -> List[str]:
        if browser:
            return cmd + ["--cookies-from-browser", browser]
        if cookies_file and os.path.exists(cookies_file):
            return cmd + ["--cookies", cookies_file]
        return cmd

    # Try with web client + cookies
    attempts.append(with_cookies(base_cmd + ["--extractor-args", "youtube:player_client=web"]))
    # Try android client (often bypasses nsig/403)
    attempts.append(with_cookies(base_cmd + ["--extractor-args", "youtube:player_client=android"]))
    # Try ios client
    attempts.append(with_cookies(base_cmd + ["--extractor-args", "youtube:player_client=ios"]))
    # Try without cookies + android
    attempts.append(base_cmd + ["--extractor-args", "youtube:player_client=android"]) 
    # Plain fallback
    attempts.append(base_cmd)

    last_error: Optional[Exception] = None
    for attempt in attempts:
        try:
            subprocess.check_call(attempt + [url])
            break
        except subprocess.CalledProcessError as e:
            last_error = e
            continue
    else:
        if last_error:
            raise last_error

    candidates = [
        f for f in os.listdir(out_dir)
        if not f.endswith(".part") and not f.endswith(".ytdl")
    ]
    if not candidates:
        raise RuntimeError("Download failed: no output files produced")

    candidates.sort(key=lambda f: os.path.getmtime(os.path.join(out_dir, f)), reverse=True)
    filename = candidates[0]
    title = os.path.splitext(filename)[0]
    return os.path.join(out_dir, filename), title

def normalize_audio(input_file: str, output_file: str):
    cmd = ["ffmpeg", "-y", "-i", input_file, "-ac", "1", "-ar", "16000", output_file]
    subprocess.check_call(cmd)

def transcribe_audio_whisper(audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(model="whisper-1", file=f)
    return resp.text

def try_download_subtitles(url: str, out_dir: str) -> Optional[str]:
    """Attempt to download subtitles using yt-dlp. Returns path to .vtt/.srt or None."""
    out_template = os.path.join(out_dir, "%(title)s.%(ext)s")

    base_cmd = [
        "yt-dlp",
        "--no-playlist",
        "--restrict-filenames",
        "--geo-bypass",
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0 Safari/537.36",
        "--add-header", "Referer:https://www.youtube.com",
        "-o", out_template,
        "--write-auto-subs",
        "--write-subs",
        "--sub-langs", "en.*",
        "--sub-format", "vtt/srt/best",
        "--skip-download",
    ]

    cookies_path = os.path.join(os.getcwd(), "cookies.txt")
    if os.path.exists(cookies_path):
        base_cmd += ["--cookies", cookies_path]

    attempts: List[List[str]] = [
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

    candidates = [
        os.path.join(out_dir, f)
        for f in os.listdir(out_dir)
        if f.endswith(".vtt") or f.endswith(".srt")
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

def subtitles_to_plain_text(subtitle_path: str) -> str:
    """Convert .vtt/.srt content into plain text by removing timestamps, cue ids, and tags."""
    import re

    with open(subtitle_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    plain_lines: List[str] = []

    timestamp_pattern = re.compile(r"\\d{2}:\\d{2}:\\d{2}\\.\\d{3}\\s+--\\>\\s+\\d{2}:\\d{2}:\\d{2}\\.\\d{3}")
    srt_timestamp_pattern = re.compile(r"\\d{2}:\\d{2}:\\d{2},\\d{3}\\s+--\\>\\s+\\d{2}:\\d{2}:\\d{2},\\d{3}")
    cue_number_pattern = re.compile(r"^\\d+$")
    settings_pattern = re.compile(r"\\balign:|\\bposition:|\\bsize:|\\bline:")
    inline_timecode_pattern = re.compile(r"<\\d{2}:\\d{2}:\\d{2}\\.\\d{3}>")

    for raw in lines:
        line = raw.strip()
        if not line:
            plain_lines.append("")
            continue
        # Drop headers / metadata
        if line.startswith("WEBVTT") or line.startswith("Kind:") or line.startswith("Language:"):
            continue
        # Drop cue numbers and timestamp lines
        if cue_number_pattern.match(line):
            continue
        if timestamp_pattern.search(line) or srt_timestamp_pattern.search(line):
            continue
        if settings_pattern.search(line):
            continue

        # Remove inline tags like <c>...</c>, <00:00:00.000>, and any other simple tags
        line = inline_timecode_pattern.sub("", line)
        line = re.sub(r"</?c[^>]*>", "", line)  # remove <c> style tags
        line = re.sub(r"</?[^>]+>", "", line)   # remove any remaining tags
        # Remove bracketed tokens like [ __ ] or [Music]
        line = re.sub(r"\[[^\]]*\]", "", line)
        # Remove inline VTT settings like align:start position:0% size:NN line:MM
        line = re.sub(r"\b(?:align|position|size|line):[^\s%]+%?", "", line, flags=re.IGNORECASE)

        # Collapse multiple spaces created by removals
        line = re.sub(r"\\s+", " ", line).strip()
        if line:
            plain_lines.append(line)

    # Remove consecutive duplicates and collapse blank lines
    out: List[str] = []
    prev_blank = False
    prev_text: Optional[str] = None
    seen_texts = set()
    for l in plain_lines:
        if not l:
            if not prev_blank and out:
                out.append("")
            prev_blank = True
            continue
        # Skip exact consecutive duplicates (case-insensitive)
        if prev_text is not None and l.lower() == prev_text.lower():
            continue
        # Skip global duplicates to prevent repeats across overlapping cues
        key = l.lower()
        if key in seen_texts:
            continue
        out.append(l)
        prev_text = l
        prev_blank = False
        seen_texts.add(key)

    return "\n".join(out).strip()

def chunk_text(text: str, max_words: int = 450) -> List[str]:
    sentences = text.split(". ")
    chunks, cur, cur_len = [], [], 0
    for s in sentences:
        cur.append(s)
        cur_len += len(s.split())
        if cur_len > max_words:
            chunks.append(". ".join(cur).strip() + ".")
            cur, cur_len = [], 0
    if cur:
        chunks.append(". ".join(cur).strip())
    return chunks

def summarize_chunk(chunk_text: str) -> str:
    prompt = f"""
You are an assistant that summarizes spoken video transcripts.

Transcript chunk:
\"\"\"{chunk_text}\"\"\"

Provide a short summary (3–6 sentences) and list 3 key takeaways.
Format:
SUMMARY:
- <sentences>
KEY TAKEAWAYS:
1.
2.
3.
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You summarize transcripts concisely."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=400,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

def combine_summaries(chunk_summaries: List[str], final_len_sentences: int = 6) -> str:
    joined = "\n\n".join(chunk_summaries)
    prompt = f"""
You are an assistant that produces a concise final summary from chunk summaries.

Chunk summaries:
{joined}

Produce a polished final summary in {final_len_sentences} sentences,
a 4-bullet quick TL;DR, and 3 suggested timestamps/topics.
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You synthesize content into clean summaries."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

# ----------------------
# API Endpoints
# ----------------------
@app.post("/summarize", response_model=SummaryResponse)
async def summarize_url(req: SummarizeRequest):
    tmpdir = tempfile.mkdtemp()
    try:
        # Validate runtime prerequisites
        ensure_tools_available()
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY environment variable is not set")

        audio_file, title = download_audio(str(req.url), tmpdir)
        wav_file = os.path.join(tmpdir, "audio.wav")
        normalize_audio(audio_file, wav_file)

        transcript = transcribe_audio_whisper(wav_file)

        chunks = chunk_text(transcript)
        chunk_summaries = [summarize_chunk(c) for c in chunks]

        final = combine_summaries(chunk_summaries, final_len_sentences=req.desired_sentences)

        return SummaryResponse(title=title, summary=final, chunk_summaries=chunk_summaries)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Media processing error: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "FastAPI is running!"}


# ----------------------
# Captions Endpoint + UI
# ----------------------

@app.post("/captions", response_model=CaptionsResponse)
async def get_captions(req: CaptionsRequest):
    tmpdir = tempfile.mkdtemp()
    try:
        ensure_tools_available()
        # First try subtitles
        sub_file = try_download_subtitles(str(req.url), tmpdir)
        if sub_file:
            return CaptionsResponse(captions=subtitles_to_plain_text(sub_file), source="subtitles")

        # Fallback to Whisper transcription
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(status_code=400, detail="OPENAI_API_KEY is not set and no subtitles available")

        audio_file, _title = download_audio(str(req.url), tmpdir)
        wav_file = os.path.join(tmpdir, "audio.wav")
        normalize_audio(audio_file, wav_file)
        text = transcribe_audio_whisper(wav_file)
        return CaptionsResponse(captions=text, source="whisper")
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Media processing error: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.get("/captions-ui", response_class=HTMLResponse)
async def captions_ui():
    # Minimal HTML UI
    html = """
<!doctype html>
<html>
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Captions Generator</title>
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem; }
      .container { max-width: 800px; margin: 0 auto; }
      input[type=text] { width: 100%; padding: 10px; font-size: 16px; }
      button { padding: 10px 16px; font-size: 16px; margin-top: 8px; }
      pre { white-space: pre-wrap; background: #f7f7f7; padding: 12px; border-radius: 6px; }
      .muted { color: #666; font-size: 14px; }
    </style>
  </head>
  <body>
    <div class=\"container\">
      <h1>Captions Generator</h1>
      <p class=\"muted\">Paste a YouTube URL and get captions.</p>
      <input id=\"url\" type=\"text\" placeholder=\"https://www.youtube.com/watch?v=...\" />
      <button id=\"go\">Get Captions</button>
      <div id=\"status\" class=\"muted\" style=\"margin-top:8px;\"></div>
      <h3>Captions</h3>
      <pre id=\"out\"></pre>
    </div>
    <script>
      const btn = document.getElementById('go');
      const urlInput = document.getElementById('url');
      const out = document.getElementById('out');
      const status = document.getElementById('status');
      btn.addEventListener('click', async () => {
        const url = urlInput.value.trim();
        if (!url) { alert('Please enter a URL'); return; }
        out.textContent = '';
        status.textContent = 'Fetching captions...';
        try {
          const resp = await fetch('/captions', {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify({ url })
          });
          if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || ('HTTP ' + resp.status));
          }
          const data = await resp.json();
          status.textContent = 'Source: ' + data.source;
          out.textContent = data.captions;
        } catch (e) {
          status.textContent = 'Error: ' + e.message;
        }
      });
    </script>
  </body>
</html>
"""
    return HTMLResponse(content=html)

# ----------------------
# Alternative Inputs (No yt-dlp)
# ----------------------

@app.post("/summarize_upload", response_model=SummaryResponse)
async def summarize_upload(
    file: UploadFile = File(...),
    desired_sentences: int = Form(6),
):
    tmpdir = tempfile.mkdtemp()
    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(status_code=400, detail="OPENAI_API_KEY is not set")

        # Save uploaded content
        input_path = os.path.join(tmpdir, file.filename or "input.bin")
        with open(input_path, "wb") as out:
            out.write(await file.read())

        # Normalize to wav mono 16k
        wav_file = os.path.join(tmpdir, "audio.wav")
        ensure_tools_available()
        normalize_audio(input_path, wav_file)

        # Transcribe and summarize
        transcript = transcribe_audio_whisper(wav_file)
        chunks = chunk_text(transcript)
        chunk_summaries = [summarize_chunk(c) for c in chunks]
        final = combine_summaries(chunk_summaries, final_len_sentences=desired_sentences)
        return SummaryResponse(title=file.filename, summary=final, chunk_summaries=chunk_summaries)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Media processing error: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class AudioURLRequest(BaseModel):
    audio_url: HttpUrl
    desired_sentences: int = 6


@app.post("/summarize_audio_url", response_model=SummaryResponse)
async def summarize_audio_url(req: AudioURLRequest):
    tmpdir = tempfile.mkdtemp()
    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(status_code=400, detail="OPENAI_API_KEY is not set")
        ensure_tools_available()

        # Download direct audio file without yt-dlp
        import requests

        resp = requests.get(str(req.audio_url), timeout=60)
        if resp.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Failed to fetch audio: HTTP {resp.status_code}")
        # Guess extension from Content-Type
        content_type = resp.headers.get("content-type", "application/octet-stream")
        ext = {
            "audio/mpeg": ".mp3",
            "audio/mp3": ".mp3",
            "audio/mp4": ".m4a",
            "audio/x-m4a": ".m4a",
            "audio/webm": ".webm",
            "audio/ogg": ".ogg",
            "audio/wav": ".wav",
        }.get(content_type.split(";")[0].lower(), ".bin")

        src_path = os.path.join(tmpdir, f"download{ext}")
        with open(src_path, "wb") as f:
            f.write(resp.content)

        wav_file = os.path.join(tmpdir, "audio.wav")
        normalize_audio(src_path, wav_file)

        transcript = transcribe_audio_whisper(wav_file)
        chunks = chunk_text(transcript)
        chunk_summaries = [summarize_chunk(c) for c in chunks]
        final = combine_summaries(chunk_summaries, final_len_sentences=req.desired_sentences)
        return SummaryResponse(title=os.path.basename(str(req.audio_url)), summary=final, chunk_summaries=chunk_summaries)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Media processing error: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
