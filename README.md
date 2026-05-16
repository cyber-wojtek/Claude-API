# 1MinAI-API

An unofficial **async Python client** for the [1min.AI](https://1min.ai) web interface.

> **Disclaimer:** This library reverse-engineers the 1min.AI browser API. It is not affiliated with or endorsed by 1min.AI. Use responsibly and in accordance with their [terms of service](https://1min.ai/terms).

---

## Features

- **Multi-turn conversations** — stateful `ChatSession` keeps context across turns automatically
- **Streaming** — yield tokens in real time via Server-Sent Events
- **Web search grounding** — live data, toggleable per-session or per-request
- **30+ chat models** — GPT, Claude, Gemini, Grok, DeepSeek, Llama, Mistral, Qwen, and more
- **Image generation** — Flux, DALL-E, GPT-Image, Gemini, Grok, Stable Diffusion, Recraft, Leonardo, and more
- **Image editing** — upscale, background removal/replacement, inpainting, variations, face swap, sketch-to-image, 3D, object removal, text removal, outpainting
- **Text to speech** — OpenAI TTS and ElevenLabs
- **Speech to text** — Whisper, GPT-4o Transcribe, ElevenLabs, Google STT
- **Voice tools** — isolate, clone, change, and design voices; generate sound effects
- **Video generation** — Kling, Sora, Veo, Luma Ray, Wan, and more; text-to-video and image-to-video
- **Music generation** — Google Lyria, Suno, Udio, MusicGen
- **Code generation** — GPT Codex, Qwen3 Coder, Grok Code, DeepSeek Coder, Claude
- **Content tools** — grammar, paraphrase, rewrite, summarise, expand, shorten, translate, blog/email/social/ad copy, YouTube tools, document translation
- **Asset API** — upload local files or public URLs (images, PDFs, audio, video) for use across all endpoints
- **Conversation management** — create, list, fetch history, and delete server-side threads
- **Credit estimation** — pre-flight cost check before sending a request
- **Async-first** — built on `aiohttp`, zero blocking calls

---

## Related projects

### ChatAI Console

A full-featured self-hosted web UI that supports Claude, ChatWithAI.app, and 1min.AI accounts. 

**Features:**
- Multi-account management — Claude, ChatWithAI.app, and 1min.AI accounts in one place
- Real-time streaming with artifact rendering
- File upload and canvas preview panel
- Google OAuth sign-in flow
- Credit tracking and usage history
- Local conversation storage with pinning

---

## Installation

Requires Python **3.10+**.

```sh
pip install oneminai-webapi
```

---

## Authentication

### API key

1. Sign in at [app.1min.ai](https://app.1min.ai).
2. Go to **Settings → API** and copy your key.
3. Pass it to `OneMinAIClient`.

```python
client = OneMinAIClient("YOUR_API_KEY")
```

Store it in an environment variable rather than hardcoding it:

```python
import os
client = OneMinAIClient(os.environ["ONEMINAI_API_KEY"])
```

> Keep your key secret — it grants full access to your 1min.AI account and credits.

### OAuth (Google access token)

If you already have a Google OAuth2 access token (`ya29.…`) — for example obtained via a browser extension — pass it directly:

```python
client = OneMinAIClient()
user = await client.oauth_login("ya29.a0AQvPy…")
# all subsequent calls are authenticated automatically
```

The JWT returned by 1min.AI is stored internally. No API key is needed when using this flow.

---

## Quick start

```python
import asyncio
from oneminai_webapi import OneMinAIClient

async def main():
    async with OneMinAIClient("YOUR_API_KEY") as client:
        r = await client.generate_content("What is the capital of Poland?")
        print(r.text)   # "Warsaw"

asyncio.run(main())
```

The `async with` block closes the underlying HTTP session automatically. For manual lifecycle management:

```python
client = OneMinAIClient("YOUR_API_KEY")
try:
    r = await client.generate_content("Hello!")
    print(r.text)
finally:
    await client.close()
```

---

## Usage

### Single-turn generation

```python
r = await client.generate_content("Translate 'hello' to Japanese.")
print(r.text)    # "こんにちは"
print(r.model)   # "gpt-4.1-nano"
print(str(r))    # same as r.text
```

Attach images or documents:

```python
asset = await client.upload_asset("chart.png")
r = await client.generate_content(
    "Describe what you see in this chart.",
    images=[asset.asset_key],
)

doc = await client.upload_asset("report.pdf")
r2 = await client.generate_content(
    "Summarise this document.",
    files=[doc.file_id],
)
```

---

### Streaming

```python
async for chunk in client.generate_content_stream("Write me a haiku."):
    print(chunk.text_delta, end="", flush=True)
print()
```

---

### Multi-turn chat

`start_chat()` returns a `ChatSession` that tracks context automatically:

```python
chat = client.start_chat()

r1 = await chat.send_message("My name is Alice.")
r2 = await chat.send_message("What is my name?")   # → "Your name is Alice."
print(r2.text)
```

Streaming within a session:

```python
async for chunk in chat.send_message_stream("Write a short poem about it."):
    print(chunk.text_delta, end="", flush=True)
print()

# Context persists across streaming and non-streaming turns
r3 = await chat.send_message("Now explain it line by line.")
```

---

### Web search

```python
# Enable for every turn in the session
chat = client.start_chat(web_search=True, num_of_site=5)
r = await chat.send_message("What AI models were released this week?")

# Or enable per-request only
chat2 = client.start_chat()
r2 = await chat2.send_message("Latest Python release?", web_search=True)
```

---

### Model selection

```python
from oneminai_webapi.constants import ChatModel

# Via enum
r = await client.generate_content("Hello!", model=ChatModel.CLAUDE_SONNET_4_6)

# Via raw string — any model ID from the catalog works
r2 = await client.generate_content("Hello!", model="gemini-2.5-flash")

# Session-level default
chat = client.start_chat(model=ChatModel.GPT_4O)
```

Available chat models (partial list — use `list_models()` for the full live catalog):

| Enum | Model ID |
|---|---|
| `ChatModel.GPT_4_1_NANO` | `gpt-4.1-nano` |
| `ChatModel.GPT_4_1_MINI` | `gpt-4.1-mini` |
| `ChatModel.GPT_4_1` | `gpt-4.1` |
| `ChatModel.GPT_4O` | `gpt-4o` |
| `ChatModel.GPT_4O_MINI` | `gpt-4o-mini` |
| `ChatModel.GPT_5` | `gpt-5` |
| `ChatModel.O3` | `o3` |
| `ChatModel.O4_MINI` | `o4-mini` |
| `ChatModel.CLAUDE_HAIKU_4_5` | `claude-haiku-4-5-20251001` |
| `ChatModel.CLAUDE_SONNET_4_6` | `claude-sonnet-4-6` |
| `ChatModel.CLAUDE_OPUS_4_6` | `claude-opus-4-6` |
| `ChatModel.GEMINI_2_5_FLASH` | `gemini-2.5-flash` |
| `ChatModel.GEMINI_2_5_PRO` | `gemini-2.5-pro` |
| `ChatModel.GROK_3` | `grok-3` |
| `ChatModel.GROK_4` | `grok-4-0709` |
| `ChatModel.DEEPSEEK_CHAT` | `deepseek-chat` |
| `ChatModel.DEEPSEEK_REASONER` | `deepseek-reasoner` |
| `ChatModel.LLAMA_4_SCOUT` | `meta/llama-4-scout-instruct` |
| `ChatModel.MISTRAL_LARGE` | `mistral-large-latest` |
| `ChatModel.SONAR_PRO` | `sonar-pro` |

---

### Persistent server-side conversations

```python
# Create a named thread
conv = await client.create_conversation("Research session")

# Chat within it — history is stored server-side
r = await client.chat(
    "What is quantum entanglement?",
    conversation_id=conv.conversation_id,
)

# Fetch full message history
messages = await client.get_conversation_messages(conv.conversation_id)
for msg in messages:
    print(f"[{msg.role}]: {msg.content}")

# List all threads
all_convs = await client.list_conversations()

# Delete when done
await client.delete_conversation(conv.conversation_id)
```

---

### Image generation

```python
from oneminai_webapi.constants import ImageModel

result = await client.generate_image(
    "A serene mountain lake at golden hour, photorealistic",
    model=ImageModel.FLUX_1_1_PRO,
    width=1024,
    height=1024,
    num_images=2,
)

print(result.image.url)          # first image URL
for img in result.images:
    await img.save("./outputs")  # download all to disk
```

---

### Image editing

```python
asset = await client.upload_asset("photo.jpg")

upscaled  = await client.upscale_image(asset.asset_key, scale=2)
no_bg     = await client.remove_background(asset.asset_key)
new_bg    = await client.replace_background(asset.asset_key, "A sunny beach at sunset")
edited    = await client.edit_image(asset.asset_key, "Make the sky purple")
inpainted = await client.inpaint_image(asset.asset_key, mask.asset_key, "A red rose")
extended  = await client.extend_image(asset.asset_key, direction="right")
variants  = await client.create_image_variation(asset.asset_key, num_images=4)
swapped   = await client.swap_face(face_img.asset_key, target_img.asset_key)
rendered  = await client.sketch_to_image(sketch.asset_key, "A cozy cottage")
model_3d  = await client.image_to_3d(asset.asset_key)
clean     = await client.remove_text_from_image(asset.asset_key)
replaced  = await client.search_and_replace_in_image(asset.asset_key, "red car", "blue bicycle")

prompt    = await client.image_to_prompt(asset.asset_key)
print(prompt.text)
```

---

### Text to speech

```python
from oneminai_webapi.constants import TTSModel, TTSVoice

audio = await client.text_to_speech(
    "Hello, this is a test.",
    model=TTSModel.OPENAI_TTS_1_HD,
    voice=TTSVoice.NOVA,
    speed=1.0,
)
await audio.save("./outputs", verbose=True)

# ElevenLabs — pass voice name as a plain string
el = await client.text_to_speech(
    "Sounds natural.",
    model=TTSModel.ELEVENLABS,
    voice="Rachel",
)

# Sound effects
sfx = await client.text_to_sound(
    "Heavy rain on a tin roof, distant thunder",
    duration=10.0,
)
```

---

### Speech to text

```python
from oneminai_webapi.constants import STTModel

asset = await client.upload_asset("recording.mp3")

result    = await client.speech_to_text(asset.asset_key, model=STTModel.WHISPER_1)
print(result.text)

translated = await client.translate_audio(asset.asset_key, "English")
captions   = await client.generate_captions(asset.asset_key)
```

---

### Voice tools

```python
clean    = await client.isolate_voice(audio.asset_key)
changed  = await client.change_voice(audio.asset_key, target_voice="Josh")
cloned   = await client.clone_voice(audio.asset_key, name="MyVoice")
designed = await client.design_voice(
    "A calm, middle-aged British woman with a warm tone",
    sample_text="Hello, how can I help you today?",
)
```

---

### Video generation

```python
from oneminai_webapi.constants import VideoModel

video = await client.generate_video(
    "Drone shot rising over snow-capped mountains at dawn",
    model=VideoModel.KLING,
    duration=5,
    aspect_ratio="16:9",
)
await video.save("./outputs")

asset = await client.upload_asset("frame.jpg")
vid2  = await client.image_to_video(
    asset.asset_key,
    prompt="Camera slowly pans left, leaves gently swaying",
    model=VideoModel.LUMA,
)

swapped = await client.swap_face_in_video(video.video_url, face_img.asset_key)
```

---

### Music generation

```python
from oneminai_webapi.constants import MusicModel

track = await client.generate_music(
    "Upbeat lo-fi hip hop, 90 BPM, warm piano chords",
    model=MusicModel.LYRIA_2,
)
await track.save("./outputs")

score = await client.generate_music(
    "Epic cinematic orchestral, building tension",
    instrumental=True,
    duration=30.0,
)
```

---

### Code generation

```python
from oneminai_webapi.constants import CodeModel

result = await client.generate_code(
    "Async HTTP client with exponential-backoff retry using aiohttp.",
    model=CodeModel.CLAUDE_SONNET_4_6,
)
print(result.text)

# With live web search to reference latest docs
result2 = await client.generate_code(
    "FastAPI endpoint validating JWT with PyJWT 2.x.",
    web_search=True,
)
```

---

### Content tools

```python
# Text editing
corrected   = await client.check_grammar("its a beautifull day!")
paraphrased = await client.paraphrase("Text to rewrite.", tone="casual")
rewritten   = await client.rewrite("Formal text.", tone="friendly")
summary     = await client.summarize(long_article)
expanded    = await client.expand_content("Short stub.")
shortened   = await client.shorten_content(verbose_text)

# Translation & detection
polish      = await client.translate("Good morning", "Polish")
detection   = await client.detect_ai_content(suspicious_text)

# SEO
keywords    = await client.research_keywords("async Python AI")

# Content generation
article     = await client.generate_blog_article("AI trends in 2025")
email       = await client.generate_email("Partnership proposal to Acme Corp")
reply       = await client.generate_email_reply(original_email, "Accept politely")
tweet       = await client.generate_social_post("Launch day!", platform="x")
linkedin    = await client.generate_social_post("We shipped v1!", platform="linkedin")
fb_ad       = await client.generate_ad_copy("Fast async Python SDK", platform="facebook")
slides      = await client.generate_presentation("The future of AI in 2026")

# YouTube
yt_summary  = await client.summarize_youtube("https://youtu.be/dQw4w9WgXcQ")
yt_script   = await client.transcribe_youtube("https://youtu.be/dQw4w9WgXcQ")
yt_polish   = await client.translate_youtube("https://youtu.be/dQw4w9WgXcQ", "Polish")

# Document translation
doc         = await client.upload_asset("contract.pdf")
translated  = await client.translate_document(doc.file_id, "Spanish")
```

---

### Asset upload

```python
import oneminai_webapi

# Local file
asset = await client.upload_asset(
    "report.pdf",
    asset_type=oneminai_webapi.AssetType.DOCUMENT,
)

# Raw bytes
asset2 = await client.upload_asset(
    data=some_bytes,
    filename="image.png",
    mime_type="image/png",
    asset_type=oneminai_webapi.AssetType.IMAGE,
)

# Public URL
asset3 = await client.upload_asset_from_url("https://example.com/photo.jpg")

# Use in chat
reply = await client.generate_content(
    "Summarise this document.",
    files=[asset.file_id],      # file_id for documents in chat
)

# Use in image APIs
upscaled = await client.upscale_image(asset2.asset_key)  # asset_key for media
```

`AssetRecord` fields:

| Field | Description |
|---|---|
| `asset_key` | Key for image / video / audio API calls |
| `file_id` | UUID for `files=[]` in chat attachments |
| `asset_type` | `"DOCUMENT"`, `"IMAGE"`, `"AUDIO"`, `"VIDEO"` |

---

### Account & credits

```python
user = await client.get_current_user()
print(user.email, user.plan, user.credit)

balance = await client.get_team_credits()

estimate = await client.estimate_chat_cost(
    "Write a long essay about quantum computing.",
    models=["gpt-4o", "claude-sonnet-4-6"],
    web_search=True,
)
print(f"Estimated: {estimate.total_estimated_credit} credits")

models = await client.list_models(feature="UNIFY_CHAT_WITH_AI")
```

---

## Logging

```python
import oneminai_webapi
oneminai_webapi.set_log_level("DEBUG")   # DEBUG | INFO | WARNING | ERROR
```

---

## Error handling

```python
from oneminai_webapi.exceptions import (
    AuthenticationError,   # 401 — invalid or missing API key / token
    RateLimitError,        # 429 — too many requests
    ValidationError,       # 422 — bad request payload
    APIError,              # any other 4xx / 5xx
    AssetUploadError,      # file upload failed
    OAuthError,            # OAuth login failed
)

try:
    r = await client.generate_content("Hello!")
except AuthenticationError:
    print("Invalid API key.")
except RateLimitError as e:
    print(f"Rate limited — retry in {e.retry_after_s}s")
except ValidationError as e:
    print(f"Validation error: {e.details}")
except APIError as e:
    print(f"API error {e.status_code}: {e}")
```

---

## API reference

### `OneMinAIClient`

#### Chat

| Method | Description |
|---|---|
| `generate_content(prompt, ...)` | Single-turn, full response |
| `generate_content_stream(prompt, ...)` | Single-turn, streamed |
| `chat(prompt, *, stream, ...)` | Low-level chat primitive |
| `start_chat(model, ...)` | Create a `ChatSession` |

#### Conversations

| Method | Description |
|---|---|
| `create_conversation(title)` | Create a server-side thread |
| `list_conversations()` | List all threads |
| `get_conversation(conv_id)` | Fetch thread metadata |
| `get_conversation_messages(conv_id)` | Fetch message history |
| `delete_conversation(conv_id)` | Delete a thread |

#### Images

| Method | Description |
|---|---|
| `generate_image(prompt, ...)` | Text-to-image |
| `edit_image(image_url, prompt, ...)` | Edit with instruction |
| `create_image_variation(image_url, ...)` | Generate variations |
| `extend_image(image_url, direction, ...)` | Outpaint / extend canvas |
| `inpaint_image(image_url, mask_url, prompt, ...)` | Inpaint masked region |
| `upscale_image(image_url, scale, ...)` | Upscale 2× or 4× |
| `remove_background(image_url, ...)` | Background removal |
| `replace_background(image_url, prompt, ...)` | Background replacement |
| `remove_object_from_image(image_url, ...)` | Object removal |
| `remove_text_from_image(image_url)` | Text overlay removal |
| `swap_face(source_url, target_url, ...)` | Face swap |
| `sketch_to_image(sketch_url, prompt, ...)` | Sketch → image |
| `image_to_3d(image_url, ...)` | Image → 3D model |
| `image_to_prompt(image_url, ...)` | Reverse-engineer prompt |
| `search_and_replace_in_image(image_url, ...)` | Find & replace object |

#### Audio

| Method | Description |
|---|---|
| `text_to_speech(text, ...)` | TTS |
| `speech_to_text(audio_url, ...)` | STT / transcription |
| `translate_audio(audio_url, language, ...)` | Transcribe + translate |
| `generate_captions(audio_or_video_url, ...)` | Captions / subtitles |
| `text_to_sound(prompt, ...)` | Sound effect generation |
| `isolate_voice(audio_url)` | Voice isolation |
| `change_voice(audio_url, target_voice)` | Voice conversion |
| `clone_voice(audio_url, name, ...)` | Voice cloning |
| `design_voice(description, ...)` | Voice design |

#### Video

| Method | Description |
|---|---|
| `generate_video(prompt, ...)` | Text-to-video |
| `image_to_video(image_url, ...)` | Image-to-video |
| `swap_face_in_video(video_url, face_url)` | Face swap in video |

#### Music

| Method | Description |
|---|---|
| `generate_music(prompt, ...)` | Music generation |

#### Code & content

| Method | Description |
|---|---|
| `generate_code(prompt, ...)` | Code generation |
| `check_grammar(text)` | Grammar correction |
| `paraphrase(text, ...)` | Paraphrase |
| `rewrite(text, ...)` | Rewrite |
| `summarize(text, ...)` | Summarisation |
| `expand_content(text, ...)` | Content expansion |
| `shorten_content(text, ...)` | Content shortening |
| `translate(text, target_language, ...)` | Translation |
| `detect_ai_content(text)` | AI content detection |
| `research_keywords(topic, ...)` | SEO keyword research |
| `generate_blog_article(prompt, ...)` | Blog article |
| `generate_email(prompt, ...)` | Email |
| `generate_email_reply(original, instructions, ...)` | Email reply |
| `generate_social_post(prompt, platform, ...)` | Social media post |
| `generate_social_comment(post_text, platform, ...)` | Social comment |
| `generate_ad_copy(prompt, platform, ...)` | Ad copy |
| `generate_presentation(prompt, ...)` | Presentation |
| `summarize_youtube(url, ...)` | YouTube summary |
| `transcribe_youtube(url, ...)` | YouTube transcript |
| `translate_youtube(url, language, ...)` | YouTube translation |
| `translate_document(file_id, language, ...)` | Document translation |

#### Assets

| Method | Description |
|---|---|
| `upload_asset(file_path, ...)` | Upload local file |
| `upload_asset_from_url(url, ...)` | Register public URL |

#### Account

| Method | Description |
|---|---|
| `get_current_user()` | User and team info |
| `get_team_credits()` | Credit balance |
| `estimate_chat_cost(prompt, models, ...)` | Pre-flight cost estimate |
| `list_models(feature)` | Model catalog |
| `list_team_members()` | Team members |
| `get_feature_settings(feature)` | Feature settings |
| `update_feature_settings(models, feature)` | Update feature settings |
| `update_user_settings(**settings)` | Update user settings |
| `get_notebook()` | User notebook |
| `get_unread_notification_count()` | Unread notification count |
| `get_notification(notification_id)` | Fetch a notification |
| `get_explore_posts()` | Public explore feed |
| `list_tags()` | Conversation tags |
| `list_studios()` | Team studios |
| `oauth_login(token)` | Authenticate with Google OAuth |

---

### `ChatSession`

| Method / Property | Description |
|---|---|
| `send_message(prompt, ...)` | Send a message, await full reply |
| `send_message_stream(prompt, ...)` | Send a message, stream reply |
| `delete()` | Delete the server-side conversation |
| `.conversation_id` | Server UUID of this thread |
| `.model` | Model used by this session |

---

### Return types

| Type | Key fields |
|---|---|
| `ChatOutput` | `text`, `text_delta`, `model`, `conversation_id`, `record_id` |
| `ImageOutput` | `images: list[GeneratedImage]`, `image` (first), `model`, `record_id` |
| `GeneratedImage` | `url`, `model`; `.save(dir)` downloads to disk |
| `AudioOutput` | `audio_url`, `model`, `record_id`; `.save(dir)` |
| `MusicOutput` | `audio_url`, `model`, `record_id`; `.save(dir)` |
| `VideoOutput` | `video_url`, `model`, `record_id`; `.save(dir)` |
| `TranscriptionOutput` | `text`, `model`, `record_id` |
| `AssetRecord` | `asset_key`, `file_id`, `asset_type` |
| `UserRecord` | `user_id`, `email`, `team_id`, `team_name`, `credit`, `plan` |
| `ConversationRecord` | `conversation_id`, `title` |
| `MessageRecord` | `role`, `content`, `record_id`, `credit`, `execution_time` |
| `CreditEstimate` | `models`, `total_input_tokens`, `total_estimated_credit` |

---

## Development

```sh
git clone https://github.com/cyber-wojtek/1MinAI-API
cd 1MinAI-API
pip install -e ".[dev]"

export ONEMINAI_API_KEY="your-key"
python examples.py
```

---

## References

- [1min.AI](https://1min.ai)
- [1min.AI app](https://app.1min.ai)
- [claude-webapi](https://github.com/cyber-wojtek/Claude-API) — sister project for Claude.ai
- [gemini_webapi](https://github.com/HanaokaYuzu/Gemini-API) — inspiration for the interface design