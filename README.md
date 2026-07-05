# 🤖 Gemini Proxy Server

A lightweight **FastAPI** proxy that lets you share access to Google Gemini with a friend — **without exposing your real API key**. You control the rate limit, expiry date, and the friend-facing key entirely through environment variables.

Two backends are supported:

| File | Backend |
|---|---|
| `main.py` | Google AI Studio (`google-genai` SDK — direct API key) |
| `main_vertex.py` | Google Vertex AI (`google-cloud-aiplatform` SDK — ADC / Service Account) |

---

## ✨ Features

- 🔑 **Friend-facing API key** — share a custom key, not your real one
- ⏰ **Expiry date** — key stops working after a set date/time
- 📊 **Request limit** — cap how many calls can be made
- 📈 **Usage endpoint** — see how many requests have been used
- 🔒 **No key leakage** — your real Gemini / GCP credentials never leave the server

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install uv        # if you don't have it
uv sync
```

Or with plain pip:

```bash
pip install fastapi[standard] google-genai google-cloud-aiplatform python-dotenv
```

### 2. Create your `.env` file

```bash
cp .env.example .env
# then edit .env with your values
```

`.env` for **`main.py`** (direct Gemini API):

```env
GEMINI_API_KEY=your_real_gemini_api_key_here

FRIEND_API_KEY=some-random-secret-key-you-share-with-your-friend
FRIEND_EXPIRES_AT=2026-12-31T23:59:59
FRIEND_REQUEST_LIMIT=500
```

`.env` for **`main_vertex.py`** (Vertex AI):

```env
VERTEX_PROJECT=your-gcp-project-id
VERTEX_LOCATION=us-central1
VERTEX_MODEL=gemini-2.0-flash-001

FRIEND_API_KEY=some-random-secret-key-you-share-with-your-friend
FRIEND_EXPIRES_AT=2026-12-31T23:59:59
FRIEND_REQUEST_LIMIT=500
```

> **Vertex AI Auth:** On Cloud Run / GCE the server automatically uses Application Default Credentials (ADC). Locally, either run `gcloud auth application-default login` or point `GOOGLE_APPLICATION_CREDENTIALS` to your service-account JSON file.

### 3. Run the server

```bash
# Direct Gemini API
uvicorn main:app --reload

# Vertex AI
uvicorn main_vertex:app --reload
```

---

## 📡 API Reference

All requests (except `/` and `/health`) require the friend's key in the header:

```
X-API-Key: <FRIEND_API_KEY>
```

### `GET /`
Returns a simple status message.

### `GET /health`
Health check — returns `{"status": "ok"}`.

### `GET /usage`
Returns current usage stats (no auth required).

```json
{
  "used_requests": 42,
  "request_limit": 500,
  "expires_at": "2026-12-31T23:59:59"
}
```

### `POST /chat`
Send a message to Gemini.

**Request body:**
```json
{
  "message": "Hello, what is the capital of France?"
}
```

**Response:**
```json
{
  "reply": "The capital of France is Paris."
}
```

**cURL example:**
```bash
curl -X POST https://your-deployed-url/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-friend-api-key" \
  -d '{"message": "Tell me a joke"}'
```

---

## 🌐 Deploying

### Railway / Render / Fly.io

1. Push this repo to GitHub
2. Connect the repo in the platform dashboard
3. Set all environment variables in the platform's settings UI (never commit `.env`)
4. Set the start command to:
   ```
   uvicorn main:app --host 0.0.0.0 --port $PORT
   ```
   or for Vertex:
   ```
   uvicorn main_vertex:app --host 0.0.0.0 --port $PORT
   ```

---

## 🔐 Security Notes

- `.env` is in `.gitignore` — your real API keys are **never committed**
- Your friend only ever sees their `FRIEND_API_KEY`, not your `GEMINI_API_KEY` or GCP credentials
- Rotate `FRIEND_API_KEY` anytime by updating the env var and redeploying

---

## 📄 License

MIT
