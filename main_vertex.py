import os
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
import vertexai
from vertexai.generative_models import GenerativeModel

load_dotenv()

app = FastAPI()

usage_count = 0


class ChatRequest(BaseModel):
    message: str


def verify_friend_access(
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    global usage_count

    friend_api_key = os.getenv("FRIEND_API_KEY")
    expires_at = os.getenv("FRIEND_EXPIRES_AT")
    request_limit = int(os.getenv("FRIEND_REQUEST_LIMIT", "0"))

    if not friend_api_key:
        raise HTTPException(status_code=500, detail="FRIEND_API_KEY is missing")

    if x_api_key != friend_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if expires_at:
        expiry_time = datetime.fromisoformat(expires_at)
        now = datetime.now()

        if now > expiry_time:
            raise HTTPException(status_code=403, detail="Access expired")

    if usage_count >= request_limit:
        raise HTTPException(status_code=429, detail="Request limit reached")

    usage_count += 1

    return True


def get_vertex_model() -> GenerativeModel:
    """
    Initialise Vertex AI and return a GenerativeModel.

    Required env vars:
        VERTEX_PROJECT   – your GCP project ID
        VERTEX_LOCATION  – region, e.g. "us-central1"
        VERTEX_MODEL     – model name, e.g. "gemini-2.0-flash-001"

    Authentication is handled automatically via:
        • Application Default Credentials (ADC) on Cloud Run / GCE
        • GOOGLE_APPLICATION_CREDENTIALS env var pointing to a service-account JSON
    """
    project = os.getenv("VERTEX_PROJECT")
    location = os.getenv("VERTEX_LOCATION", "us-central1")
    model_name = os.getenv("VERTEX_MODEL", "gemini-2.0-flash-001")

    if not project:
        raise HTTPException(status_code=500, detail="VERTEX_PROJECT env var is missing")

    vertexai.init(project=project, location=location)
    return GenerativeModel(model_name)


@app.get("/")
def home():
    return {"message": "Gemini proxy server (Vertex AI) is running"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/usage")
def usage():
    return {
        "used_requests": usage_count,
        "request_limit": int(os.getenv("FRIEND_REQUEST_LIMIT", "0")),
        "expires_at": os.getenv("FRIEND_EXPIRES_AT"),
    }


@app.post("/chat")
def chat(
    request: ChatRequest,
    authorized: bool = Depends(verify_friend_access),
):
    model = get_vertex_model()

    response = model.generate_content(request.message)

    return {"reply": response.text}
