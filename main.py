import os
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from google import genai

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


@app.get("/")
def home():
    return {"message": "Gemini proxy server is running"}


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
    gemini_key = os.getenv("GEMINI_API_KEY")

    if not gemini_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is missing")

    client = genai.Client(api_key=gemini_key)

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=request.message,
    )

    return {"reply": response.text}