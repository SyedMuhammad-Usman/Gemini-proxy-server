import os
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Depends, UploadFile, File, Form
from pydantic import BaseModel
from google import genai
from google.genai import types

load_dotenv()

app = FastAPI()

usage_count = 0
GEMINI_MODEL = "gemini-2.5-pro"


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

    if request_limit > 0 and usage_count >= request_limit:
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
        "model": GEMINI_MODEL,
    }


@app.post("/chat")
def chat(
    request: ChatRequest,
    authorized: bool = Depends(verify_friend_access),
):
    gemini_key = os.getenv("GEMINI_API_KEY")

    if not gemini_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is missing")

    try:
        client = genai.Client(api_key=gemini_key)

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=request.message,
        )

        return {"reply": response.text}

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/classify-image")
async def classify_image(
    image: UploadFile = File(...),
    prompt: str = Form("Classify this image. Describe what it contains."),
    authorized: bool = Depends(verify_friend_access),
):
    gemini_key = os.getenv("GEMINI_API_KEY")

    if not gemini_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is missing")

    if image.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG, PNG, and WEBP images are allowed",
        )

    image_bytes = await image.read()

    if len(image_bytes) > 20 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="Image is too large. Max size is 20MB",
        )

    try:
        client = genai.Client(api_key=gemini_key)

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=image.content_type,
                ),
                prompt,
            ],
        )

        return {"result": response.text}

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))