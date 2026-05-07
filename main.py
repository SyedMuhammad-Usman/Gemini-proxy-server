import os
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Depends, UploadFile, File
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


def get_gemini_client():
    gemini_key = os.getenv("GEMINI_API_KEY")

    if not gemini_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is missing")

    return genai.Client(api_key=gemini_key)


async def read_image(image: UploadFile):
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

    return image_bytes


def extract_text_from_image(
    image_bytes: bytes,
    mime_type: str,
    prompt: str,
):
    client = get_gemini_client()

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type,
            ),
            prompt,
        ],
        # Removed JSON formatting constraints to allow raw text output
    )

    return response.text.strip()


@app.get("/")
def home():
    return {"message": "Gemini Prescription OCR proxy server is running"}


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
    try:
        client = get_gemini_client()

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=request.message,
        )

        return {"reply": response.text}

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/extract-prescription")
async def extract_prescription_text(
    image: UploadFile = File(...),
    authorized: bool = Depends(verify_friend_access),
):
    image_bytes = await read_image(image)

    # High-performance OCR Prompt tailored for medical prescriptions
    prompt = """
You are an expert Medical Transcriptionist and advanced OCR (Optical Character Recognition) system.
Your sole task is to extract every piece of text from the provided medical prescription image EXACTLY as it appears.

CRITICAL INSTRUCTIONS FOR FORMATTING AND EXTRACTION:
1. SPATIAL PRESERVATION: You MUST preserve the exact visual layout of the text. 
   - Use newlines (Enter) to match line breaks in the document.
   - Use spaces and indentation to visually align text, dosages, and instructions exactly as written.
2. MEDICAL ACCURACY: Pay extremely close attention to:
   - Patient Information (Name, Age, Date, Vitals).
   - Medication Names (Spell them to the best of your ability, even if handwritten).
   - Dosages (e.g., 500mg, 10ml, 1 tablet).
   - Frequencies / Medical Abbreviations (e.g., OD, BD, BID, TID, SOS, x 5 days).
   - Doctor's Special Instructions (e.g., "Take after meals", "Empty stomach").
3. NO HALLUCINATION: Transcribe ONLY what is visible. If a word is completely illegible due to poor handwriting, write "[illegible]" in its place. Do not guess medications if you are entirely unsure.
4. NO CONVERSATIONAL TEXT: Do not include introductory or concluding remarks like "Here is the extracted text" or "I found the following". 
5. OUTPUT: Output pure, formatted, raw text.

Transcribe the prescription now:
"""

    try:
        extracted_text = extract_text_from_image(
            image_bytes=image_bytes,
            mime_type=image.content_type,
            prompt=prompt,
        )

        # Returning the plain formatted string inside a standard JSON response wrapper
        return {"extracted_text": extracted_text}

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
