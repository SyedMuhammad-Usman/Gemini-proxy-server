import json
import os
import re
from datetime import datetime
from typing import Optional

import google.auth.transport.requests
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, File, Header, HTTPException, UploadFile
from google import genai
from google.genai import types
from google.oauth2 import credentials as google_credentials
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Doctor AI – Prescription Reader API",
    description=(
        "An intelligent Medical OCR API powered by Gemini Vision. "
        "Upload a doctor's prescription image and receive structured data: "
        "medicine names, dosage timings, instructions, and the next appointment date."
    ),
    version="1.0.0",
)

# ── Config ────────────────────────────────────────────────────────────────────

usage_count = 0
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# ── Pydantic models ───────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str


class MedicineItem(BaseModel):
    """A single medicine entry extracted from a prescription image."""

    name: str = Field(
        description="Full medicine name including strength/dosage if visible (e.g. 'Panadol 500mg')."
    )
    time_to_eat: str = Field(
        description="Frequency or time of day to take the medicine (e.g. '1-0-1', 'Morning and Night', 'Once daily')."
    )
    instructions: str = Field(
        description="Any accompanying directions (e.g. 'After meals', 'Before breakfast', 'For 5 days'). 'Not mentioned' if absent."
    )


class PrescriptionResult(BaseModel):
    """Structured data extracted from a doctor's prescription image."""

    medicines: list[MedicineItem] = Field(
        description="List of all medicines prescribed."
    )
    next_appointment: Optional[str] = Field(
        default=None,
        description="Date or relative timeframe for the next follow-up visit. null if not mentioned.",
    )


# ── Auth ──────────────────────────────────────────────────────────────────────


def verify_friend_access(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
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
        if datetime.now() > expiry_time:
            raise HTTPException(status_code=403, detail="Access expired")

    if request_limit > 0 and usage_count >= request_limit:
        raise HTTPException(status_code=429, detail="Request limit reached")

    usage_count += 1
    return True


# ── Gemini client (Vertex AI via OAuth2) ─────────────────────────────────────


def _get_credentials():
    """
    Build OAuth2 credentials from GOOGLE_APPLICATION_CREDENTIALS_JSON env var.
    Supports both 'authorized_user' and 'service_account' credential types.
    """
    creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not creds_json:
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_APPLICATION_CREDENTIALS_JSON env var is missing",
        )

    try:
        creds_dict = json.loads(creds_json)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_APPLICATION_CREDENTIALS_JSON is not valid JSON",
        )

    cred_type = creds_dict.get("type")

    if cred_type == "authorized_user":
        return google_credentials.Credentials(
            token=None,
            refresh_token=creds_dict["refresh_token"],
            client_id=creds_dict["client_id"],
            client_secret=creds_dict["client_secret"],
            token_uri="https://oauth2.googleapis.com/token",
        )

    if cred_type == "service_account":
        from google.oauth2 import service_account

        return service_account.Credentials.from_service_account_info(
            creds_dict,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )

    raise HTTPException(
        status_code=500,
        detail=f"Unsupported credential type: '{cred_type}'. Expected 'authorized_user' or 'service_account'.",
    )


def get_gemini_client() -> genai.Client:
    project = os.getenv("VERTEX_PROJECT")
    location = os.getenv("VERTEX_LOCATION", "us-central1")

    if not project:
        raise HTTPException(status_code=500, detail="VERTEX_PROJECT env var is missing")

    return genai.Client(
        vertexai=True,
        credentials=_get_credentials(),
        project=project,
        location=location,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


async def read_image(image: UploadFile) -> bytes:
    if image.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG, PNG, and WEBP images are allowed",
        )

    image_bytes = await image.read()

    if len(image_bytes) > 20 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="Image is too large. Max size is 20 MB",
        )

    return image_bytes


def extract_json(text: str) -> dict:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```json", "", text)
        text = re.sub(r"^```", "", text)
        text = re.sub(r"```$", "", text)
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise HTTPException(
                status_code=500,
                detail="Model did not return valid JSON",
            )
        return json.loads(match.group(0))


def call_gemini_vision(image_bytes: bytes, mime_type: str, prompt: str) -> dict:
    client = get_gemini_client()

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            prompt,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )

    return extract_json(response.text)


# ── Core endpoints ────────────────────────────────────────────────────────────


@app.get("/", tags=["System"])
def home():
    return {"message": "Doctor AI Prescription Reader API is running"}


@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok"}


@app.get("/usage", tags=["System"])
def usage():
    return {
        "used_requests": usage_count,
        "request_limit": int(os.getenv("FRIEND_REQUEST_LIMIT", "0")),
        "expires_at": os.getenv("FRIEND_EXPIRES_AT"),
        "model": GEMINI_MODEL,
    }


@app.post("/chat", tags=["System"])
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


# ── Prescription Reader ───────────────────────────────────────────────────────

PRESCRIPTION_PROMPT = """\
[SYSTEM BEHAVIOR: DETERMINISTIC EXTRACTION & MEDICAL DECODING ENGINE]
You are an elite, deterministic Medical OCR and Data Extraction AI.
Your singular purpose is to process unstructured prescription images, use your deep medical
knowledge to identify and separate distinct elements, and output raw, strictly sequenced JSON.

[ABSOLUTE LAWS]
1. ZERO CHAT: Do not output greetings, explanations, or conclusions.
2. NO MARKDOWN: Do not use ```json or ``` blocks anywhere in your response.
3. PERFECT SYNTAX: Output must strictly begin with `{` and end with `}`.
4. STRICT SEQUENCE: For every medicine, the output JSON object MUST follow this exact key order,
   regardless of how the text is arranged in the image:
   First: "name"
   Second: "time_to_eat"
   Third: "instructions"
5. TIME TRANSLATION: You MUST convert medical shorthand into exact, numeric 12-hour clock times.
   - "B.D", "b.i.d", or "1-0-1" -> "8:00 AM and 8:00 PM"
   - "O.D" or daily -> "8:00 AM"
   - "T.D.S", "t.i.d", or "1-1-1" -> "8:00 AM, 2:00 PM, and 8:00 PM"
   - "Q.D.S" or "1-1-1-1" -> "8:00 AM, 12:00 PM, 4:00 PM, and 8:00 PM"
   Never return the raw shorthand (like "B.D") for timing. Always return the converted numeric time.
6. MISSING DATA: If any specific detail (name, timing, or instructions) is missing for a
   medication, you must output "Not found" for that exact field.

[TASK & EXTRACTION LOGIC]
Analyze the provided prescription image. Scan all visible text. Use your AI intelligence to
logically separate the medicine names, timings, and general instructions, even if the doctor
wrote them out of order.
For EVERY individual medicine found, you MUST extract exactly 3 data points in the strict sequence.
(Logic: If 1 medicine is found, extract 3 points. If 2 medicines are found, extract 6 points
total across two objects. If 3 medicines are found, extract 9 points).

[DATA SCHEMA TO EXTRACT]
1. "medicines": A list of objects representing each medication.
   - "name": The full name of the medicine (include strength if visible). If missing, "Not found".
   - "time_to_eat": The explicit numeric converted timing (e.g., "8:00 AM and 8:00 PM").
     If missing, "Not found".
   - "instructions": Accompanying directions or general doctor's advice written near the medicine.
     If missing, "Not found".

2. "next_appointment": The exact date, time, or relative timeframe for the follow-up visit.
   Return null if no follow-up is mentioned.

Required JSON Structure (Match this EXACT key sequence and format):
{
  "medicines": [
    {
      "name": "Augmintine",
      "time_to_eat": "8:00 AM and 8:00 PM",
      "instructions": "Don't use cold water"
    },
    {
      "name": "Brufen",
      "time_to_eat": "Not found",
      "instructions": "Take rest of two weeks"
    }
  ],
  "next_appointment": null
}
"""



@app.post(
    "/prescription/read",
    response_model=PrescriptionResult,
    summary="Read a Doctor's Prescription",
    description=(
        "Upload an image of a handwritten or printed doctor's prescription. "
        "The API uses Gemini Vision to perform Medical OCR and returns a structured "
        "JSON response containing:\n\n"
        "- **medicines**: list of all prescribed medicines with name, dosage timing, and instructions\n"
        "- **next_appointment**: follow-up appointment date or timeframe (`null` if not present)"
    ),
    tags=["Prescription Reader"],
)
async def read_prescription(
    image: UploadFile = File(
        ...,
        description="Prescription image. Accepted formats: JPEG, PNG, WEBP. Max size: 20 MB.",
    ),
    authorized: bool = Depends(verify_friend_access),
):
    image_bytes = await read_image(image)

    try:
        raw_result = call_gemini_vision(
            image_bytes=image_bytes,
            mime_type=image.content_type,
            prompt=PRESCRIPTION_PROMPT,
        )

        result = PrescriptionResult.model_validate(raw_result)
        return result.model_dump()

    except ValidationError as error:
        raise HTTPException(status_code=500, detail=error.errors())

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
