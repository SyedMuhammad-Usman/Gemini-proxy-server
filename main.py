import json
import os
import re
from datetime import datetime
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Depends, UploadFile, File
from pydantic import BaseModel, Field, ConfigDict, ValidationError
from google import genai
from google.genai import types

load_dotenv()

app = FastAPI()

usage_count = 0
GEMINI_MODEL = "gemini-2.5-flash"


SingleClass = Literal["plastic", "paper", "metal", "e waste"]


class ChatRequest(BaseModel):
    message: str


class SingleObjectResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    class_: SingleClass = Field(alias="class")


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


def extract_json(text: str):
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


def classify_image_with_prompt(
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
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )

    return extract_json(response.text)


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
    try:
        client = get_gemini_client()

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=request.message,
        )

        return {"reply": response.text}

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/classify/single-object")
async def classify_single_object(
    image: UploadFile = File(...),
    authorized: bool = Depends(verify_friend_access),
):
    image_bytes = await read_image(image)

    prompt = """
You are a strict single-object image classification API for waste/material sorting.

Your task:
Analyze the image, identify the main visible object internally, then classify that object into exactly one of the allowed material classes.

Important:
You must first understand what the object is, but you must NOT output the object name.
Only output the final material class in JSON.

Core rules:
- Focus on ONE main object only.
- Identify the main object internally before deciding the class.
- The main object is usually the largest, clearest, most central, or most visually important object.
- Ignore background, hands, tables, floors, shadows, logos, labels, text, and secondary objects.
- Do NOT describe the object.
- Do NOT return the object name.
- Return ONLY valid JSON.
- Do NOT use markdown.
- Do NOT explain.
- Do NOT add extra keys.
- The JSON must match the schema exactly.

Allowed classes:
- e waste
- metal
- paper
- plastic

Classification rules:
- "class" must be exactly one of the allowed classes.
- Never output any class outside the allowed list.
- First identify what the object is, then map it to the closest allowed material class.
- If the object is made of paper, sticky notes, cardboard, notebook paper, books, napkins, tissues, paper cups, or other paper-based material, classify it as "paper".
- If the object is an electronic device, cable, charger, battery, circuit board, phone, keyboard, mouse, remote, appliance, or gadget, classify it as "e waste" even if plastic or metal is visible.
- If the object is a bottle, container, wrapper, bag, packaging, cap, synthetic item, or clearly plastic-based object, classify it as "plastic".
- If the object is a can, foil, tin, tool, metal container, wire, screw, or metallic object, classify it as "metal".
- If an object contains multiple materials, classify it by the dominant visible material.

Required output format:
{
  "class": "plastic"
}
"""

    try:
        raw_result = classify_image_with_prompt(
            image_bytes=image_bytes,
            mime_type=image.content_type,
            prompt=prompt,
        )

        result = SingleObjectResult.model_validate(raw_result)

        return result.model_dump(by_alias=True)

    except ValidationError as error:
        raise HTTPException(status_code=500, detail=error.errors())

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
