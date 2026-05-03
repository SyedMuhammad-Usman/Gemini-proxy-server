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


SingleClass = Literal["waste", "glass", "metal", "plastic", "textile", "wood"]
MultiClass = Literal["paper", "biodegradable", "plastic", "glass", "metal", "cardboard"]
FoodClass = Literal["waste", "non_waste"]


class ChatRequest(BaseModel):
    message: str


class SingleObjectResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    confidence: int = Field(ge=0, le=100)
    class_: SingleClass = Field(alias="class")
    material: SingleClass
    recyclable: bool


class DetectedObject(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    confidence: int = Field(ge=0, le=100)
    class_: MultiClass = Field(alias="class")
    material: MultiClass


class MultipleObjectResult(BaseModel):
    object_count: int
    objects: list[DetectedObject]


class FoodResult(BaseModel):
    food: FoodClass
    confidence: int = Field(ge=0, le=100)


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
You are an image classification API.

Task:
Classify the main single object in the image.

Rules:
- Return ONLY valid JSON.
- Do not use markdown.
- Do not explain.
- Use only these classes:
  waste, glass, metal, plastic, textile, wood
- "class" and "material" must be one of those exact values.
- "recyclable" must be true or false.
- "confidence" must be an honest visual confidence estimate from 0 to 100.
- Do not return object name.
- Do not add extra keys.

JSON schema:
{
  "confidence": 80,
  "class": "plastic",
  "material": "plastic",
  "recyclable": true
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


@app.post("/classify/multiple-objects")
async def classify_multiple_objects(
    image: UploadFile = File(...),
    authorized: bool = Depends(verify_friend_access),
):
    image_bytes = await read_image(image)

    prompt = """
You are a multiple object detection and classification API.

Task:
Detect visible objects in the image and classify each one.

Rules:
- Return ONLY valid JSON.
- Do not use markdown.
- Do not explain.
- Detect up to 10 clear objects.
- Use only these classes:
  paper, biodegradable, plastic, glass, metal, cardboard
- For every detected object, "class" and "material" must be one of those exact values.
- "object_count" must equal the number of objects in the objects array.
- "confidence" must be an honest visual confidence estimate from 0 to 100.
- Do not return object names.
- Do not add extra keys.

JSON schema:
{
  "object_count": 2,
  "objects": [
    {
      "confidence": 82,
      "class": "plastic",
      "material": "plastic"
    },
    {
      "confidence": 76,
      "class": "cardboard",
      "material": "cardboard"
    }
  ]
}
"""

    try:
        raw_result = classify_image_with_prompt(
            image_bytes=image_bytes,
            mime_type=image.content_type,
            prompt=prompt,
        )

        result = MultipleObjectResult.model_validate(raw_result)

        final_result = result.model_dump(by_alias=True)
        final_result["object_count"] = len(final_result["objects"])

        return final_result

    except ValidationError as error:
        raise HTTPException(status_code=500, detail=error.errors())

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/classify/food")
async def classify_food(
    image: UploadFile = File(...),
    authorized: bool = Depends(verify_friend_access),
):
    image_bytes = await read_image(image)

    prompt = """
You are a food waste classification API.

Task:
Classify the food in the image as waste or non_waste.

Rules:
- Return ONLY valid JSON.
- Do not use markdown.
- Do not explain.
- "food" must be exactly one of:
  waste, non_waste
- Use "waste" if food looks spoiled, thrown away, dirty, rotten, leftover waste, or not usable.
- Use "non_waste" if food looks fresh, clean, edible, packaged, or usable.
- "confidence" must be an honest visual confidence estimate from 0 to 100.
- Do not add extra keys.

JSON schema:
{
  "food": "waste",
  "confidence": 80
}
"""

    try:
        raw_result = classify_image_with_prompt(
            image_bytes=image_bytes,
            mime_type=image.content_type,
            prompt=prompt,
        )

        result = FoodResult.model_validate(raw_result)

        return result.model_dump()

    except ValidationError as error:
        raise HTTPException(status_code=500, detail=error.errors())

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))