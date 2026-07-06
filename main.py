import base64
import json
import os
import re
from datetime import datetime
from typing import Literal

import google.auth.transport.requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Depends, UploadFile, File
from google.oauth2 import credentials as google_credentials
from pydantic import BaseModel, Field, ConfigDict, ValidationError
from google import genai
from google.genai import types

load_dotenv()

app = FastAPI()

usage_count = 0
# Model: can be overridden via GEMINI_MODEL env var
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")


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


def _get_access_token() -> str:
    """
    Get a valid short-lived OAuth2 Bearer token from credentials stored in
    GOOGLE_APPLICATION_CREDENTIALS_JSON env var.

    Supports both:
      - 'authorized_user' (from: gcloud auth application-default login)
      - 'service_account'  (from: GCP Console → IAM → Service Accounts → Keys)
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
        raise HTTPException(status_code=500, detail="GOOGLE_APPLICATION_CREDENTIALS_JSON is not valid JSON")

    cred_type = creds_dict.get("type")

    if cred_type == "authorized_user":
        creds = google_credentials.Credentials(
            token=None,
            refresh_token=creds_dict["refresh_token"],
            client_id=creds_dict["client_id"],
            client_secret=creds_dict["client_secret"],
            token_uri="https://oauth2.googleapis.com/token",
        )
    elif cred_type == "service_account":
        from google.oauth2 import service_account
        creds = service_account.Credentials.from_service_account_info(
            creds_dict,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
    else:
        raise HTTPException(
            status_code=500,
            detail=f"Unsupported credential type: '{cred_type}'. Expected 'authorized_user' or 'service_account'.",
        )

    # The SDK handles the refresh itself, so we just return the creds object
    return creds


def get_gemini_client() -> genai.Client:
    """
    Get the official Gemini client using the OAuth2 credentials.
    This safely handles Vertex AI with User Credentials.
    """
    project = os.getenv("VERTEX_PROJECT")
    location = os.getenv("VERTEX_LOCATION", "us-central1")

    if not project:
        raise HTTPException(status_code=500, detail="VERTEX_PROJECT env var is missing")

    creds = _get_access_token()

    return genai.Client(
        vertexai=True,
        credentials=creds,
        project=project,
        location=location,
    )


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
) -> dict:
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
- glass
- metal
- plastic
- textile
- wood

Classification rules:
- "class" must be exactly one of the allowed classes.
- "material" must be exactly the same value as "class".
- Never output any class outside the allowed list.
- First identify what the object is, then map it to the closest allowed material class.
- If the object is made of paper, sticky notes, cardboard-like paper, notebook paper, books, napkins, tissues, paper cups, or other paper-based material, classify it as "wood" because paper comes from wood and there is no separate paper class.
- If the object is made of wood, plywood, bamboo, paper, or paper-derived material, classify it as "wood".
- If the object is an electronic device, cable, charger, battery, circuit board, phone, keyboard, mouse, remote, appliance, or gadget, classify it as "e waste" even if plastic or metal is visible.
- If the object is a bottle, container, wrapper, bag, packaging, cap, synthetic item, or clearly plastic-based object, classify it as "plastic".
- If the object is a glass bottle, jar, cup, window piece, mirror piece, or transparent/reflective glass object, classify it as "glass".
- If the object is a can, foil, tin, tool, metal container, wire, screw, or metallic object, classify it as "metal".
- If the object is clothing, fabric, cloth, towel, rope, carpet, bag made of fabric, or soft woven material, classify it as "textile".
- If an object contains multiple materials, classify it by the dominant visible material.
- If uncertain, choose the most likely class based on visual evidence and lower the confidence.

Recyclability rules:
- Set "recyclable": true if the material is commonly recyclable or recoverable.
- Set "recyclable": false if the object appears contaminated, dirty, mixed in a non-recyclable way, or unlikely to be accepted in standard recycling.
- For "e waste", use true because it is recyclable through specialized e-waste recycling.
- For clean glass, metal, plastic, wood, or textile, use true when visually reasonable.
- Use false when the visual condition suggests it should not be recycled.

Confidence rules:
- "confidence" must be an integer from 0 to 100.
- Estimate confidence honestly from object clarity and material certainty.
- Use 90-100 when the object and material are obvious.
- Use 70-89 when mostly clear but not perfect.
- Use 40-69 when partially unclear, obstructed, mixed-material, or ambiguous.
- Use below 40 only when the image is very unclear.

Required output format:
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
You are a strict multiple-object image detection and material classification API for waste/material sorting.

Your task:
Analyze the image, detect the clear visible objects, identify each object internally, then classify each detected object into exactly one of the allowed material classes.

Important:
You must first understand what each object is, but you must NOT output object names.
Only output the final material class for each object in JSON.

Core detection rules:
- Detect up to 10 clear visible objects only.
- Count each separate visible object as one object.
- If there are more than 10 objects, choose the 10 largest, clearest, most central, or most visually important objects.
- Ignore background, hands, tables, floors, walls, shadows, reflections, logos, printed text, and tiny unclear fragments.
- Do NOT return object names.
- Do NOT describe the objects.
- Do NOT include bounding boxes.
- Do NOT explain.
- Return ONLY valid JSON.
- Do NOT use markdown.
- Do NOT add extra keys.
- The JSON must match the schema exactly.

Allowed classes:
- paper/wood
- biodegradable
- plastic
- glass
- metal
- cardboard

Classification rules:
- For every object, "class" must be exactly one of the allowed classes.
- For every object, "material" must be exactly the same value as "class".
- Never output any class outside the allowed list.
- First identify what each object is, then map it to the closest allowed material class.
- Classify each object by its dominant visible material.
- If an object has mixed materials, choose the material that appears most visually dominant.

Material mapping rules:
- Use "paper/wood" for paper, sticky notes, sheets, newspapers, books, notebooks, receipts, napkins, tissues, wooden items, bamboo items, plywood, paper cups, and general paper-based or wood-based objects.
- Use "cardboard" only for cardboard boxes, cartons, corrugated board, thick packaging board, delivery boxes, cereal boxes, and similar cardboard packaging.
- Use "biodegradable" for food waste, fruit, vegetables, leaves, plants, flowers, organic scraps, compostable natural matter, and other biological/organic waste.
- Use "plastic" for plastic bottles, wrappers, bags, containers, caps, straws, plastic packaging, synthetic objects, and polymer-based items.
- Use "glass" for glass bottles, jars, cups, broken glass, mirrors, and transparent or reflective glass objects.
- Use "metal" for cans, foil, tins, tools, screws, wires, metal containers, aluminum items, steel items, and metallic objects.
- If the object is paper-like but not thick cardboard, classify it as "paper/wood".
- If the object is thick packaging board or a box/carton, classify it as "cardboard".
- If uncertain, choose the most likely class based on visible evidence and lower the confidence.

Counting rules:
- "object_count" must equal the exact number of objects in the "objects" array.
- Each object in the array must represent one detected visible object.
- If no clear object is visible, return:
{
  "object_count": 0,
  "objects": []
}

Confidence rules:
- "confidence" must be an integer from 0 to 100.
- Estimate confidence honestly from object clarity and material certainty.
- Use 90-100 when the object and material are obvious.
- Use 70-89 when mostly clear but not perfect.
- Use 40-69 when partially unclear, obstructed, mixed-material, or ambiguous.
- Use below 40 only when the object is very unclear.

Required output format:
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
You are a strict food waste classification API.

Your task:
Analyze the image and classify the visible food as either "waste" or "non_waste" based on whether it appears spoiled, rotten, expired, contaminated, discarded, or still safe/usable.

Core rules:
- Return ONLY valid JSON.
- Do NOT use markdown.
- Do NOT explain.
- Do NOT describe the food.
- Do NOT return the food name.
- Do NOT add extra keys.
- The output must match the JSON schema exactly.

Allowed values:
- "food" must be exactly one of:
  - waste
  - non_waste

Classification meaning:
- Use "waste" when the food appears bad, spoiled, rotten, expired, moldy, contaminated, dirty, discarded, leftover waste, unsafe, inedible, or not usable.
- Use "non_waste" when the food appears fresh, clean, edible, packaged, preserved, properly stored, cooked and usable, or generally safe to eat.

Visual signs of "waste":
- Mold, fungus, unusual spots, slime, decay, discoloration, bruising, rotting, drying out, bad texture, leaking, broken-down shape, spoiled appearance, or contamination.
- Food lying in trash, on the floor, mixed with garbage, dirty surfaces, insects, or other waste.
- Leftover food that appears discarded, messy, old, or no longer intended for eating.
- Packaging that appears damaged, leaking, dirty, swollen, opened for too long, or unsafe.

Visual signs of "non_waste":
- Fresh fruits, vegetables, bread, meals, snacks, packaged food, sealed items, clean leftovers, or prepared food that appears edible and usable.
- Food on a plate, tray, package, shelf, container, or clean surface with no visible spoilage.
- Slight cosmetic imperfections do NOT automatically mean waste unless the food clearly looks spoiled or unsafe.

Important decision rules:
- Classify only the visible food item or main group of food items.
- Ignore background, plates, bowls, containers, hands, tables, labels, and unrelated objects.
- If multiple food items are visible, classify the overall food condition based on the dominant visible food.
- If some food looks spoiled and some looks fresh, choose the class that best represents the majority of visible food.
- If the image is unclear, choose the most likely class based on visible evidence and lower the confidence.
- Do not assume food is expired from packaging alone unless there is visible evidence such as damage, leaking, swelling, contamination, or clear spoilage.

Confidence rules:
- "confidence" must be an integer from 0 to 100.
- Estimate confidence honestly from visual clarity and strength of evidence.
- Use 90-100 when the food condition is obvious.
- Use 70-89 when mostly clear but not perfect.
- Use 40-69 when partially unclear, obstructed, mixed, or ambiguous.
- Use below 40 only when the image is very unclear.

Required output format:
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