from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

# First API call with reasoning
response = client.chat.completions.create(
  model="nvidia/nemotron-nano-12b-v2-vl:free",
  messages=[
          {
            "role": "user",
            "content": "How many r's are in the word 'strawberry'?"
          }
        ],
  extra_body={"reasoning": {"enabled": True}}
)

# Extract the assistant message with reasoning_details
response = response.choices[0].message

# Preserve the assistant message with reasoning_details
messages = [
  {"role": "user", "content": "How many r's are in the word 'strawberry'?"},
  {
    "role": "assistant",
    "content": response.content,
    "reasoning_details": response.reasoning_details  # Pass back unmodified
  },
  {"role": "user", "content": "Are you sure? Think carefully."}
]

# Second API call - model continues reasoning from where it left off
response2 = client.chat.completions.create(
  model="nvidia/nemotron-nano-12b-v2-vl:free",
  messages=messages,
  extra_body={"reasoning": {"enabled": True}}
)

import requests
from bs4 import BeautifulSoup

def fetch_website_text(url: str) -> str:
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove scripts/styles
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # Basic cleanup + truncate to avoid huge prompts
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned = "\n".join(lines)
    return cleaned[:8000]  # keep first 8k chars

from openai import OpenAI
import os
import base64
import json
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Mount your static files (CSS/JS)
app.mount("/static", StaticFiles(directory="."),name="static")

# Initialize OpenRouter Client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)
@app.get("/")
async def read_index():
    # Serves your HTML frontend
    return FileResponse('frontend.html')
  # put your website here
TARGET_URL = "https://dtu.ac.in" 
@app.post("/chat")
async def chat(
    message: str = Form(...),
    history: str = Form(...),
    file: UploadFile = File(None),
):
    messages = json.loads(history)

    # Fetch website data (you can cache this in memory or a file)
    site_text = fetch_website_text(TARGET_URL)

    # System message that forces the model to use only that site
    system_prompt = (
        "You are a chatbot that answers ONLY using information from the "
        f"following website content. If the answer is not in this content, say "
        "'I don't know based on the provided website.'\n\n"
        f"WEBSITE CONTENT:\n{site_text}"
    )

    # Ensure first message is a system message for this behavior
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": system_prompt})
    else:
        messages[0]["content"] = system_prompt

    # Build current user content (text + optional image)
    content = [{"type": "text", "text": message}]

    if file and file.content_type and file.content_type.startswith("image/"):
        file_bytes = await file.read()
        base64_image = base64.b64encode(file_bytes).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{file.content_type};base64,{base64_image}"
            },
        })

    messages.append({"role": "user", "content": content})

    response = client.chat.completions.create(
        model="nvidia/nemotron-nano-12b-v2-vl:free",
        messages=messages,
    )

    assistant_message = response.choices[0].message
    if isinstance(assistant_message.content, list):
        text_parts = [
            part.get("text", "")
            for part in assistant_message.content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        reply_text = "".join(text_parts)
    else:
        reply_text = assistant_message.content

    return {"response": reply_text}
