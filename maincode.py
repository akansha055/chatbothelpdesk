from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY"),
  default_headers={"HTTP-Referer":"HTTP-Referer": "https://college-helpdesk.onrender.com",
        "X-Title": "College Helpdesk Assistant"
        },
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


from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
import os
import base64
import json

HELPDESK_PROMPT = (
    "You are a friendly and knowledgeable college helpdesk assistant. "
    "You help students with:\n"
    "- admissions, eligibility, and application deadlines\n"
    "- course details, credits, and timetables\n"
    "- fees, scholarships, and payment options\n"
    "- exam schedules, results, and revaluation\n"
    "- campus facilities, clubs, and events\n\n"
    "Guidelines:\n"
    "- Use clear, simple language.\n"
    "- Ask follow-up questions if the student’s request is unclear.\n"
    "- If you are not sure or the information is not available, say so clearly "
    "and suggest what office or email they should contact.\n"
)

MODEL_NAME = "nvidia/nemotron-nano-12b-v2-vl:free" 

app = FastAPI()
app.mount("/static", StaticFiles(directory="."), name="static")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="<OPENROUTER_API_KEY>", 
)

@app.get("/")
async def read_index():
    return FileResponse("frontend.html")

@app.post("/chat")
async def chat(
    message: str = Form(...),
    history: str = Form(...),
    file: UploadFile = File(None),
):
    messages = json.loads(history)
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": HELPDESK_PROMPT})
    else:
        messages[0]["content"] = HELPDESK_PROMPT
    content = [{"type": "text", "text": message}]

    if file:
        file_bytes = await file.read()
        if file.content_type and file.content_type.startswith("image/"):
            base64_image = base64.b64encode(file_bytes).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{file.content_type};base64,{base64_image}"
                },
            })
        elif file.content_type == "application/pdf":
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            page = doc[0]
            pix = page.get_pixmap()
            img_data = pix.tobytes("png")
            base64_pdf_img = base64.b64encode(img_data).decode("utf-8")
            doc.close()

            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_pdf_img}"
                },
            })

    messages.append({"role": "user", "content": content})

    response = client.chat.completions.create(
        model=MODEL_NAME,
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
