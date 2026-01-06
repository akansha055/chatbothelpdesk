import os
import base64
import json
import fitz  
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="<OPENROUTER_API_KEY>", 
    default_headers={
        "HTTP-Referer": "https://college-helpdesk.onrender.com",
        "X-Title": "College Helpdesk Assistant",
    },
)

HELPDESK_PROMPT = (
    "You are a friendly and knowledgeable college helpdesk assistant. "
    "You help students with admissions, course details, fees, and events. "
    "Use clear, simple language."
)

MODEL_NAME = "nvidia/nemotron-nano-12b-v2-vl:free" 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
async def read_index():
    return FileResponse("frontend.html")

@app.post("/chat")
async def chat(
    message: str = Form(...),
    history: str = Form(...),
    file: UploadFile = File(None),
):
    try:
        messages = json.loads(history)
    except:
        messages = []

    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": HELPDESK_PROMPT})
    
    content = [{"type": "text", "text": message}]

    if file:
        file_bytes = await file.read()
        if file.content_type and file.content_type.startswith("image/"):
            base64_image = base64.b64encode(file_bytes).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{file.content_type};base64,{base64_image}"}
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
                "image_url": {"url": f"data:image/png;base64,{base64_pdf_img}"}
            })

    messages.append({"role": "user", "content": content})

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
    )

    reply_text = response.choices[0].message.content
    return {"response": reply_text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
