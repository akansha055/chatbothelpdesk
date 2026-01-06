#from openai import OpenAI

#client = OpenAI(
  #base_url="https://openrouter.ai/api/v1",
  #api_key="<OPENROUTER_API_KEY>",
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

import os
import base64
import json
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

@app.get("/")
async def read_index():
    return FileResponse('templates/frontend.html')

@app.post("/chat")
async def chat(
    message: str = Form(...), 
    history: str = Form(...), 
    file: UploadFile = File(None)
):
    messages = json.loads(history)
    content = [{"type": "text", "text": message}]
    
    if file and file.content_type.startswith('image/'):
        file_bytes = await file.read()
        base64_image = base64.b64encode(file_bytes).decode('utf-8')
        
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{file.content_type};base64,{base64_image}"
            }
        })
    
    messages.append({"role": "user", "content": content})
    
    response = client.chat.completions.create(
        model="nvidia/nemotron-nano-12b-v2-vl:free",
        messages=messages
    )
    
    return {"response": response.choices[0].message.content}
