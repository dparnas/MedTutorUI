import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import traceback
from fastapi import HTTPException

from langchain_openai import ChatOpenAI

load_dotenv(r'C:\Users\dovid\PycharmProjects\MedTutorUI\variable.env')
# load_dotenv()
app = FastAPI()

# ---- LLM client ----
def get_llm() -> ChatOpenAI:
    api_key = os.environ["LLMOD_API_KEY"]
    base_url = os.getenv("LLMOD_BASE_URL", "https://api.llmod.ai/v1")
    model = os.getenv("LLMOD_MODEL", "RPRTHPB-gpt-5-mini")
    return ChatOpenAI(api_key=api_key, base_url=base_url, model=model)

# ---- Required schemas ----
class ExecuteRequest(BaseModel):
    prompt: str

@app.get("/", response_class=HTMLResponse)
def home():
    # Serve a single-page UI (we'll create ui.html next)
    with open("ui.html", "r", encoding="utf-8") as f:
        return f.read()

# ---- Required endpoints (stub minimal) ----
@app.get("/api/team_info")
def team_info():
    return {
        "group_batch_order_number": "3_1",
        "team_name": "דוד + אורן + תומר",
        "students": [
            {"name": "Dovid Parnas", "email": "dparnas@campus.technion.ac.il"},
            {"name": "Tomer Dovzhenko", "email": "tomer.d@campus.technion.ac.il"},
            {"name": "Oren Chickli", "email": "francois.c@campus.technion.ac.il"},
        ],
    }

@app.get("/api/agent_info")
def agent_info():
    return {
        "description": "Simple chat wrapper around a single LLMod.ai LLM call.",
        "purpose": "Demonstrate /api/execute + steps trace + minimal chat UI.",
        "prompt_template": {
            "template": "SYSTEM: {system}\n\nCONVERSATION:\n{history}\n\nNEW USER MESSAGE:\n{user}"
        },
        "prompt_examples": [
            {
                "prompt": "Hi, can you explain what an EIF is?",
                "full_response": "An efficient influence function (EIF) is ...",
                "steps": [
                    {
                        "module": "LLM",
                        "prompt": {"compiled_prompt": "..."},
                        "response": {"content": "..."},
                    }
                ],
            }
        ],
    }

@app.get("/api/model_architecture")
def model_architecture():
    # Keep it simple for now; later match your CLAUDE.md architecture.
    return {
        "architecture": "Single LLM call",
        "modules": [
            {"name": "LLM", "description": "One ChatOpenAI invoke() call to LLMod.ai."}
        ],
    }

# ---- Core endpoint: POST /api/execute ---
@app.post("/api/execute")
def execute(req: ExecuteRequest):
    try:
        llm = get_llm()
        ai_msg = llm.invoke(req.prompt)
        content = getattr(ai_msg, "content", str(ai_msg))

        steps = [{
            "module": "LLM",
            "prompt": {"compiled_prompt": req.prompt},
            "response": {"content": content},
        }]

        return {"status": "success", "error": None, "response": content, "steps": steps}

    except Exception as e:
        tb = traceback.format_exc()
        # Print to Render logs
        print("ERROR in /api/execute:", repr(e))
        print(tb)

        # Return a JSON error payload (so your test script can print it)
        return {
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "traceback": tb,
            "response": "",
            "steps": []
        }

@app.get("/api/debug_env")
def debug_env():
    import os
    return {
        "has_LLMOD_API_KEY": bool(os.getenv("LLMOD_API_KEY")),
        "LLMOD_BASE_URL": os.getenv("LLMOD_BASE_URL"),
        "LLMOD_MODEL": os.getenv("LLMOD_MODEL"),
    }
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8000))
#     uvicorn.run("main:app", host="127.0.0.1", port=port, reload=True)
