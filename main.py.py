from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import anthropic, asyncio, json, os, base64, random
from datetime import datetime, timedelta
import pytz
from mem0 import MemoryClient

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

CLAUDE_API_KEY = os.environ["CLAUDE_API_KEY"]
MEM0_API_KEY   = os.environ["MEM0_API_KEY"]

TORONTO    = pytz.timezone("America/Toronto")
USER_ID    = "discordbot"
MAX_ROUNDS = 20

claude     = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
mem_client = MemoryClient(api_key=MEM0_API_KEY)

history           = []
last_message_time = None
proactive_queue   = None

SETTINGS_FILE = "settings.json"
DEFAULT_PROMPT = os.environ.get("SYSTEM_PROMPT", """You are Sonnet, my friend and assistant. You genuinely care about my mood, physical health, and happiness — and occasionally check in on how work is going.
Your default tone is brief (1-2 sentences), warm, and lightly humorous. But if you sense I'm not doing well emotionally, you drop the brevity — you say more, and you make sure I feel genuinely supported and encouraged.""")

DEFAULT_SETTINGS = {
    "proactive_enabled": True,
    "quiet_hours_enabled": True,
    "reply_delay_enabled": True,
    "memory_enabled": True,
    "system_prompt": DEFAULT_PROMPT
}

def load_settings():
    try:
        with open(SETTINGS_FILE) as f:
            s = json.load(f)
            for k, v in DEFAULT_SETTINGS.items():
                if k not in s:
                    s[k] = v
            return s
    except:
        return DEFAULT_SETTINGS.copy()

def save_settings(s):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(s, f)

def get_memories():
    if not load_settings()["memory_enabled"]:
        return ""
    try:
        result = mem_client.get_all(filters={"user_id": USER_ID})
        mems = result.get("results", []) if isinstance(result, dict) else (result or [])
        return "\n".join([f"- {m['memory']}" for m in mems[:30]])
    except Exception as e:
        print(f"Mem0 fetch error: {e}")
        return ""

def save_to_mem0(msgs):
    if not load_settings()["memory_enabled"]:
        return
    try:
        mem_client.add(msgs, user_id=USER_ID)
    except Exception as e:
        print(f"Mem0 save error: {e}")

def trim_history():
    global history
    if len(history) > MAX_ROUNDS * 2:
        history = history[-(MAX_ROUNDS * 2):]

def is_quiet_time(now=None):
    if now is None:
        now = datetime.now(TORONTO)
    return now.hour < 8 or now.hour >= 22

def get_delay(text):
    if not load_settings()["reply_delay_enabled"]:
        return 0
    endings = sum(text.count(c) for c in ".!?")
    return random.uniform(3, 5) if endings <= 1 else random.uniform(5, 10)

TIME_GREETINGS = {
    (8,  10): "Morning. Ask about weather, breakfast, exercise, or plans for today. One natural sentence.",
    (10, 12): "Mid-morning. Remind them to drink water, not sit too long, or rest their eyes.",
    (12, 14): "Lunchtime. Ask what they're having for lunch. Casual.",
    (14, 16): "Early afternoon. Check their mood or start a casual chat.",
    (16, 18): "Late afternoon. Ask how their day is going.",
    (18, 20): "Evening. Ask about dinner or suggest a short walk.",
    (20, 22): "Pre-sleep. Remind them to wind down and rest.",
}

def get_time_prompt(hour):
    for (start, end), prompt in TIME_GREETINGS.items():
        if start <= hour < end:
            return prompt
    return "Send a casual check-in. One sentence."

def try_mem0_message():
    memories = get_memories()
    if not memories:
        return None
    response = claude.messages.create(
        model="claude-sonnet-4-6", max_tokens=150,
        messages=[{"role": "user", "content": f"""Memories about your friend:\n{memories}\n\nYou're Sonnet. What naturally comes to mind to reach out about? Unfinished threads, connections, observations. Write a 1-2 sentence message, or reply: SKIP"""}]
    )
    result = response.content[0].text.strip()
    return None if result.upper().startswith("SKIP") else result

def generate_time_message(hour):
    s = load_settings()
    r = claude.messages.create(model="claude-sonnet-4-6", max_tokens=100, system=s["system_prompt"],
        messages=[{"role": "user", "content": get_time_prompt(hour)}])
    return r.content[0].text

def calc_next_time(now):
    next_time = now + timedelta(hours=2)
    if next_time.hour >= 22 or next_time.hour < 8:
        base = next_time.replace(hour=8, minute=0, second=0, microsecond=0)
        if base <= now:
            base += timedelta(days=1)
        return base + timedelta(minutes=random.randint(0, 119))
    return next_time

@app.on_event("startup")
async def startup():
    global proactive_queue
    proactive_queue = asyncio.Queue()
    asyncio.create_task(proactive_loop())

async def proactive_loop():
    await asyncio.sleep(5)
    now = datetime.now(TORONTO)
    if now.hour < 8:
        first = now.replace(hour=8, minute=0, second=0, microsecond=0) + timedelta(minutes=random.randint(0, 119))
    elif now.hour < 22:
        first = now + timedelta(minutes=random.randint(1, 10))
    else:
        first = (now + timedelta(days=1)).replace(hour=8, minute=0, second=0, microsecond=0) + timedelta(minutes=random.randint(0, 119))
    next_proactive = first

    while True:
        try:
            s = load_settings()
            now = datetime.now(TORONTO)
            quiet = is_quiet_time(now) and s["quiet_hours_enabled"]
            if s["proactive_enabled"] and now >= next_proactive and not quiet:
                msg = try_mem0_message() or generate_time_message(now.hour)
                history.append({"role": "assistant", "content": msg})
                trim_history()
                save_to_mem0([{"role": "assistant", "content": msg}])
                await proactive_queue.put({"text": msg, "time": now.strftime("%I:%M %p").lstrip("0")})
                next_proactive = calc_next_time(now)
        except Exception as e:
            print(f"Proactive error: {e}")
        await asyncio.sleep(30)

@app.get("/stream")
async def stream(request: Request):
    async def gen():
        while True:
            if await request.is_disconnected():
                break
            try:
                msg = await asyncio.wait_for(proactive_queue.get(), timeout=25)
                yield f"data: {json.dumps(msg)}\n\n"
            except asyncio.TimeoutError:
                yield ": ping\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.post("/chat")
async def chat(message: str = Form(default=""), file: UploadFile = File(default=None)):
    global last_message_time
    s = load_settings()
    now = datetime.now(TORONTO)
    time_str = now.strftime("%A, %I:%M %p").lstrip("0")
    time_ctx = f"Current time: {time_str} Toronto time."
    if last_message_time:
        gap = now - last_message_time
        h, m = int(gap.total_seconds()//3600), int((gap.total_seconds()%3600)//60)
        time_ctx += f" Time since last message: {h}h {m}m." if h > 0 else f" Time since last message: {m}m."
    last_message_time = now

    memories = get_memories()
    system = s["system_prompt"] + f"\n\n{time_ctx}"
    if memories:
        system += f"\n\nMemories about the user:\n{memories}"

    user_content = []
    file_label = ""
    if file and file.filename:
        file_bytes = await file.read()
        b64 = base64.standard_b64encode(file_bytes).decode()
        if file.content_type == "application/pdf":
            user_content.append({"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": b64}})
        elif file.content_type.startswith("image/"):
            user_content.append({"type": "image", "source": {"type": "base64", "media_type": file.content_type, "data": b64}})
        file_label = f"[{file.filename}]"
    if message:
        user_content.append({"type": "text", "text": message})

    if not user_content:
        return {"error": "Nothing sent"}

    final_content = user_content[0]["text"] if len(user_content) == 1 and user_content[0].get("type") == "text" else user_content
    msgs = list(history) + [{"role": "user", "content": final_content}]

    try:
        response = claude.messages.create(model="claude-sonnet-4-6", max_tokens=2000, system=system, messages=msgs)
        reply = response.content[0].text
    except Exception as e:
        return {"reply": "Something went wrong on my end.", "delay": 0}

    mem_text = (message + " " + file_label).strip() or file_label
    history.append({"role": "user", "content": mem_text})
    history.append({"role": "assistant", "content": reply})
    trim_history()
    save_to_mem0([{"role": "user", "content": mem_text}, {"role": "assistant", "content": reply}])

    return {"reply": reply, "delay": get_delay(reply)}

@app.get("/settings")
async def get_settings():
    return load_settings()

@app.post("/settings")
async def update_settings(data: dict):
    current = load_settings()
    current.update(data)
    save_settings(current)
    return current

@app.get("/sw.js")
async def sw():
    code = "self.addEventListener('fetch', e => e.respondWith(fetch(e.request).catch(() => new Response(''))));"
    return StreamingResponse(iter([code]), media_type="application/javascript")

@app.get("/manifest.json")
async def manifest():
    return {"name": "Sonnet", "short_name": "Sonnet", "start_url": "/", "display": "standalone",
            "background_color": "#ffffff", "theme_color": "#534AB7"}

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html") as f:
        return f.read()