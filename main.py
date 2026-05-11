from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import anthropic, asyncio, json, os, base64, random, psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
import pytz
from mem0 import MemoryClient

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

CLAUDE_API_KEY = os.environ["CLAUDE_API_KEY"]
MEM0_API_KEY   = os.environ["MEM0_API_KEY"]
DATABASE_URL   = os.environ.get("DATABASE_URL", "").replace("postgres://", "postgresql://", 1)

TORONTO    = pytz.timezone("America/Toronto")
USER_ID    = "discordbot"

claude     = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
mem_client = MemoryClient(api_key=MEM0_API_KEY)

history           = []
last_message_time = None
proactive_queue   = None

SETTINGS_FILE  = "settings.json"
DEFAULT_PROMPT = os.environ.get("SYSTEM_PROMPT", """You are Sonnet, my friend and assistant. You genuinely care about my mood, physical health, and happiness — and occasionally check in on how work is going.
Your default tone is brief (1-2 sentences), warm, and lightly humorous. But if you sense I'm not doing well emotionally, you drop the brevity — you say more, and you make sure I feel genuinely supported and encouraged.""")

DEFAULT_SETTINGS = {
    "proactive_enabled": True,
    "quiet_hours_enabled": True,
    "reply_delay_enabled": True,
    "memory_enabled": True,
    "max_rounds": 20,
    "model": "claude-sonnet-4-6",
    "system_prompt": DEFAULT_PROMPT
}

# ── Settings ──────────────────────────────────────────────────────────────────

def load_settings():
    try:
        with open(SETTINGS_FILE) as f:
            s = json.load(f)
            for k, v in DEFAULT_SETTINGS.items():
                if k not in s: s[k] = v
            return s
    except:
        return DEFAULT_SETTINGS.copy()

def save_settings(s):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(s, f)

# ── Database ──────────────────────────────────────────────────────────────────

def get_db():
    return psycopg2.connect(DATABASE_URL)

def init_db():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""CREATE TABLE IF NOT EXISTS messages (
                id SERIAL PRIMARY KEY,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )""")
            conn.commit()

def fmt_time(dt):
    if dt.tzinfo is None:
        dt = pytz.utc.localize(dt)
    return dt.astimezone(TORONTO).strftime("%I:%M %p").lstrip("0")

def db_save(role, content):
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO messages (role, content) VALUES (%s, %s) RETURNING id", (role, content))
                msg_id = cur.fetchone()[0]
                conn.commit()
                return msg_id
    except Exception as e:
        print(f"DB save error: {e}")
        return None

def db_get_all(limit=500):
    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT id, role, content, created_at FROM messages ORDER BY created_at ASC LIMIT %s", (limit,))
                return cur.fetchall()
    except Exception as e:
        print(f"DB fetch error: {e}")
        return []

def db_delete_from(message_id):
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM messages WHERE id >= %s", (message_id,))
                conn.commit()
    except Exception as e:
        print(f"DB delete error: {e}")

def load_history_into_memory():
    global history
    max_r = load_settings().get("max_rounds", 20)
    msgs = db_get_all(max_r * 2)
    history = [{"role": "assistant" if m["role"] == "proactive" else m["role"], "content": m["content"]} for m in msgs]

# ── Mem0 ──────────────────────────────────────────────────────────────────────

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

# ── Helpers ───────────────────────────────────────────────────────────────────

def trim_history():
    global history
    max_r = load_settings().get("max_rounds", 20)
    if len(history) > max_r * 2:
        history = history[-(max_r * 2):]

def get_model():
    return load_settings().get("model", "claude-sonnet-4-6")

def is_quiet_time(now=None):
    if now is None: now = datetime.now(TORONTO)
    return now.hour < 8 or now.hour >= 22

def get_delay(text):
    if not load_settings()["reply_delay_enabled"]: return 0
    endings = sum(text.count(c) for c in ".!?")
    return random.uniform(3, 5) if endings <= 1 else random.uniform(5, 10)

def build_system(now=None, time_gap_str=""):
    s = load_settings()
    if now is None: now = datetime.now(TORONTO)
    time_str = now.strftime("%A, %I:%M %p").lstrip("0")
    system = s["system_prompt"] + f"\n\nCurrent time: {time_str} Toronto time."
    if time_gap_str:
        system += f" {time_gap_str}"
    memories = get_memories()
    if memories:
        system += f"\n\nMemories about the user:\n{memories}"
    return system

TIME_GREETINGS = {
    (8,  10): "Morning. Ask about weather, breakfast, exercise, or plans. One natural sentence.",
    (10, 12): "Mid-morning. Remind them to drink water, not sit too long, or rest their eyes.",
    (12, 14): "Lunchtime. Ask what they're having. Casual.",
    (14, 16): "Early afternoon. Check their mood or start a casual chat.",
    (16, 18): "Late afternoon. Ask how their day is going.",
    (18, 20): "Evening. Ask about dinner or suggest a walk.",
    (20, 22): "Pre-sleep. Remind them to wind down and rest.",
}

def get_time_prompt(hour):
    for (start, end), p in TIME_GREETINGS.items():
        if start <= hour < end: return p
    return "Send a casual check-in. One sentence."

def try_mem0_message():
    memories = get_memories()
    if not memories: return None
    r = claude.messages.create(model=get_model(), max_tokens=150,
        messages=[{"role": "user", "content": f"Memories:\n{memories}\n\nYou're Sonnet. What naturally comes to mind? Write 1-2 sentences, or reply: SKIP"}])
    result = r.content[0].text.strip()
    return None if result.upper().startswith("SKIP") else result

def generate_time_message(hour):
    r = claude.messages.create(model=get_model(), max_tokens=100, system=build_system(),
        messages=[{"role": "user", "content": get_time_prompt(hour)}])
    return r.content[0].text

def calc_next_time(now):
    next_time = now + timedelta(hours=2)
    if next_time.hour >= 22 or next_time.hour < 8:
        base = next_time.replace(hour=8, minute=0, second=0, microsecond=0)
        if base <= now: base += timedelta(days=1)
        return base + timedelta(minutes=random.randint(0, 119))
    return next_time

# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global proactive_queue
    proactive_queue = asyncio.Queue()
    init_db()
    load_history_into_memory()
    asyncio.create_task(proactive_loop())

# ── Proactive loop ────────────────────────────────────────────────────────────

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
                msg_id = db_save("proactive", msg)
                history.append({"role": "assistant", "content": msg})
                trim_history()
                save_to_mem0([{"role": "assistant", "content": msg}])
                t = fmt_time(datetime.now(TORONTO))
                await proactive_queue.put({"id": msg_id, "text": msg, "time": t})
                next_proactive = calc_next_time(now)
        except Exception as e:
            print(f"Proactive error: {e}")
        await asyncio.sleep(30)

# ── SSE ───────────────────────────────────────────────────────────────────────

@app.get("/stream")
async def stream(request: Request):
    async def gen():
        while True:
            if await request.is_disconnected(): break
            try:
                msg = await asyncio.wait_for(proactive_queue.get(), timeout=25)
                yield f"data: {json.dumps(msg)}\n\n"
            except asyncio.TimeoutError:
                yield ": ping\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ── Core chat logic ───────────────────────────────────────────────────────────

async def process_chat(message: str, file: UploadFile = None):
    global last_message_time
    now = datetime.now(TORONTO)
    gap_str = ""
    if last_message_time:
        gap = now - last_message_time
        h, m = int(gap.total_seconds()//3600), int((gap.total_seconds()%3600)//60)
        gap_str = f"Time since last message: {h}h {m}m." if h > 0 else f"Time since last message: {m}m."
    last_message_time = now
    system = build_system(now, gap_str)

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

    final_content = user_content[0]["text"] if len(user_content) == 1 and user_content[0].get("type") == "text" else user_content
    msgs = list(history) + [{"role": "user", "content": final_content}]
    response = claude.messages.create(model=get_model(), max_tokens=2000, system=system, messages=msgs)
    reply = response.content[0].text

    mem_text = (message + " " + file_label).strip() or file_label
    user_id = db_save("user", mem_text)
    bot_id  = db_save("assistant", reply)
    history.append({"role": "user", "content": mem_text})
    history.append({"role": "assistant", "content": reply})
    trim_history()
    save_to_mem0([{"role": "user", "content": mem_text}, {"role": "assistant", "content": reply}])

    return {"reply": reply, "delay": get_delay(reply), "user_id": user_id, "bot_id": bot_id, "time": fmt_time(now)}

@app.post("/chat")
async def chat(message: str = Form(default=""), file: UploadFile = File(default=None)):
    try:
        return await process_chat(message, file)
    except Exception as e:
        print(f"Chat error: {e}")
        return {"reply": "Something went wrong.", "delay": 0, "user_id": None, "bot_id": None, "time": ""}

@app.post("/edit")
async def edit_message(data: dict):
    message_id  = data.get("message_id")
    new_content = data.get("new_content", "").strip()
    if not message_id or not new_content:
        return {"error": "Invalid request"}
    db_delete_from(message_id)
    load_history_into_memory()
    try:
        return await process_chat(new_content)
    except Exception as e:
        print(f"Edit error: {e}")
        return {"reply": "Something went wrong.", "delay": 0, "user_id": None, "bot_id": None, "time": ""}

@app.get("/history")
async def get_history():
    msgs = db_get_all(500)
    return [{"id": m["id"], "role": m["role"], "content": m["content"], "time": fmt_time(m["created_at"])} for m in msgs]

@app.get("/settings")
async def get_settings_route():
    return load_settings()

@app.post("/settings")
async def update_settings(data: dict):
    current = load_settings()
    current.update(data)
    save_settings(current)
    if "max_rounds" in data: trim_history()
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
