"""
AI Memory Gateway — 带记忆系统的 LLM 转发网关
=============================================
让你的 AI 拥有长期记忆。

工作原理：
1. 接收客户端（Kelivo / ChatBox / 任何 OpenAI 兼容客户端）的消息
2. 自动搜索数据库中的相关记忆，注入 system prompt
3. 转发给 LLM API（支持 OpenRouter / OpenAI / 任何兼容接口）
4. 后台自动存储对话 + 用 AI 提取新记忆

环境变量 MEMORY_ENABLED=false 时退化为纯转发网关（第一阶段）。
"""

import os
import json
import uuid
import asyncio
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse

from database import init_tables, close_pool, save_message, search_memories, save_memory, get_all_memories_count, get_recent_memories, get_all_memories, get_pool
from memory_extractor import extract_memories

# ============================================================
# 配置项 —— 全部从环境变量读取，部署时在云平台面板里设置
# ============================================================

# 你的 API Key（OpenRouter / OpenAI / 其他兼容服务）
API_KEY = os.getenv("API_KEY", "")

# API 地址（改这个就能切换不同的 LLM 服务商）
# OpenRouter: https://openrouter.ai/api/v1/chat/completions
# OpenAI:     https://api.openai.com/v1/chat/completions
# 本地 Ollama: http://localhost:11434/v1/chat/completions
API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")

# 默认模型（如果客户端没指定就用这个）
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "anthropic/claude-sonnet-4")

# 网关端口
PORT = int(os.getenv("PORT", "8080"))

# 记忆系统开关（数据库出问题时可以临时关掉）
MEMORY_ENABLED = os.getenv("MEMORY_ENABLED", "false").lower() == "true"

# 每次注入的最大记忆条数
MAX_MEMORIES_INJECT = int(os.getenv("MAX_MEMORIES_INJECT", "15"))

# 额外的请求头（有些 API 需要，比如 OpenRouter 需要 Referer）
EXTRA_REFERER = os.getenv("EXTRA_REFERER", "https://ai-memory-gateway.local")
EXTRA_TITLE = os.getenv("EXTRA_TITLE", "AI Memory Gateway")


# ============================================================
# 人设加载
# ============================================================

def load_system_prompt():
    """从 system_prompt.txt 文件读取人设内容"""
    prompt_path = os.path.join(os.path.dirname(__file__), "system_prompt.txt")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                return content
    except FileNotFoundError:
        pass
    print("ℹ️  未找到 system_prompt.txt 或文件为空，将不注入 system prompt")
    return ""


SYSTEM_PROMPT = load_system_prompt()
if SYSTEM_PROMPT:
    print(f"✅ 人设已加载，长度：{len(SYSTEM_PROMPT)} 字符")
else:
    print("ℹ️  无人设，纯转发模式")


# ============================================================
# 应用生命周期管理
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动时初始化数据库，关闭时断开连接"""
    if MEMORY_ENABLED:
        try:
            await init_tables()
            count = await get_all_memories_count()
            print(f"✅ 记忆系统已启动，当前记忆数量：{count}")
        except Exception as e:
            print(f"⚠️  数据库初始化失败: {e}")
            print("⚠️  记忆系统将不可用，但网关仍可正常转发")
    else:
        print("ℹ️  记忆系统已关闭（设置 MEMORY_ENABLED=true 开启）")
    
    yield
    
    if MEMORY_ENABLED:
        await close_pool()


app = FastAPI(title="AI Memory Gateway", version="1.0.0", lifespan=lifespan)


# ============================================================
# 记忆注入
# ============================================================

async def build_system_prompt_with_memories(user_message: str) -> str:
    """
    构建带记忆的 system prompt
    1. 用用户消息搜索相关记忆
    2. 格式化成文本拼接到人设后面
    """
    if not MEMORY_ENABLED:
        return SYSTEM_PROMPT
    
    try:
        memories = await search_memories(user_message, limit=MAX_MEMORIES_INJECT)
        
        if not memories:
            return SYSTEM_PROMPT
        
        memory_lines = [f"- {mem['content']}" for mem in memories]
        memory_text = "\n".join(memory_lines)
        
        enhanced_prompt = f"""{SYSTEM_PROMPT}

【从过往对话中检索到的相关记忆】
以下是与当前话题可能相关的历史信息，自然地融入对话中，不要刻意提起"我记得"：
{memory_text}"""
        
        print(f"📚 注入了 {len(memories)} 条相关记忆")
        return enhanced_prompt
        
    except Exception as e:
        print(f"⚠️  记忆检索失败: {e}，使用纯人设")
        return SYSTEM_PROMPT


# ============================================================
# 后台记忆处理
# ============================================================

async def process_memories_background(session_id: str, user_msg: str, assistant_msg: str, model: str):
    """后台异步：存储对话 + 提取记忆（不阻塞主流程）"""
    try:
        await save_message(session_id, "user", user_msg, model)
        await save_message(session_id, "assistant", assistant_msg, model)
        
        # 获取已有记忆，传给提取模型做对比去重
        existing = await get_recent_memories(limit=80)
        existing_contents = [r["content"] for r in existing]
        
        messages_for_extraction = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
        new_memories = await extract_memories(messages_for_extraction, existing_memories=existing_contents)
        
        # 过滤垃圾记忆（不靠模型自觉，硬过滤）
        META_BLACKLIST = [
            "记忆库", "记忆系统", "检索", "没有被记录", "没有被提取",
            "记忆遗漏", "尚未被记录", "写入不完整", "检索功能",
            "系统没有返回", "关键词匹配", "语义匹配", "语义检索",
            "阈值", "数据库", "seed", "导入", "部署",
            "bug", "debug", "端口", "网关",
        ]
        
        filtered_memories = []
        for mem in new_memories:
            content = mem["content"]
            if any(kw in content for kw in META_BLACKLIST):
                print(f"🚫 过滤掉meta记忆: {content[:60]}...")
                continue
            filtered_memories.append(mem)
        
        for mem in filtered_memories:
            await save_memory(
                content=mem["content"],
                importance=mem["importance"],
                source_session=session_id,
            )
        
        if filtered_memories:
            total = await get_all_memories_count()
            print(f"💾 已保存 {len(filtered_memories)} 条新记忆（过滤了 {len(new_memories) - len(filtered_memories)} 条），总计 {total} 条")
            
    except Exception as e:
        print(f"⚠️  后台记忆处理失败: {e}")


# ============================================================
# API 接口
# ============================================================

@app.get("/")
async def health_check():
    """健康检查"""
    memory_count = 0
    if MEMORY_ENABLED:
        try:
            memory_count = await get_all_memories_count()
        except:
            pass
    
    return {
        "status": "running",
        "gateway": "AI Memory Gateway v1.0",
        "system_prompt_loaded": len(SYSTEM_PROMPT) > 0,
        "system_prompt_length": len(SYSTEM_PROMPT),
        "memory_enabled": MEMORY_ENABLED,
        "memory_count": memory_count,
    }


@app.get("/v1/models")
async def list_models():
    """模型列表（让客户端不报错）"""
    return {
        "object": "list",
        "data": [
            {
                "id": DEFAULT_MODEL,
                "object": "model",
                "created": 1700000000,
                "owned_by": "ai-memory-gateway",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """核心转发接口"""
    if not API_KEY:
        return JSONResponse(
            status_code=500,
            content={"error": "API_KEY 未设置，请在环境变量中配置"},
        )
    
    body = await request.json()
    messages = body.get("messages", [])
    
    # ---------- 提取用户最新消息 ----------
    user_message = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                user_message = content
            elif isinstance(content, list):
                user_message = " ".join(
                    item.get("text", "") for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                )
            break
    
    # ---------- 构建 system prompt ----------
    if SYSTEM_PROMPT or (MEMORY_ENABLED and user_message):
        if MEMORY_ENABLED and user_message:
            enhanced_prompt = await build_system_prompt_with_memories(user_message)
        else:
            enhanced_prompt = SYSTEM_PROMPT
        
        if enhanced_prompt:
            has_system = any(msg.get("role") == "system" for msg in messages)
            if has_system:
                for i, msg in enumerate(messages):
                    if msg.get("role") == "system":
                        messages[i]["content"] = enhanced_prompt + "\n\n" + msg["content"]
                        break
            else:
                messages.insert(0, {"role": "system", "content": enhanced_prompt})
    
    body["messages"] = messages
    
    # ---------- 模型处理 ----------
    model = body.get("model", DEFAULT_MODEL)
    if not model:
        model = DEFAULT_MODEL
    body["model"] = model
    
    # ---------- 生成 session ID ----------
    session_id = str(uuid.uuid4())[:8]
    
    # ---------- 转发请求 ----------
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    # OpenRouter 需要的额外头
    if "openrouter" in API_BASE_URL:
        headers["HTTP-Referer"] = EXTRA_REFERER
        headers["X-Title"] = EXTRA_TITLE
    
    is_stream = body.get("stream", False)
    
    if is_stream:
        return StreamingResponse(
            stream_and_capture(headers, body, session_id, user_message, model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    else:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(API_BASE_URL, headers=headers, json=body)
            
            if response.status_code == 200:
                resp_data = response.json()
                assistant_msg = ""
                try:
                    assistant_msg = resp_data["choices"][0]["message"]["content"]
                except (KeyError, IndexError):
                    pass
                
                if MEMORY_ENABLED and user_message and assistant_msg:
                    asyncio.create_task(
                        process_memories_background(session_id, user_message, assistant_msg, model)
                    )
                
                return JSONResponse(status_code=200, content=resp_data)
            else:
                return JSONResponse(status_code=response.status_code, content=response.json())


async def stream_and_capture(headers: dict, body: dict, session_id: str, user_message: str, model: str):
    """流式响应 + 捕获完整回复"""
    full_response = []
    
    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream("POST", API_BASE_URL, headers=headers, json=body) as response:
            async for line in response.aiter_lines():
                if line:
                    yield line + "\n"
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            data = json.loads(line[6:])
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                full_response.append(content)
                        except (json.JSONDecodeError, KeyError, IndexError):
                            pass
    
    assistant_msg = "".join(full_response)
    if MEMORY_ENABLED and user_message and assistant_msg:
        asyncio.create_task(
            process_memories_background(session_id, user_message, assistant_msg, model)
        )


# ============================================================
# 记忆管理接口
# ============================================================

@app.get("/debug/memories")
async def debug_memories(q: str = "", limit: int = 20):
    """查看和搜索记忆"""
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用（设置 MEMORY_ENABLED=true 开启）"}
    
    try:
        if q:
            memories = await search_memories(q, limit=limit)
        else:
            from database import get_recent_memories
            memories = await get_recent_memories(limit=limit)
        
        total = await get_all_memories_count()
        
        return {
            "total_memories": total,
            "query": q or "(最近记忆)",
            "results": [
                {
                    "content": m["content"],
                    "importance": m["importance"],
                    "created_at": str(m["created_at"]),
                }
                for m in memories
            ],
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/import/seed-memories")
async def import_seed_memories():
    """一次性导入预置记忆（从 seed_memories.py）"""
    try:
        from seed_memories import run_seed_import
        result = await run_seed_import()
        return result
    except ImportError:
        return {"error": "未找到 seed_memories.py，请参考 seed_memories_example.py 创建"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/export/memories")
async def export_memories():
    """
    导出所有记忆为 JSON（用于备份或迁移）
    浏览器访问这个地址就会返回所有记忆数据
    """
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用（设置 MEMORY_ENABLED=true 开启）"}
    
    try:
        memories = await get_all_memories()
        # 把 datetime 转成字符串
        for mem in memories:
            if mem.get("created_at"):
                mem["created_at"] = str(mem["created_at"])
        
        return {
            "total": len(memories),
            "exported_at": str(__import__("datetime").datetime.now()),
            "memories": memories,
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/import/memories", response_class=HTMLResponse)
async def import_memories_page():
    """导入记忆的网页界面"""
    if not MEMORY_ENABLED:
        return HTMLResponse("<h3>记忆系统未启用（设置 MEMORY_ENABLED=true 开启）</h3>")
    
    return HTMLResponse("""
    <!DOCTYPE html>
    <html><head><meta charset="utf-8"><title>导入记忆</title>
    <style>
        body { font-family: sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; }
        textarea { width: 100%; height: 200px; font-size: 14px; margin: 10px 0; }
        button { padding: 10px 20px; font-size: 16px; cursor: pointer; background: #4CAF50; color: white; border: none; border-radius: 4px; }
        button:hover { background: #45a049; }
        input[type="file"] { margin: 10px 0; font-size: 14px; }
        #result { margin-top: 15px; padding: 10px; white-space: pre-wrap; }
        .ok { background: #e8f5e9; } .err { background: #ffebee; }
        .divider { margin: 20px 0; text-align: center; color: #999; }
    </style></head><body>
    <h2>📥 导入记忆</h2>
    <p><b>方式一：</b>上传从 <code>/export/memories</code> 保存的 JSON 文件</p>
    <input type="file" id="file" accept=".json">
    <div class="divider">—— 或者 ——</div>
    <p><b>方式二：</b>直接粘贴 JSON</p>
    <textarea id="json" placeholder='粘贴导出的 JSON'></textarea>
    <br><button onclick="doImport()">导入</button>
    <div id="result"></div>
    <script>
    async function doImport() {
        const r = document.getElementById('result');
        const file = document.getElementById('file').files[0];
        const text = document.getElementById('json').value.trim();
        
        let jsonStr = '';
        if (file) {
            jsonStr = await file.text();
        } else if (text) {
            jsonStr = text;
        } else {
            r.className = 'err'; r.textContent = '请先上传文件或粘贴 JSON'; return;
        }
        
        try {
            const parsed = JSON.parse(jsonStr);
            r.className = ''; r.textContent = '导入中...';
            const resp = await fetch('/import/memories', {
                method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(parsed)
            });
            const data = await resp.json();
            if (data.error) { r.className = 'err'; r.textContent = '❌ ' + data.error; }
            else { r.className = 'ok'; r.textContent = '✅ 导入完成！新增 ' + data.imported + ' 条，跳过 ' + data.skipped + ' 条（已存在），总计 ' + data.total + ' 条'; }
        } catch(e) { r.className = 'err'; r.textContent = '❌ JSON 格式错误：' + e.message; }
    }
    </script></body></html>
    """)



@app.post("/import/memories")
async def import_memories(request: Request):
    """
    从 JSON 导入记忆（用于迁移或恢复备份）
    接受 /export/memories 导出的格式，自动跳过已存在的记忆
    """
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用（设置 MEMORY_ENABLED=true 开启）"}
    
    try:
        data = await request.json()
        memories = data.get("memories", [])
        
        if not memories:
            return {"error": "没有找到记忆数据，请确认 JSON 格式正确"}
        
        imported = 0
        skipped = 0
        
        for mem in memories:
            content = mem.get("content", "")
            if not content:
                continue
            
            # 检查是否已存在
            pool = await get_pool()
            async with pool.acquire() as conn:
                existing = await conn.fetchval(
                    "SELECT COUNT(*) FROM memories WHERE content = $1", content
                )
            
            if existing > 0:
                skipped += 1
                continue
            
            await save_memory(
                content=content,
                importance=mem.get("importance", 5),
                source_session=mem.get("source_session", "json-import"),
            )
            imported += 1
        
        total = await get_all_memories_count()
        return {
            "status": "done",
            "imported": imported,
            "skipped": skipped,
            "total": total,
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# 启动入口
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print(f"🚀 AI Memory Gateway 启动中... 端口 {PORT}")
    print(f"📝 人设长度：{len(SYSTEM_PROMPT)} 字符")
    print(f"🤖 默认模型：{DEFAULT_MODEL}")
    print(f"🔗 API 地址：{API_BASE_URL}")
    print(f"🧠 记忆系统：{'开启' if MEMORY_ENABLED else '关闭'}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
