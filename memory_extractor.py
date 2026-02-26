"""
记忆提取模块 —— 用 LLM 从对话中提炼关键记忆
=============================================
每次对话结束后，把最近的对话内容发给一个便宜的模型，
让它提取出值得记住的信息，存到数据库里。

为了省钱，记忆提取用便宜的模型（比如 Haiku），不用 Opus/Sonnet。
"""

import os
import json
import httpx
from typing import List, Dict

# 复用主网关的 API Key 和地址
API_KEY = os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")

# 用来提取记忆的模型（便宜的就行）
MEMORY_MODEL = os.getenv("MEMORY_MODEL", "anthropic/claude-haiku-4")

# OpenRouter 额外头
EXTRA_REFERER = os.getenv("EXTRA_REFERER", "https://ai-memory-gateway.local")
EXTRA_TITLE = os.getenv("EXTRA_TITLE", "AI Memory Gateway")


EXTRACTION_PROMPT = """你是一个记忆提取助手。你的任务是从对话内容中提取值得长期记住的关键信息。

请从以下对话中提取记忆条目。每条记忆应该是一句简洁的陈述句。

提取规则：
1. 提取关于用户的事实信息（喜好、习惯、经历、计划等）
2. 提取重要的情感时刻或关系里程碑
3. 提取用户提到的具体事件、人名、地点
4. 提取用户表达的需求、偏好或反馈
5. 不要提取泛泛的聊天内容（比如"用户说了你好"）
6. 不要提取 AI 助手自己的回复内容，只关注用户透露的信息
7. 每条记忆都要是独立的、完整的句子
8. 给每条记忆打一个重要程度分数（1-10），10 最重要

请用以下 JSON 格式返回（不要包含其他内容）：
[
  {"content": "记忆内容", "importance": 分数},
  {"content": "记忆内容", "importance": 分数}
]

如果这段对话没有值得记住的新信息，返回空数组：[]
"""


async def extract_memories(messages: List[Dict[str, str]]) -> List[Dict]:
    """
    从对话消息中提取记忆
    """
    if not API_KEY:
        print("⚠️  API_KEY 未设置，跳过记忆提取")
        return []
    
    if not messages:
        return []
    
    conversation_text = ""
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if role == "user":
            conversation_text += f"用户: {content}\n"
        elif role == "assistant":
            conversation_text += f"AI: {content}\n"
    
    if not conversation_text.strip():
        return []
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    if "openrouter" in API_BASE_URL:
        headers["HTTP-Referer"] = EXTRA_REFERER
        headers["X-Title"] = EXTRA_TITLE
    
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                API_BASE_URL,
                headers=headers,
                json={
                    "model": MEMORY_MODEL,
                    "max_tokens": 1000,
                    "messages": [
                        {"role": "system", "content": EXTRACTION_PROMPT},
                        {"role": "user", "content": f"请从以下对话中提取记忆：\n\n{conversation_text}"},
                    ],
                },
            )
            
            if response.status_code != 200:
                print(f"⚠️  记忆提取请求失败: {response.status_code}")
                return []
            
            data = response.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            
            memories = json.loads(text)
            
            if not isinstance(memories, list):
                return []
            
            valid_memories = []
            for mem in memories:
                if isinstance(mem, dict) and "content" in mem:
                    valid_memories.append({
                        "content": str(mem["content"]),
                        "importance": int(mem.get("importance", 5)),
                    })
            
            print(f"📝 从对话中提取了 {len(valid_memories)} 条记忆")
            return valid_memories
            
    except json.JSONDecodeError as e:
        print(f"⚠️  记忆提取结果解析失败: {e}")
        return []
    except Exception as e:
        print(f"⚠️  记忆提取出错: {e}")
        return []
