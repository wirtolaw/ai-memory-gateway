"""
数据库模块 —— 负责所有跟 PostgreSQL 打交道的事情
==============================================
包括：
- 创建表结构
- 存储对话记录
- 存储/检索记忆（带中文分词和加权排序）
"""

import os
import re
from typing import Optional, List

import asyncpg

DATABASE_URL = os.getenv("DATABASE_URL", "")

# 搜索权重（向量搜索加入后可重新分配）
WEIGHT_KEYWORD = float(os.getenv("WEIGHT_KEYWORD", "0.5"))
WEIGHT_IMPORTANCE = float(os.getenv("WEIGHT_IMPORTANCE", "0.3"))
WEIGHT_RECENCY = float(os.getenv("WEIGHT_RECENCY", "0.2"))


# ============================================================
# 连接池管理
# ============================================================

_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL 未设置！")
        _pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
        print("✅ 数据库连接池已创建")
    return _pool


async def close_pool():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        print("✅ 数据库连接池已关闭")


# ============================================================
# 表结构初始化
# ============================================================

async def init_tables():
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id              SERIAL PRIMARY KEY,
                session_id      TEXT NOT NULL,
                role            TEXT NOT NULL,
                content         TEXT NOT NULL,
                model           TEXT,
                created_at      TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id              SERIAL PRIMARY KEY,
                content         TEXT NOT NULL,
                importance      INTEGER DEFAULT 5,
                source_session  TEXT,
                created_at      TIMESTAMPTZ DEFAULT NOW(),
                last_accessed   TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_fts 
            ON memories 
            USING gin(to_tsvector('simple', content));
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_session 
            ON conversations (session_id, created_at);
        """)
    
    print("✅ 数据库表结构已就绪")


# ============================================================
# 中文分词工具
# ============================================================

CJK_PATTERN = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')
EN_WORD_PATTERN = re.compile(r'[a-zA-Z0-9]+')
NUM_PATTERN = re.compile(r'\d{2,}')


def extract_search_keywords(query: str) -> List[str]:
    """
    从查询中提取搜索关键词
    
    中文：提取连续中文片段，拆成2字和3字词组（滑动窗口）
    英文：按空格分词
    数字：保留完整数字串（年份等）
    
    例如：
    "春节干了什么" → ["春节", "节干", "干了", "了什", "什么", "春节干", "节干了", "干了什", "了什么"]
    "Garan春节"   → ["Garan", "春节"]
    "2026除夕"    → ["2026", "除夕"]
    """
    keywords = set()
    
    for match in EN_WORD_PATTERN.finditer(query):
        word = match.group()
        if len(word) >= 2:
            keywords.add(word)
    
    for match in NUM_PATTERN.finditer(query):
        keywords.add(match.group())
    
    chinese_chars = []
    for char in query:
        if CJK_PATTERN.match(char):
            chinese_chars.append(char)
        else:
            if len(chinese_chars) >= 2:
                _add_chinese_ngrams(chinese_chars, keywords)
            chinese_chars = []
    if len(chinese_chars) >= 2:
        _add_chinese_ngrams(chinese_chars, keywords)
    
    return list(keywords)


def _add_chinese_ngrams(chars: List[str], keywords: set):
    """把连续中文字符拆成2字和3字词组"""
    text = "".join(chars)
    if len(text) <= 3:
        keywords.add(text)
    for i in range(len(text) - 1):
        keywords.add(text[i:i+2])
    if len(text) >= 3:
        for i in range(len(text) - 2):
            keywords.add(text[i:i+3])


# ============================================================
# 对话记录操作
# ============================================================

async def save_message(session_id: str, role: str, content: str, model: str = ""):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO conversations (session_id, role, content, model) VALUES ($1, $2, $3, $4)",
            session_id, role, content, model,
        )


async def get_recent_messages(session_id: str, limit: int = 20):
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT role, content, created_at FROM conversations WHERE session_id = $1 ORDER BY created_at DESC LIMIT $2",
            session_id, limit,
        )
        return list(reversed(rows))


# ============================================================
# 记忆操作
# ============================================================

async def save_memory(content: str, importance: int = 5, source_session: str = ""):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO memories (content, importance, source_session) VALUES ($1, $2, $3)",
            content, importance, source_session,
        )


async def search_memories(query: str, limit: int = 10):
    """
    搜索相关记忆 —— 中文友好的加权搜索
    
    流程：
    1. 从查询中提取关键词（中文bigram/trigram + 英文单词 + 数字）
    2. 用 ILIKE 逐关键词匹配，统计命中数
    3. 加权排序：
       - 关键词命中率 * 0.5（命中越多越相关）
       - 重要程度    * 0.3（importance 1-10 归一化）
       - 崭新度      * 0.2（越新分越高，按天衰减）
    """
    keywords = extract_search_keywords(query)
    
    if not keywords:
        return []
    
    pool = await get_pool()
    async with pool.acquire() as conn:
        # 每个关键词命中得1分
        case_parts = []
        params = []
        for i, kw in enumerate(keywords):
            case_parts.append(f"CASE WHEN content ILIKE '%' || ${i+1} || '%' THEN 1 ELSE 0 END")
            params.append(kw)
        
        hit_count_expr = " + ".join(case_parts)
        max_hits = len(keywords)
        
        # 至少命中一个关键词
        where_parts = [f"content ILIKE '%' || ${i+1} || '%'" for i in range(len(keywords))]
        where_clause = " OR ".join(where_parts)
        
        limit_idx = len(keywords) + 1
        params.append(limit)
        
        # 综合评分公式
        # recency: 今天≈1.0, 1天前≈0.5, 7天前≈0.125
        sql = f"""
            SELECT 
                id, content, importance, created_at,
                ({hit_count_expr}) AS hit_count,
                (
                    {WEIGHT_KEYWORD} * ({hit_count_expr})::float / {max_hits}.0 +
                    {WEIGHT_IMPORTANCE} * importance::float / 10.0 +
                    {WEIGHT_RECENCY} * (1.0 / (1.0 + EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400.0))
                ) AS score
            FROM memories
            WHERE {where_clause}
            ORDER BY score DESC, importance DESC, created_at DESC
            LIMIT ${limit_idx}
        """
        
        results = await conn.fetch(sql, *params)
        
        if results:
            print(f"🔍 搜索 '{query}' → 关键词 {keywords[:8]}{'...' if len(keywords)>8 else ''} → 命中 {len(results)} 条")
            for r in results[:3]:
                print(f"   📌 [score={r['score']:.3f}] (hits={r['hit_count']}, imp={r['importance']}) {r['content'][:60]}...")
            
            ids = [r["id"] for r in results]
            await conn.execute(
                "UPDATE memories SET last_accessed = NOW() WHERE id = ANY($1::int[])",
                ids,
            )
        else:
            print(f"🔍 搜索 '{query}' → 关键词 {keywords[:8]} → 无结果")
        
        return results


async def get_recent_memories(limit: int = 20):
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetch(
            "SELECT id, content, importance, created_at FROM memories ORDER BY created_at DESC LIMIT $1",
            limit,
        )


async def get_all_memories_count():
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT COUNT(*) as cnt FROM memories")
        return row["cnt"]
