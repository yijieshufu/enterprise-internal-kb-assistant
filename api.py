import json
import math
import os
import re
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from openai import OpenAI
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STORE_PATH = BASE_DIR / "kb_store.json"
COLLECTION_NAME = "enterprise_internal_kb"
MAX_HISTORY_MESSAGES = 6

SAMPLE_DOCS = [
    "employee_handbook.md",
    "reimbursement_policy.md",
    "it_service_guide.md",
]

DIRECTORY = [
    {
        "name": "王敏",
        "department": "人力资源部",
        "title": "HRBP",
        "email": "wangmin@acme.local",
        "phone": "021-6000-1001",
        "responsibility": "负责入职、转正、调岗和考勤制度答疑。",
    },
    {
        "name": "李浩",
        "department": "财务部",
        "title": "费用核算专员",
        "email": "lihao@acme.local",
        "phone": "021-6000-2003",
        "responsibility": "负责差旅报销、发票规范和付款进度说明。",
    },
    {
        "name": "陈工",
        "department": "IT 服务台",
        "title": "桌面支持工程师",
        "email": "it-helpdesk@acme.local",
        "phone": "021-6000-3000",
        "responsibility": "负责账号权限、VPN、邮箱和办公设备支持。",
    },
]

FOLLOW_UP_HINTS = [
    "那",
    "这个",
    "这个呢",
    "那个",
    "那个呢",
    "它",
    "他",
    "她",
    "他们",
    "继续",
    "然后",
    "再",
    "还要",
    "还需要",
]

DATA_DIR.mkdir(exist_ok=True)

app = FastAPI(title="企业内部知识库问答助手")


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class AskRequest(BaseModel):
    question: str
    top_k: int = 3
    retrieval_mode: str = "hybrid_rerank"
    history: list[ChatMessage] = []


def load_local_env(env_path: Path):
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and value and key not in os.environ:
            os.environ[key] = value


load_local_env(BASE_DIR.parent / ".env")


def split_text(text: str, chunk_size: int = 350, chunk_overlap: int = 60):
    cleaned = re.sub(r"\r\n?", "\n", text).strip()
    if not cleaned:
        return []

    if "\n## " in cleaned:
        sections = re.split(r"(?=\n##\s+)", f"\n{cleaned}")
        heading_chunks = [section.strip() for section in sections if section.strip()]
        if len(heading_chunks) > 1:
            return heading_chunks

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", cleaned) if part.strip()]
    chunks = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)

        if len(paragraph) <= chunk_size:
            current = paragraph
            continue

        start = 0
        while start < len(paragraph):
            end = min(start + chunk_size, len(paragraph))
            chunks.append(paragraph[start:end])
            if end == len(paragraph):
                break
            start = max(end - chunk_overlap, start + 1)
        current = ""

    if current:
        chunks.append(current)

    return chunks


def load_store():
    if not STORE_PATH.exists():
        return []
    return json.loads(STORE_PATH.read_text(encoding="utf-8"))


def save_store(items: list[dict]):
    STORE_PATH.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def build_llm_client():
    api_key = os.getenv("KIMI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("未检测到 KIMI_API_KEY 或 OPENAI_API_KEY。")

    if os.getenv("KIMI_API_KEY"):
        base_url = os.getenv("KIMI_BASE_URL") or "https://api.moonshot.cn/v1"
        model = os.getenv("KIMI_MODEL") or "moonshot-v1-8k"
    else:
        base_url = os.getenv("OPENAI_BASE_URL")
        model = os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

    return OpenAI(api_key=api_key, base_url=base_url), model


def tokenize(text: str):
    lowered = text.lower()
    raw_parts = re.findall(r"[a-z0-9@._-]+|[\u4e00-\u9fff]+", lowered)
    tokens = []
    for part in raw_parts:
        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            tokens.extend(list(part))
            if len(part) > 1:
                tokens.extend(part[i : i + 2] for i in range(len(part) - 1))
        else:
            tokens.append(part)
    return tokens


def normalize_scores(scores: list[float]):
    if not scores:
        return []
    max_score = max(scores)
    min_score = min(scores)
    if math.isclose(max_score, min_score):
        return [1.0 for _ in scores]
    return [(score - min_score) / (max_score - min_score) for score in scores]


def keyword_retrieve(collection: list[dict], question: str, top_k: int):
    rows = []
    query_tokens = set(tokenize(question))
    for item in collection:
        text_tokens = tokenize(item["text"])
        overlap = sum(1 for token in text_tokens if token in query_tokens)
        score = overlap / max(len(query_tokens), 1)
        rows.append(
            {
                "text": item["text"],
                "metadata": item["metadata"],
                "distance": None,
                "dense_score": score,
            }
        )
    rows.sort(key=lambda x: x["dense_score"], reverse=True)
    return rows[:top_k]


def hybrid_retrieve(collection: list[dict], question: str, top_k: int):
    query_tokens = set(tokenize(question))
    dense_raw_scores = []
    keyword_raw_scores = []
    rows = []

    for item in collection:
        text = item["text"]
        text_tokens = tokenize(text)
        overlap = sum(1 for token in text_tokens if token in query_tokens)
        keyword_score = overlap / max(len(query_tokens), 1)
        phrase_bonus = 1.0 if question[: min(len(question), 8)] in text else 0.0
        dense_score = 0.7 * keyword_score + 0.3 * phrase_bonus

        dense_raw_scores.append(dense_score)
        keyword_raw_scores.append(keyword_score)
        rows.append(
            {
                "text": text,
                "metadata": item["metadata"],
                "distance": None,
                "dense_score": dense_score,
                "keyword_overlap": overlap,
                "bm25_score": keyword_score,
            }
        )

    dense_norm = normalize_scores(dense_raw_scores)
    keyword_norm = normalize_scores(keyword_raw_scores)

    for row, dense_score_norm, keyword_score_norm in zip(rows, dense_norm, keyword_norm):
        row["hybrid_score"] = 0.7 * dense_score_norm + 0.3 * keyword_score_norm

    rows.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return rows[:top_k]


def simple_rerank(question: str, candidates: list[dict]):
    query_tokens = set(tokenize(question))
    reranked = []

    for item in candidates:
        text_tokens = tokenize(item["text"])
        overlap = sum(1 for token in text_tokens if token in query_tokens)
        bonus = overlap / max(len(query_tokens), 1)
        rerank_score = item["hybrid_score"] + 0.2 * bonus
        reranked.append({**item, "keyword_overlap": overlap, "rerank_score": rerank_score})

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked


def trim_history(history: list[ChatMessage]):
    non_empty = [msg for msg in history if msg.content.strip()]
    return non_empty[-MAX_HISTORY_MESSAGES:]


def summarize_text(text: str, max_len: int = 120):
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3] + "..."


def format_conversation_context(history: list[ChatMessage]):
    if not history:
        return ""
    lines = []
    for message in history:
        speaker = "用户" if message.role == "user" else "助手"
        lines.append(f"{speaker}：{message.content.strip()}")
    return "\n".join(lines)


def build_search_query(question: str, history: list[ChatMessage]):
    if not history:
        return question.strip()

    recent_user = ""
    recent_assistant = ""
    for message in reversed(history):
        if not recent_assistant and message.role == "assistant":
            recent_assistant = summarize_text(message.content)
        elif recent_assistant and message.role == "user":
            recent_user = summarize_text(message.content)
            break

    parts = [part for part in [recent_user, recent_assistant, question.strip()] if part]
    return " ".join(parts) if parts else question.strip()


def needs_previous_context(question: str):
    stripped = question.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    short_question = len(stripped) <= 12
    return short_question and any(hint in stripped or hint in lowered for hint in FOLLOW_UP_HINTS)


def route_question(question: str):
    lowered = question.lower()
    directory_keywords = [
        "电话",
        "邮箱",
        "联系人",
        "谁负责",
        "负责人",
        "联系谁",
        "找谁",
        "it",
        "vpn",
        "账号",
        "权限",
        "helpdesk",
    ]
    if any(keyword in lowered or keyword in question for keyword in directory_keywords):
        return "directory_lookup"
    return "knowledge_base_search"


def lookup_directory(search_text: str):
    query_tokens = tokenize(search_text)
    scored = []
    for person in DIRECTORY:
        haystack = " ".join(str(value) for value in person.values()).lower()
        exact_hits = sum(1 for token in query_tokens if len(token) > 1 and token in haystack)
        fuzzy_hits = sum(1 for token in query_tokens if len(token) == 1 and token in haystack)
        score = exact_hits * 3 + fuzzy_hits
        scored.append((score, person))

    scored.sort(key=lambda item: item[0], reverse=True)
    top_score = scored[0][0] if scored else 0
    if top_score > 0:
        matched = [person for score, person in scored if score == top_score]
    else:
        matched = DIRECTORY[:1]

    lines = []
    for person in matched:
        lines.append(
            f"{person['name']}，{person['department']} {person['title']}，"
            f"邮箱 {person['email']}，电话 {person['phone']}。职责：{person['responsibility']}"
        )

    answer = "根据内部通讯录，相关联系人如下：\n" + "\n".join(lines)
    references = [
        {
            "text": line,
            "source": "internal_directory",
            "chunk_index": index,
            "distance": None,
            "dense_score": None,
            "bm25_score": None,
            "hybrid_score": None,
            "rerank_score": None,
            "keyword_overlap": None,
        }
        for index, line in enumerate(lines)
    ]
    return answer, references


def ingest_text(filename: str, text: str):
    chunks = split_text(text)
    if not chunks:
        return 0

    existing = load_store()
    filtered = [item for item in existing if item["metadata"]["source"] != filename]
    for index, chunk in enumerate(chunks):
        filtered.append(
            {
                "id": f"{filename}_chunk_{index}",
                "text": chunk,
                "metadata": {"source": filename, "chunk_index": index},
            }
        )
    save_store(filtered)
    return len(chunks)


def list_documents():
    store = load_store()
    chunk_count_by_source = {}
    for item in store:
        source = item["metadata"]["source"]
        chunk_count_by_source[source] = chunk_count_by_source.get(source, 0) + 1

    local_paths = {path.name: path for path in DATA_DIR.glob("*") if path.is_file()}
    all_sources = sorted(set(chunk_count_by_source) | set(local_paths))

    documents = []
    for source in all_sources:
        path = local_paths.get(source)
        documents.append(
            {
                "filename": source,
                "chunk_count": chunk_count_by_source.get(source, 0),
                "in_store": source in chunk_count_by_source,
                "local_file": source in local_paths,
                "size_bytes": path.stat().st_size if path else None,
            }
        )
    return documents


def delete_document(filename: str):
    safe_name = Path(filename).name
    if not safe_name:
        raise HTTPException(status_code=400, detail="文件名不能为空。")

    store = load_store()
    filtered = [item for item in store if item["metadata"]["source"] != safe_name]
    removed_chunks = len(store) - len(filtered)
    save_store(filtered)

    file_path = DATA_DIR / safe_name
    removed_file = False
    if file_path.exists():
        file_path.unlink()
        removed_file = True

    if removed_chunks == 0 and not removed_file:
        raise HTTPException(status_code=404, detail="未找到目标文档。")

    return {
        "message": "文档已删除。",
        "filename": safe_name,
        "removed_chunks": removed_chunks,
        "removed_file": removed_file,
    }


def answer_with_llm(question: str, history: list[ChatMessage], references: list[dict]):
    conversation_context = format_conversation_context(history) or "无"
    retrieved_context = "\n\n".join([f"[引用{i + 1}] {row['text']}" for i, row in enumerate(references)]) or "无"

    prompt = f"""你是企业内部知识库问答助手。
请优先依据检索结果回答问题。历史对话仅用于补足指代、省略和追问含义，不允许仅依据历史编造制度、联系人或流程。
如果历史和检索内容冲突，以检索内容为准。
如果当前问题依赖历史才能理解，但当前上下文仍不足，请明确回答“当前上下文不足，请补充说明”。
如果检索内容不足，请明确回答“根据当前知识库内容，无法准确回答这个问题”。
如果同一段上下文里存在多个条件，请优先回答与当前问题最直接对应的条款，不要把附加条件当成主答案。

历史对话：
{conversation_context}

检索结果：
{retrieved_context}

当前问题：{question}

输出要求：
1. 先给出简洁答案
2. 最后一行写“引用：”并列出使用到的引用编号；如果没有引用则写“引用：无”
"""

    client, model = build_llm_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


def build_references(rows: list[dict]):
    return [
        {
            "text": row["text"],
            "source": row["metadata"]["source"],
            "chunk_index": row["metadata"]["chunk_index"],
            "distance": row.get("distance"),
            "dense_score": row.get("dense_score"),
            "bm25_score": row.get("bm25_score"),
            "hybrid_score": row.get("hybrid_score"),
            "rerank_score": row.get("rerank_score"),
            "keyword_overlap": row.get("keyword_overlap"),
        }
        for row in rows
    ]


@app.get("/health")
def health():
    return {"status": "ok", "collection": COLLECTION_NAME}


@app.get("/stats")
def stats():
    store = load_store()
    local_docs = sorted([path.name for path in DATA_DIR.glob("*.md")]) + sorted(
        [path.name for path in DATA_DIR.glob("*.txt")]
    )
    return {
        "collection": COLLECTION_NAME,
        "document_count": len(store),
        "document_file_count": len({item["metadata"]["source"] for item in store}),
        "local_files": local_docs,
    }


@app.get("/documents")
def documents():
    return {"documents": list_documents()}


@app.delete("/documents")
def remove_document(filename: str = Query(..., description="待删除的文件名")):
    return delete_document(filename)


@app.post("/clear")
def clear_collection():
    save_store([])
    return {"message": "知识库已清空。"}


@app.post("/ingest-defaults")
def ingest_defaults():
    ingested = []
    for filename in SAMPLE_DOCS:
        path = DATA_DIR / filename
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        chunk_count = ingest_text(filename, text)
        ingested.append({"filename": filename, "chunks": chunk_count})
    if not ingested:
        raise HTTPException(status_code=404, detail="未找到内置样例文档。")
    return {"message": "内置企业文档已入库。", "files": ingested}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空。")

    content = await file.read()
    save_path = DATA_DIR / file.filename
    save_path.write_bytes(content)

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="当前仅支持 UTF-8 编码的 md/txt 文件。") from exc

    chunk_count = ingest_text(file.filename, text)
    if chunk_count == 0:
        raise HTTPException(status_code=400, detail="文档内容为空，无法入库。")

    return {
        "message": "上传并入库成功。",
        "filename": file.filename,
        "chunks": chunk_count,
    }


@app.post("/ask")
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空。")

    history = trim_history(req.history)
    history_used = len(history)
    search_query = build_search_query(question, history)

    if needs_previous_context(question) and not history:
        return {
            "question": question,
            "route": "context_insufficient",
            "retrieval_mode": req.retrieval_mode.strip().lower(),
            "answer": "当前上下文不足，请补充说明。",
            "references": [],
            "search_query": search_query,
            "history_used": history_used,
        }

    route = route_question(search_query)
    if route == "directory_lookup":
        answer, references = lookup_directory(search_query)
        return {
            "question": question,
            "route": route,
            "retrieval_mode": "structured_lookup",
            "answer": answer,
            "references": references,
            "search_query": search_query,
            "history_used": history_used,
        }

    collection = load_store()
    if not collection:
        raise HTTPException(status_code=400, detail="知识库为空，请先导入内置文档或上传文件。")

    mode = req.retrieval_mode.strip().lower()
    if mode == "keyword":
        rows = keyword_retrieve(collection, search_query, req.top_k)
    elif mode == "hybrid":
        rows = hybrid_retrieve(collection, search_query, req.top_k)
    elif mode == "hybrid_rerank":
        rows = simple_rerank(search_query, hybrid_retrieve(collection, search_query, req.top_k))
    else:
        raise HTTPException(status_code=400, detail="retrieval_mode 仅支持 keyword、hybrid、hybrid_rerank。")

    references = build_references(rows)
    answer = answer_with_llm(question, history, references)
    return {
        "question": question,
        "route": route,
        "retrieval_mode": mode,
        "answer": answer,
        "references": references,
        "search_query": search_query,
        "history_used": history_used,
    }
