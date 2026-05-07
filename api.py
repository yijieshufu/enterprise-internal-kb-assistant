import math
import os
import re
import json
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from openai import OpenAI
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STORE_PATH = BASE_DIR / "kb_store.json"
COLLECTION_NAME = "enterprise_internal_kb"

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

DATA_DIR.mkdir(exist_ok=True)

app = FastAPI(title="企业内部知识库问答助手")


class AskRequest(BaseModel):
    question: str
    top_k: int = 3
    retrieval_mode: str = "hybrid_rerank"


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
    return [token for token in re.split(r"[\s,，。！？；:：/()\-\[\]]+", text.lower()) if token]


def normalize_scores(scores: list[float]):
    if not scores:
        return []
    max_score = max(scores)
    min_score = min(scores)
    if math.isclose(max_score, min_score):
        return [1.0 for _ in scores]
    return [(score - min_score) / (max_score - min_score) for score in scores]


def dense_retrieve(collection, question: str, top_k: int):
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


def hybrid_retrieve(collection, question: str, top_k: int):
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
        reranked.append(
            {
                **item,
                "keyword_overlap": overlap,
                "rerank_score": rerank_score,
            }
        )

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked


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
    ]
    if any(keyword in question for keyword in directory_keywords) or "helpdesk" in lowered:
        return "directory_lookup"
    return "knowledge_base_search"


def lookup_directory(question: str):
    matched = []
    for person in DIRECTORY:
        haystack = " ".join(person.values()).lower()
        if any(token in haystack for token in tokenize(question)):
            matched.append(person)

    if not matched:
        matched = DIRECTORY[:2]

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


def answer_with_llm(question: str, references: list[dict]):
    contexts = [row["text"] for row in references]
    context_text = "\n\n".join([f"[引用{i + 1}] {doc}" for i, doc in enumerate(contexts)])

    prompt = f"""你是企业内部知识库问答助手。
请严格基于给定上下文回答问题，不要编造制度、联系人或流程。
如果上下文不足，请明确回答“根据当前知识库内容，无法准确回答这个问题”。

上下文：
{context_text}

用户问题：{question}

输出要求：
1. 先给出简洁答案
2. 最后一行写“引用：”并列出使用到的引用编号
"""

    client, model = build_llm_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


@app.get("/health")
def health():
    return {"status": "ok", "collection": COLLECTION_NAME}


@app.get("/stats")
def stats():
    local_docs = sorted([path.name for path in DATA_DIR.glob("*.md")]) + sorted(
        [path.name for path in DATA_DIR.glob("*.txt")]
    )
    store = load_store()
    return {
        "collection": COLLECTION_NAME,
        "document_count": len(store),
        "local_files": local_docs,
    }


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

    route = route_question(question)
    if route == "directory_lookup":
        answer, references = lookup_directory(question)
        return {
            "question": question,
            "route": route,
            "retrieval_mode": "structured_lookup",
            "answer": answer,
            "references": references,
        }

    collection = load_store()
    if len(collection) == 0:
        raise HTTPException(status_code=400, detail="知识库为空，请先导入内置文档或上传文件。")

    mode = req.retrieval_mode.strip().lower()
    if mode == "keyword":
        rows = dense_retrieve(collection, question, req.top_k)
    elif mode == "hybrid":
        rows = hybrid_retrieve(collection, question, req.top_k)
    elif mode == "hybrid_rerank":
        rows = simple_rerank(question, hybrid_retrieve(collection, question, req.top_k))
    else:
        raise HTTPException(status_code=400, detail="retrieval_mode 仅支持 keyword、hybrid、hybrid_rerank。")

    references = [
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

    answer = answer_with_llm(question, references)
    return {
        "question": question,
        "route": route,
        "retrieval_mode": mode,
        "answer": answer,
        "references": references,
    }
