import requests
import streamlit as st


API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="企业内部知识库问答助手", layout="wide")
st.title("企业内部知识库问答助手")
st.caption("适合简历展示的最小项目：企业制度问答 + 内部通讯录路由 + 单会话多轮记忆")


def safe_json(response):
    try:
        return response.json()
    except Exception:
        return {"detail": response.text}


def get_health():
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        if resp.ok:
            return safe_json(resp)
    except Exception:
        return None
    return None


def get_stats():
    try:
        resp = requests.get(f"{API_BASE}/stats", timeout=5)
        if resp.ok:
            return safe_json(resp)
    except Exception:
        return None
    return None


def get_documents():
    try:
        resp = requests.get(f"{API_BASE}/documents", timeout=10)
        if resp.ok:
            return safe_json(resp).get("documents", [])
    except Exception:
        return []
    return []


def reset_chat_history():
    st.session_state.chat_history = []


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "retrieval_mode" not in st.session_state:
    st.session_state.retrieval_mode = "hybrid_rerank"

if "top_k" not in st.session_state:
    st.session_state.top_k = 3

health = get_health()
stats = get_stats() if health else None
documents = get_documents() if health else []

with st.sidebar:
    st.header("服务状态")
    if health:
        st.success("后端已连接")
        st.caption(f"Collection: {health.get('collection', '-')}")
    else:
        st.error("后端未连接，请先启动 FastAPI")

    if stats:
        st.metric("当前 chunk 数", stats.get("document_count", 0))
        st.metric("当前文档数", stats.get("document_file_count", 0))

    st.divider()
    st.header("检索设置")
    st.session_state.top_k = st.slider("Top-K", min_value=1, max_value=5, value=st.session_state.top_k)
    retrieval_options = {
        "keyword": "Keyword Retrieval",
        "hybrid": "Hybrid Retrieval",
        "hybrid_rerank": "Hybrid + Rerank",
    }
    retrieval_mode = st.selectbox(
        "检索模式",
        options=list(retrieval_options.keys()),
        index=list(retrieval_options.keys()).index(st.session_state.retrieval_mode),
        format_func=lambda item: retrieval_options[item],
    )
    st.session_state.retrieval_mode = retrieval_mode

    st.divider()
    st.header("数据准备")
    if st.button("导入内置企业文档", use_container_width=True):
        try:
            resp = requests.post(f"{API_BASE}/ingest-defaults", timeout=60)
            data = safe_json(resp)
            if resp.ok:
                st.success(data["message"])
                st.rerun()
            else:
                st.error(data.get("detail", resp.text))
        except requests.RequestException as exc:
            st.error(f"导入失败：{exc}")

    uploaded_file = st.file_uploader("上传 md/txt 文档", type=["md", "txt"])
    if uploaded_file and st.button("上传并入库", use_container_width=True):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        try:
            resp = requests.post(f"{API_BASE}/upload", files=files, timeout=60)
            data = safe_json(resp)
            if resp.ok:
                st.success(data["message"])
                st.rerun()
            else:
                st.error(data.get("detail", resp.text))
        except requests.RequestException as exc:
            st.error(f"上传失败：{exc}")

    if st.button("清空知识库", use_container_width=True):
        try:
            resp = requests.post(f"{API_BASE}/clear", timeout=30)
            data = safe_json(resp)
            if resp.ok:
                st.success(data["message"])
                st.rerun()
            else:
                st.error(data.get("detail", resp.text))
        except requests.RequestException as exc:
            st.error(f"清空失败：{exc}")

    if st.button("清空当前对话", use_container_width=True):
        reset_chat_history()
        st.rerun()

    st.divider()
    st.header("文档管理")
    if documents:
        for doc in documents:
            status = []
            if doc.get("in_store"):
                status.append(f"{doc.get('chunk_count', 0)} chunks")
            if doc.get("local_file"):
                status.append("本地文件")

            with st.expander(doc["filename"]):
                st.caption(" | ".join(status) if status else "未入库")
                if st.button(f"删除 {doc['filename']}", key=f"delete_{doc['filename']}", use_container_width=True):
                    try:
                        resp = requests.delete(
                            f"{API_BASE}/documents",
                            params={"filename": doc["filename"]},
                            timeout=30,
                        )
                        data = safe_json(resp)
                        if resp.ok:
                            st.success(data["message"])
                            st.rerun()
                        else:
                            st.error(data.get("detail", resp.text))
                    except requests.RequestException as exc:
                        st.error(f"删除失败：{exc}")
    else:
        st.caption("当前没有可管理的文档。")


st.subheader("示例问题")
st.write("- 新员工试用期多久？")
st.write("- 那转正要提前多久申请？")
st.write("- VPN 连不上该联系谁？")
st.write("- 那他的邮箱是什么？")

st.subheader("对话")

action_col, info_col = st.columns([1, 3])
with action_col:
    if st.button("清空上下文", use_container_width=True):
        reset_chat_history()
        st.rerun()
with info_col:
    st.caption("只清空当前会话历史，不会删除已上传文档或知识库内容。")

for index, message in enumerate(st.session_state.chat_history):
    with st.chat_message("user" if message["role"] == "user" else "assistant"):
        st.write(message["content"])
        if message["role"] == "assistant":
            route = message.get("route", "-")
            retrieval_mode = message.get("retrieval_mode", "-")
            search_query = message.get("search_query", "")
            history_used = message.get("history_used", 0)
            st.caption(
                f"route={route} | retrieval_mode={retrieval_mode} | history_used={history_used}"
            )
            if search_query:
                st.caption(f"search_query={search_query}")

            references = message.get("references", [])
            for ref_index, ref in enumerate(references):
                score_parts = []
                if isinstance(ref.get("distance"), (int, float)):
                    score_parts.append(f"distance={ref['distance']:.4f}")
                if isinstance(ref.get("dense_score"), (int, float)):
                    score_parts.append(f"dense={ref['dense_score']:.4f}")
                if isinstance(ref.get("bm25_score"), (int, float)):
                    score_parts.append(f"keyword={ref['bm25_score']:.4f}")
                if isinstance(ref.get("hybrid_score"), (int, float)):
                    score_parts.append(f"hybrid={ref['hybrid_score']:.4f}")
                if isinstance(ref.get("rerank_score"), (int, float)):
                    score_parts.append(f"rerank={ref['rerank_score']:.4f}")
                if isinstance(ref.get("keyword_overlap"), int):
                    score_parts.append(f"overlap={ref['keyword_overlap']}")

                label = f"{ref['source']} | chunk {ref['chunk_index']}"
                if score_parts:
                    label += " | " + " | ".join(score_parts)
                with st.expander(label, expanded=False):
                    st.write(ref["text"])


question = st.chat_input("输入问题，支持基于上一轮继续追问")

if question:
    st.session_state.chat_history.append({"role": "user", "content": question})

    history_payload = [
        {"role": item["role"], "content": item["content"]}
        for item in st.session_state.chat_history[:-1]
    ]

    try:
        resp = requests.post(
            f"{API_BASE}/ask",
            json={
                "question": question,
                "top_k": st.session_state.top_k,
                "retrieval_mode": st.session_state.retrieval_mode,
                "history": history_payload,
            },
            timeout=120,
        )
        data = safe_json(resp)
        if resp.ok:
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": data["answer"],
                    "route": data.get("route"),
                    "retrieval_mode": data.get("retrieval_mode"),
                    "search_query": data.get("search_query"),
                    "history_used": data.get("history_used", 0),
                    "references": data.get("references", []),
                }
            )
        else:
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": data.get("detail", resp.text),
                    "route": "error",
                    "retrieval_mode": st.session_state.retrieval_mode,
                    "search_query": "",
                    "history_used": len(history_payload),
                    "references": [],
                }
            )
    except requests.RequestException as exc:
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": f"提问失败：{exc}",
                "route": "error",
                "retrieval_mode": st.session_state.retrieval_mode,
                "search_query": "",
                "history_used": len(history_payload),
                "references": [],
            }
        )

    st.rerun()
