import requests
import streamlit as st


API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="企业内部知识库问答助手", layout="wide")
st.title("企业内部知识库问答助手")
st.caption("适合简历展示的最小项目：企业制度问答 + 内部通讯录路由")


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
    st.header("数据准备")

    if st.button("导入内置企业文档", use_container_width=True):
        try:
            resp = requests.post(f"{API_BASE}/ingest-defaults", timeout=60)
            data = safe_json(resp)
            if resp.ok:
                st.success(data["message"])
                for item in data.get("files", []):
                    st.write(f"{item['filename']} -> {item['chunks']} chunks")
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
                st.write({"filename": data.get("filename"), "chunks": data.get("chunks")})
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
            else:
                st.error(data.get("detail", resp.text))
        except requests.RequestException as exc:
            st.error(f"清空失败：{exc}")

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
st.write("- 新员工试用期多久，转正要准备什么？")
st.write("- 差旅发票多久内提交？")
st.write("- VPN 连不上该找谁？")

question = st.text_input("输入问题", value="差旅报销最晚什么时候提交发票？")
top_k = st.slider("Top-K", min_value=1, max_value=5, value=3)
retrieval_mode = st.selectbox(
    "检索模式",
    options=[
        ("keyword", "Keyword Retrieval"),
        ("hybrid", "Hybrid Retrieval"),
        ("hybrid_rerank", "Hybrid + Rerank"),
    ],
    format_func=lambda x: x[1],
)

if st.button("开始提问", type="primary"):
    try:
        resp = requests.post(
            f"{API_BASE}/ask",
            json={
                "question": question,
                "top_k": top_k,
                "retrieval_mode": retrieval_mode[0],
            },
            timeout=120,
        )
        data = safe_json(resp)
        if resp.ok:
            st.subheader("回答")
            st.write(data["answer"])
            st.caption(
                f"route={data.get('route', '-')} | retrieval_mode={data.get('retrieval_mode', retrieval_mode[0])}"
            )

            st.subheader("引用内容")
            for ref in data["references"]:
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
                with st.expander(label):
                    st.write(ref["text"])
        else:
            st.error(data.get("detail", resp.text))
    except requests.RequestException as exc:
        st.error(f"提问失败：{exc}")
