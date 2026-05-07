# 企业内部知识库问答助手

这是一个适合简历展示的 `RAG + 轻量 Agent Router` 项目。

功能：

- 企业制度文档上传、切分、向量入库
- `Keyword / Hybrid / Hybrid + Rerank` 三种检索模式
- 基于知识库生成回答，并返回引用片段
- 对“找谁处理 VPN / 权限 / 联系方式”这类问题走内部通讯录路由

## 目录

- `api.py`：FastAPI 后端
- `web.py`：Streamlit 前端
- `data/`：内置企业制度样例文档

## 启动

```bash
pip install -r requirements.txt
uvicorn api:app --reload
streamlit run web.py
```

## 说明

程序会优先读取项目上一层目录下的 `.env`，兼容：

- `KIMI_API_KEY`
- `KIMI_BASE_URL`
- `KIMI_MODEL`
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`

## 推荐演示问题

- 新员工试用期多久，转正流程是什么？
- 差旅发票最晚什么时候提交？
- VPN 连不上该联系谁？
- 报销系统每月什么时间截止？
