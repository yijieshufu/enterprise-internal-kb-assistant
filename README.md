# 企业内部知识库问答助手

一个面向企业内部制度文档的最小可演示 `RAG + 轻量 Agent Router` 项目，适合放到 GitHub 和简历中展示。

## 项目简介

很多企业内部资料分散在员工手册、报销制度、IT 支持说明、FAQ 文档里。这个项目的目标是搭建一个可运行的知识库问答助手，支持：

- 上传企业内部文档并完成入库
- 基于知识库进行问答
- 返回答案引用来源
- 对不同类型问题做简单路由

当前版本重点解决的是两个场景：

1. 企业制度类问题  
   例如：试用期多久、报销截止时间、转正需要准备什么

2. 内部联系人查询类问题  
   例如：VPN 连不上联系谁、权限开通找谁、IT 支持邮箱是什么

## 当前已完成功能

### 1. 知识库文档导入

- 支持上传 `md / txt` 文档
- 支持读取内置样例企业文档
- 自动进行文本切分并写入本地知识库

### 2. RAG 问答流程

- 用户输入问题
- 后端检索相关文档片段
- 调用大模型生成最终回答
- 返回引用内容，便于溯源

### 3. 检索策略

当前提供 3 种检索模式：

- `Keyword Retrieval`
- `Hybrid Retrieval`
- `Hybrid + Rerank`

可以在前端页面中切换不同检索模式进行对比。

### 4. 轻量 Agent Router

项目增加了一个简单的问题路由逻辑：

- 制度 / 流程类问题 -> `knowledge_base_search`
- 联系人 / 权限 / VPN 类问题 -> `directory_lookup`

这样项目就不只是“检索后回答”，而是具备了最小的多策略分流能力。

### 5. 前后端联调

- `FastAPI` 提供接口
- `Streamlit` 提供演示页面
- 支持在页面中直接提问、导入文档、查看引用内容

## 项目结构

```text
enterprise-internal-kb-assistant/
├─ api.py               # FastAPI 后端
├─ web.py               # Streamlit 前端
├─ requirements.txt     # 项目依赖
├─ .gitignore
├─ README.md
└─ data/                # 内置企业样例文档
   ├─ employee_handbook.md
   ├─ reimbursement_policy.md
   └─ it_service_guide.md
```

## 技术栈

- Python
- FastAPI
- Streamlit
- OpenAI / Kimi 兼容接口
- 本地 JSON 知识库存储

## 已内置的演示数据

项目自带 3 份企业内部样例文档：

- `employee_handbook.md`：员工手册
- `reimbursement_policy.md`：费用报销制度
- `it_service_guide.md`：IT 服务指南

另外还内置了一份简单的内部通讯录数据，用于联系人查询路由演示。

## 启动方式

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置模型 API Key

程序会优先读取项目上一层目录中的 `.env` 文件，支持以下变量：

```env
KIMI_API_KEY=your_key
KIMI_BASE_URL=https://api.moonshot.cn/v1
KIMI_MODEL=moonshot-v1-8k
```

也兼容：

```env
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=your_base_url
OPENAI_MODEL=gpt-4o-mini
```

### 3. 启动后端

```bash
uvicorn api:app --reload
```

### 4. 启动前端

```bash
streamlit run web.py
```

## 推荐测试问题

### 知识库问答

- 新员工试用期多久？
- 转正要准备什么？
- 差旅报销最晚什么时候提交发票？
- 报销系统每月什么时候截止？

### 路由问答

- VPN 连不上该联系谁？
- 找谁开通账号权限？

## 当前版本特点

- 功能闭环完整：上传 -> 入库 -> 检索 -> 回答 -> 引用
- 有最小路由能力，不只是单纯 RAG
- 适合后续继续扩展：
  - Embedding 检索
  - 多知识库路由
  - Web Search
  - SQL / API 工具调用
  - 评测体系

## 后续可扩展方向

- 接入真实向量数据库
- 增加 Embedding 检索与召回对比
- 增加 Query Rewrite
- 增加更完整的 Agent 工具调用链
- 增加评测集与效果评估

## 项目定位

这个项目当前定位为：

**适合简历展示的企业内部知识库问答 MVP**

它已经具备：

- 明确业务场景
- 可运行前后端
- 基本 RAG 能力
- 最小 Agent 路由能力

适合继续打磨成完整作品集项目。
