# 小红书文案智能体（Xiaohongshu AI Copywriter）

基于 LangChain 和阿里云百炼大模型构建的智能文案生成 Agent，支持意图识别、双版本文案生成、质量评分与自动择优，并提供 API 服务和可视化界面。同时集成了 RAG 知识库，可参考历史爆款案例生成更真实的文案。

## ✨ 功能特性

- 🎯 **意图识别**：将用户自然语言输入（如“小白鞋”）自动解析为产品、目标人群、风格参数。
- ✍️ **双版本文案生成**：使用两套不同提示词同时生成文案，模拟 A/B 测试，自动选择更优版本。
- 📊 **质量评分**：调用大模型从标题、正文、排版、标签四维度打分，并提供优化建议。
- 🏆 **自动择优**：根据评分选择更优文案输出。
- 📚 **RAG 知识库**：可上传历史爆款文案，生成时检索相似案例，提升文案质量。
- 🚀 **API 服务**：基于 FastAPI 提供 RESTful 接口，附带 Swagger 文档。
- 🎨 **可视化前端**：简洁美观的网页界面，方便演示和交互。

## 🛠 技术栈

- LangChain、FastAPI、Uvicorn
- 阿里云百炼 Qwen 大模型（`qwen-plus`）
- 阿里云百炼 Embedding 模型（`text-embedding-v1`）
- Python 3.11+
- 前端：HTML5 + CSS3 + JavaScript (Fetch API)

## 📦 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/alongway78/xiaohongshu-ai-agent.git
cd xiaohongshu-ai-agent
```

### 2. 安装依赖

建议使用虚拟环境（可选）：
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

安装所需包：
```bash
python -m pip install -r requirements.txt
```

### 3. 配置环境变量

创建 `.env` 文件（与 `agent.py` 同级），填入你的阿里云百炼 API Key：
```ini
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx
```

### 4. （可选）准备知识库

在项目根目录下创建 `knowledge_base` 文件夹，放入 `.txt` 格式的小红书爆款文案（UTF-8 编码）。程序会自动加载并构建向量索引。

### 5. 启动 API 服务

```bash
python api.py
```

服务启动后，访问 `http://localhost:8000/docs` 查看 Swagger 文档，或使用前端界面 `http://localhost:8080/test.html`（需另启静态服务器，见下文）。

### 6. 使用前端界面（可选）

在另一个终端启动静态服务器：
```bash
python -m http.server 8080
```
然后打开浏览器访问 `http://localhost:8080/test.html`。

### 7. 测试 API

```bash
curl -X POST http://localhost:8000/generate -H "Content-Type: application/json" -d '{"keyword": "小白鞋"}'
```

## 📁 项目结构

```
.
├── agent.py                # 智能体核心逻辑（LangChain）
├── api.py                  # FastAPI 服务入口
├── test.html               # 可视化前端页面
├── requirements.txt        # Python 依赖
├── .env                    # 环境变量（需自行创建）
├── .gitignore
├── knowledge_base/         # 知识库文件夹（可自行添加 .txt 文件）
└── README.md
```

## 🧪 示例输出

输入关键词：`小白鞋`

返回结果示例：
```json
{
  "best_copy": "标题：学生党听好！！这双小白鞋让我被追着问链接3次！！👟\n\n...",
  "score": 9,
  "suggestions": "标题吸睛有力，正文场景真实，建议增加实拍图..."
}
```

## 🧠 项目亮点与难点

- **意图识别**：使用 PromptTemplate + JsonOutputParser 实现结构化输出，并通过异常处理保证鲁棒性。
- **A/B 测试**：双版本生成与自动择优，模拟线上实验流程。
- **RAG 知识库**：基于 Chroma 向量库 + 阿里云百炼 Embedding，解决文案风格不真实的问题。
- **工程化**：FastAPI 封装、CORS 支持、环境变量管理、Docker 可部署。
- **难点解决**：LangChain 与阿里云百炼的兼容性（`check_embedding_ctx_length=False`）、文档空块过滤、跨域问题等。

## 📝 后续计划

- [ ] Docker 化部署
- [ ] 支持多轮对话记忆
- [ ] 接入更多平台（抖音、B站）
- [ ] 使用 LangGraph 重构多智能体

## 📄 许可证

本项目仅供学习交流使用。

## 📧 联系

如有问题，欢迎提 Issue 或邮件联系：2746988278@qq.com
```
