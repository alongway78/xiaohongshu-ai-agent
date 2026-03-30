import os
from pathlib import Path


import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

# 社区包中的组件
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma

# 文本分割器（在 langchain 主包中）
from langchain_text_splitters import CharacterTextSplitter
# ================== 1. 初始化大模型 ==================
llm = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY")  # 从环境变量读取
)

# ================== 2. 初始化 Embedding 模型（用于 RAG） ==================
embeddings = OpenAIEmbeddings(
    model="text-embedding-v1",   # 改为正确的模型名
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    check_embedding_ctx_length=False   # 关键参数！关闭OpenAI特有的长度检查
)
# ================== 3. RAG 知识库检索器 ==================
_retriever = None

def get_retriever():
    """加载知识库文档并构建检索器（懒加载）"""
    global _retriever
    if _retriever is not None:
        return _retriever

    knowledge_dir = Path("knowledge_base")
    if not knowledge_dir.exists() or not any(knowledge_dir.glob("*.txt")):
        print("⚠️ 知识库目录不存在或无 txt 文件，RAG 功能将不可用")
        return None

    try:
        docs = []
        for file_path in knowledge_dir.glob("*.txt"):
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs.extend(loader.load())
        if not docs:
            return None

        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = text_splitter.split_documents(docs)
        # 过滤掉空内容的文档块
        split_docs = [doc for doc in split_docs if doc.page_content.strip()]
        try:
            test_emb = embeddings.embed_documents(["测试文本"])
            print("Embedding 测试成功")
        except Exception as e:
            print(f"Embedding 测试失败: {e}")

        vectorstore = Chroma.from_documents(split_docs, embeddings)
        _retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        print("✅ 知识库加载成功，RAG 已启用")
        return _retriever
    except Exception as e:
        print(f"⚠️ 知识库加载失败: {e}，RAG 功能不可用")
        return None

def retrieve_similar_cases(query: str) -> str:
    """根据查询检索相似案例，返回拼接后的文本"""
    retriever = get_retriever()
    if retriever is None:
        return ""
    try:
        docs = retriever.invoke(query)
        if not docs:
            return ""
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        print(f"⚠️ 检索失败: {e}")
        return ""

# ================== 4. 意图识别模块 ==================
intent_prompt = PromptTemplate(
    template="""
你是一个信息提取器。用户输入：{keyword}

你需要提取：
- product: 产品名称
- audience: 目标人群（如“学生党”“上班族”“妈妈”等）
- style: 期望风格（只能是“种草”、“测评”、“干货”之一）

如果用户未指定人群或风格，根据产品特点合理推测。

严格要求：
- 只输出一个 JSON 对象，不要有任何其他文字、注释、Markdown 代码块。
- 输出示例：{{"product": "护手霜", "audience": "学生党", "style": "种草"}}
""",
    input_variables=["keyword"]
)
intent_chain = intent_prompt | llm | JsonOutputParser()

# ================== 5. A/B 文案生成（带 RAG 参考） ==================
# 版本A：标准种草风（加入 similar_cases 参考）
template_a = """
你是一个小红书文案专家。

以下是一些相关的爆款案例供你参考风格和结构（可能为空）：
{similar_cases}

产品：{product}
目标人群：{audience}
风格：{style}

请写一篇200字左右的小红书文案，包含标题、正文、emoji、话题标签。语气亲切，多用感叹号和口语化表达。
"""
prompt_a = PromptTemplate(
    template=template_a,
    input_variables=["product", "audience", "style", "similar_cases"]
)
chain_a = prompt_a | llm | StrOutputParser()

# 版本B：活泼热词风（加入 similar_cases 参考）
template_b = """
你是一个小红书爆款写手。

以下是一些相关的爆款案例供你参考风格和结构（可能为空）：
{similar_cases}

产品：{product}
面向人群：{audience}
要求风格：{style}

请写一篇200字左右的小红书文案，要求：
- 开头用“救命！”或“谁懂啊！”
- 多用“绝绝子”“YYDS”等热词
- 至少5个emoji
- 结尾带互动引导（如“评论区蹲的姐妹扣1”）
"""
prompt_b = PromptTemplate(
    template=template_b,
    input_variables=["product", "audience", "style", "similar_cases"]
)
chain_b = prompt_b | llm | StrOutputParser()

# ================== 6. 质量评分模块 ==================
score_prompt = PromptTemplate(
    template="""
你是一个文案质量评估专家。文案内容：{copy}

请按以下标准评分（1-3分，总分10分）：
- 标题吸引力（1-3）
- 正文感染力（1-3）
- emoji与排版（1-2）
- 话题标签相关性（1-2）

输出一个JSON对象，格式如下，不要有任何其他文字：
{{"score": 总分, "title_score": 分值, "content_score": 分值, "style_score": 分值, "tag_score": 分值, "suggestions": "优化建议"}}
""",
    input_variables=["copy"]
)
score_chain = score_prompt | llm | JsonOutputParser()

def score_copy(copy):
    try:
        return score_chain.invoke({"copy": copy})
    except Exception as e:
        print(f"评分失败：{e}")
        return {"score": 0, "suggestions": "评分失败"}

# ================== 7. 选择器 ==================
def select_best(copy_a, copy_b, score_a, score_b):
    score_val_a = int(score_a.get("score", 0))
    score_val_b = int(score_b.get("score", 0))
    if score_val_a >= score_val_b:
        return copy_a, score_a
    else:
        return copy_b, score_b

# ================== 8. 主流程（集成 RAG） ==================
def run_agent(keyword: str):
    print(f"输入关键词：{keyword}")

    # 1. 意图识别
    intent = intent_chain.invoke({"keyword": keyword})
    product = intent.get("product", "护手霜")
    audience = intent.get("audience", "精致女生")
    style = intent.get("style", "种草")
    print(f"识别结果：产品={product}, 人群={audience}, 风格={style}")

    # 2. RAG 检索：根据用户输入或产品名检索相似案例
    similar_cases = retrieve_similar_cases(keyword) or retrieve_similar_cases(product)
    if similar_cases:
        case_count = len(similar_cases.split('\n\n'))
        print(f"检索到 {case_count} 篇参考案例")
    else:
        print("未检索到参考案例，将直接生成")
    # 3. 生成两个版本（传入 similar_cases）
    copy_a = chain_a.invoke({
        "product": product,
        "audience": audience,
        "style": style,
        "similar_cases": similar_cases
    })
    copy_b = chain_b.invoke({
        "product": product,
        "audience": audience,
        "style": style,
        "similar_cases": similar_cases
    })
    print("两个版本生成完成。")

    # 4. 评分
    score_a = score_copy(copy_a)
    score_b = score_copy(copy_b)
    print(f"版本A得分：{score_a.get('score')}, 版本B得分：{score_b.get('score')}")

    # 5. 择优
    best_copy, best_score = select_best(copy_a, copy_b, score_a, score_b)
    print("择优完成。")

    return {
        "best_copy": best_copy,
        "best_score": best_score,
        "version_a": copy_a,
        "version_b": copy_b,
        "score_a": score_a,
        "score_b": score_b
    }

# ================== 9. 测试入口 ==================
if __name__ == "__main__":
    keyword = "请帮我生成有关小红帽的文案"
    result = run_agent(keyword)

    print("\n" + "="*50)
    print("【最佳文案】")
    print(result["best_copy"])
    print("\n【评分报告】")
    print(f"得分：{result['best_score']['score']}")
    print(f"建议：{result['best_score']['suggestions']}")