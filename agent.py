import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

# ================== 1. 初始化大模型（阿里云百炼 Qwen） ==================
# llm = ChatOpenAI(
#     model="qwen-plus",                       # 百炼可用模型
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#     api_key="sk-"   # 替换成你自己的Key
# )
import os
from dotenv import load_dotenv

load_dotenv()  # 加载 .env 文件

# 初始化模型时使用环境变量
llm = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY")  # 从环境变量获取
)
# ================== 2. 意图识别模块 ==================
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

# ================== 3. A/B 文案生成 ==================
# 版本A：标准种草风
template_a = """
你是一个小红书文案专家。产品：{product}，目标人群：{audience}，风格：{style}。
请写一篇200字左右的小红书文案，包含标题、正文、emoji、话题标签。语气亲切，多用感叹号和口语化表达。
"""
prompt_a = PromptTemplate(template=template_a, input_variables=["product", "audience", "style"])
chain_a = prompt_a | llm | StrOutputParser()

# 版本B：活泼热词风
template_b = """
你是一个小红书爆款写手。产品：{product}，面向：{audience}，要求风格：{style}。
请写一篇200字左右的小红书文案，要求：开头用“救命！”或“谁懂啊！”，多用“绝绝子”“YYDS”等热词，至少5个emoji，结尾带互动引导（如“评论区蹲的姐妹扣1”）。
"""
prompt_b = PromptTemplate(template=template_b, input_variables=["product", "audience", "style"])
chain_b = prompt_b | llm | StrOutputParser()

# ================== 4. 质量评分模块 ==================
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

# 评分函数（带异常处理）
def score_copy(copy):
    try:
        return score_chain.invoke({"copy": copy})
    except Exception as e:
        print(f"评分失败：{e}")
        return {"score": 0, "suggestions": "评分失败"}

# ================== 5. 选择器 ==================
def select_best(copy_a, copy_b, score_a, score_b):
    score_val_a = int(score_a.get("score", 0))
    score_val_b = int(score_b.get("score", 0))
    if score_val_a >= score_val_b:
        return copy_a, score_a
    else:
        return copy_b, score_b

# ================== 6. 主流程 ==================
def run_agent(keyword: str):
    print(f"输入关键词：{keyword}")

    # 意图识别
    intent = intent_chain.invoke({"keyword": keyword})
    product = intent.get("product", "护手霜")
    audience = intent.get("audience", "精致女生")
    style = intent.get("style", "种草")
    print(f"识别结果：产品={product}, 人群={audience}, 风格={style}")

    # 生成两个版本
    copy_a = chain_a.invoke({"product": product, "audience": audience, "style": style})
    copy_b = chain_b.invoke({"product": product, "audience": audience, "style": style})
    print("两个版本生成完成。")

    # 评分
    score_a = score_copy(copy_a)
    score_b = score_copy(copy_b)
    print(f"版本A得分：{score_a.get('score')}, 版本B得分：{score_b.get('score')}")

    # 选择最优
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

# ================== 7. 测试入口 ==================
if __name__ == "__main__":
    keyword = "请帮我生成有关小红帽的文案"
    result = run_agent(keyword)

    print("\n" + "="*50)
    print("【最佳文案】")
    print(result["best_copy"])
    print("\n【评分报告】")
    print(f"得分：{result['best_score']['score']}")
    print(f"建议：{result['best_score']['suggestions']}")