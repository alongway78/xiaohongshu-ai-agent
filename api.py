from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import run_agent

app = FastAPI(title="小红书文案智能体")

# 添加 CORS 中间件，允许所有来源和所有方法
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Request(BaseModel):
    keyword: str

class Response(BaseModel):
    best_copy: str
    score: int
    suggestions: str

@app.post("/generate")
async def generate(request: Request):
    result = run_agent(request.keyword)
    return Response(
        best_copy=result["best_copy"],
        score=result["best_score"]["score"],
        suggestions=result["best_score"]["suggestions"]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)