import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import sys
import os
from contextlib import asynccontextmanager
import json
from datetime import datetime

# 1. 导入你的核心业务逻辑
# 确保 customer_service_example121.py 在同一目录下
try:
    from customer_service_example121 import PhoenixTrackedRAGSystem
except ImportError as e:
    print(f"严重错误: 无法导入核心模块。请确保 'customer_service_example121.py' 在当前目录下。")
    print(f"错误详情: {e}")
    sys.exit(1)

# --- 全局变量 ---
# 用于存储你的 RAG 系统实例，避免每次请求都重新初始化
rag_system_instance: Optional[PhoenixTrackedRAGSystem] = None

# --- 生命周期管理 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    服务器启动和关闭时的逻辑。
    在启动时初始化 RAG 系统，确保只加载一次模型和文档。
    """
    global rag_system_instance
    print("正在启动 API 服务器...")
    print("正在初始化 RAG 系统 (这可能需要几秒钟)...")
    
    try:
        # 实例化系统
        rag_system_instance = PhoenixTrackedRAGSystem()
        # 加载数据和配置
        rag_system_instance.initialize_fire_system()
        print("✅ RAG 系统初始化完成，准备接收请求！")
    except Exception as e:
        print(f"❌ RAG 系统初始化失败: {e}")
        # 这里不退出，允许服务器启动，但后续请求会报错
    
    yield
    
    # 服务器关闭时的清理工作（如果有）
    print("服务器正在关闭...")
    rag_system_instance = None

# --- FastAPI 应用定义 ---
app = FastAPI(
    title="智慧消防数字孪生指挥系统 API",
    description="用于连接 Unity 前端与 RAG 多智能体后端的接口服务",
    version="1.0.0",
    lifespan=lifespan
)

# --- CORS 设置 (至关重要) ---
# 允许 Unity (或其他前端) 跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中建议改为具体的域名，演示环境用 "*" 允许所有
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic 数据模型 ---
# 定义请求和响应的数据结构，这会让你的 API 文档自动生成

class QueryRequest(BaseModel):
    question: str = Field(..., description="用户输入的火灾情况描述", example="3号楼5楼发生火灾，有人员被困")

class ResourceInfo(BaseModel):
    resource_id: str
    resource_type: str
    location: str

class CoordinationResult(BaseModel):
    status: str
    message: str
    allocated: List[ResourceInfo]
    unavailable: List[Dict[str, Any]]
    guidance: List[Dict[str, Any]]

class PipelineResult(BaseModel):
    retrieved_docs_count: int
    generated_response: Dict[str, Any]  # 你的结构化响应（阶段一、阶段二等）
    resource_coordination: Dict[str, Any]
    evaluation: Dict[str, Any]
    overall_score: float

class EmergencyResponse(BaseModel):
    question: str
    timestamp: str
    pipeline_result: PipelineResult

# --- API 接口 ---

@app.get("/")
async def root():
    """健康检查接口"""
    return {"status": "online", "system": "Fire Emergency RAG Agent"}

@app.post("/api/emergency", response_model=EmergencyResponse)
async def handle_emergency(request: QueryRequest):
    """
    处理紧急火灾情况的核心接口。
    Unity 发送文本，这里返回完整的决策 JSON。
    """
    global rag_system_instance
    
    if not rag_system_instance:
        raise HTTPException(status_code=503, detail="系统尚未初始化完成，请稍后再试")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    try:
        print(f"收到请求: {request.question}")
        
        # 调用你的核心逻辑
        # 注意：这里直接调用同步方法，FastAPI 会自动在线程池中运行它
        result = rag_system_instance.process_fire_question(request.question)
        
        return result
        
    except Exception as e:
        print(f"处理请求时发生错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/resources/status")
async def get_resource_status():
    """
    获取当前所有资源的实时状态。
    用于 Unity 中的仪表盘显示（无需提问即可查看资源）。
    """
    global rag_system_instance
    
    if not rag_system_instance:
        raise HTTPException(status_code=503, detail="系统尚未初始化")
        
    try:
        # 访问 ResourceCoordinator 获取状态
        status = rag_system_instance.resource_coordinator.get_resource_status_report()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reset")
async def reset_system():
    """
    重置系统状态（例如演示结束后清空资源占用）。
    """
    global rag_system_instance
    if rag_system_instance:
        # 这里你可以添加重置逻辑，目前我们可以简单地重新初始化资源
        # 简单粗暴的方法：重新实例化 coordinator
        from customer_service_example121 import ResourceCoordinator
        rag_system_instance.resource_coordinator = ResourceCoordinator()
        return {"message": "系统资源已重置"}
    return {"message": "系统未运行"}

# --- 启动入口 ---
if __name__ == "__main__":
    # 使用 uvicorn 启动服务器
    # host="0.0.0.0" 允许局域网内其他电脑访问（例如 Unity 在另一台电脑上）
    # port=8000 是默认端口
    print("启动服务器中... 请访问 http://localhost:8000/docs 查看接口文档")
    uvicorn.run(app, host="0.0.0.0", port=8000)