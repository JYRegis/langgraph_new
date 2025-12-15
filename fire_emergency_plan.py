import json
import networkx as nx
import os
from typing import Dict, Any, Optional
from rag import create_rag_system, RAGSystem

# 检查依赖
LANGGRAPH_AVAILABLE = False
try:
    import langgraph
    from test import load_model_config, build_agent_graph, AgentState
    LANGGRAPH_AVAILABLE = True
except ImportError:
    print("Warning: langgraph not found")

class FireEmergencySystem:
    """火灾应急响应方案生成系统"""
    
    def __init__(self, docs_directory: Optional[str] = None):
        if not LANGGRAPH_AVAILABLE:
            raise ImportError("langgraph 库未安装")
        
        # 加载模型配置
        self.model_config = load_model_config()
        # 构建智能体图
        self.app = build_agent_graph(self.model_config)
        
        # 初始化RAG系统
        self.rag_system: Optional[RAGSystem] = None
        if docs_directory:
            self.initialize_rag(docs_directory)
    
    def initialize_rag(self, docs_directory: str):
        """初始化RAG系统并加载文档"""
        try:
            # [关键修复] 从配置中提取 API Key
            api_key = None
            if self.model_config:
                # 尝试从常用模型配置中获取 Key
                for model in ['qwen', 'deepseek', 'qwen-plus', 'deepseek-r1']:
                    if model in self.model_config and 'api_key' in self.model_config[model]:
                        api_key = self.model_config[model]['api_key']
                        break
            
            if not api_key:
                print("警告: FireEmergencySystem 未找到 API Key，RAG 初始化可能失败")

            # [关键修复] 传递 api_key
            self.rag_system = create_rag_system(
                docs_directory, 
                api_key=api_key, 
                chunk_size=300, 
                chunk_overlap=50
            )
            print(f"FireEmergencySystem RAG 已初始化，文档目录: {docs_directory}")
        except Exception as e:
            print(f"FireEmergencySystem RAG 初始化失败: {e}")
            self.rag_system = None
    
    def add_document_to_rag(self, file_path: str) -> bool:
        """向RAG系统添加单个文档"""
        # 注意：这里如果 rag_system 未初始化，可能需要再次获取 key，
        # 为简化，假设 rag_system 已在 __init__ 中尝试初始化。
        if self.rag_system:
             # 这里 rag.py 的 RAGSystem 没有 add_document 方法，
             # 但有 add_documents_from_directory 或 initialize。
             # 简单起见，我们假设外部主要用目录初始化。
             # 如果必须添加单文件，需要扩展 RAGSystem。
             # 此处为保持兼容，仅打印。
             print("暂不支持单文件动态添加，请使用目录初始化")
             return False
        return False

    def format_response_as_json(self, fire_alarm_info: str) -> Dict[str, Any]:
        """将LLM输出格式化为JSON"""
        try:
            from test import LLMProcessor
            llm_processor = LLMProcessor(self.model_config)
            
            base_prompt = f"""
            你是一名应急专家。根据报警信息：{fire_alarm_info}
            制定三阶段应急方案。
            请严格仅返回以下JSON格式，不要包含Markdown代码块或其他文字：
            {{
              "阶段一": {{ "消防员": [], "医生": [], "保安": [], "物业": [] }},
              "阶段二": {{ "消防员": [], "医生": [], "保安": [], "物业": [] }},
              "阶段三": {{ "消防员": [], "医生": [], "保安": [], "物业": [] }}
            }}
            """
            
            if self.rag_system:
                system_prompt = self.rag_system.enhance_prompt(base_prompt, top_k=5)
            else:
                system_prompt = base_prompt
            
            raw_result = llm_processor._call_llm_api(system_prompt)
            
            # 清理 Markdown 标记
            clean_json = raw_result.replace("```json", "").replace("```", "").strip()
            
            try:
                return json.loads(clean_json)
            except:
                print("JSON解析失败，返回原始文本结构")
                return {"text_response": raw_result}
                
        except Exception as e:
            print(f"生成方案出错: {e}")
            return {"error": str(e)}

    # 为了兼容原有调用，保留 generate_emergency_plan 接口
    def generate_emergency_plan(self, fire_alarm_info: str, thread_id: str = "fire_thread"):
        # 简单返回格式化结果
        return self.format_response_as_json(fire_alarm_info)