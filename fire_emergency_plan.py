import json
import networkx as nx
import os
from typing import Dict, Any, Optional
from rag import create_rag_system, RAGSystem

# 先检查 langgraph 是否已安装
LANGGRAPH_AVAILABLE = False
try:
    # 尝试导入所需库
    import langgraph
    # 导入我们的智能体系统
    from test import (
        load_model_config,
        build_agent_graph,
        AgentState
    )
    LANGGRAPH_AVAILABLE = True
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装所有依赖: pip install langgraph networkx pydantic requests")
    LANGGRAPH_AVAILABLE = False

class FireEmergencySystem:
    """火灾应急响应方案生成系统"""
    
    def __init__(self, docs_directory: Optional[str] = None):
        if not LANGGRAPH_AVAILABLE:
            raise ImportError("langgraph 库未安装或导入失败")
        
        # 加载模型配置
        self.model_config = load_model_config()
        # 构建智能体图
        self.app = build_agent_graph(self.model_config)
        
        # 初始化RAG系统
        self.rag_system: Optional[RAGSystem] = None
        if docs_directory:
            self.initialize_rag(docs_directory)
    
    def initialize_rag(self, docs_directory: str):
        """
        初始化RAG系统并加载文档
        
        Args:
            docs_directory: 包含相关文档的目录路径
        """
        try:
            # 创建并初始化RAG系统，设置合适的文档分块参数
            self.rag_system = create_rag_system(docs_directory, chunk_size=300, chunk_overlap=50)
            print(f"RAG系统已初始化，文档目录: {docs_directory}")
        except Exception as e:
            print(f"RAG系统初始化失败: {e}")
            self.rag_system = None
    
    def add_document_to_rag(self, file_path: str) -> bool:
        """
        向RAG系统添加单个文档
        
        Args:
            file_path: 文档文件路径
        
        Returns:
            是否添加成功
        """
        if not self.rag_system:
            try:
                # 创建RAG系统，设置合适的文档分块参数
                self.rag_system = RAGSystem(chunk_size=300, chunk_overlap=50)
            except Exception as e:
                print(f"创建RAG系统失败: {e}")
                return False
        
        return self.rag_system.add_document(file_path)
    
    def create_fire_emergency_graph(self) -> nx.Graph:
        """创建火灾应急响应相关的图结构"""
        # 创建有向图表示角色和任务关系
        graph = nx.DiGraph()
        
        # 添加主要角色节点
        graph.add_node("消防员", type="应急救援", priority="高")
        graph.add_node("医生", type="医疗救援", priority="中高")
        graph.add_node("保安", type="现场秩序", priority="中")
        graph.add_node("物业", type="信息支持", priority="中")
        graph.add_node("被困人员", type="救援对象", priority="最高")
        graph.add_node("火灾现场", type="环境", priority="中")
        
        # 添加角色之间的关系
        graph.add_edge("消防员", "被困人员", relation="救援", importance=5000)
        graph.add_edge("医生", "被困人员", relation="治疗", importance=4)
        graph.add_edge("保安", "火灾现场", relation="控制", importance=3)
        graph.add_edge("物业", "火灾现场", relation="监控", importance=3)
        graph.add_edge("物业", "消防员", relation="信息支持", importance=4)
        graph.add_edge("物业", "医生", relation="信息支持", importance=3)
        graph.add_edge("物业", "保安", relation="协调", importance=3)
        graph.add_edge("保安", "消防员", relation="协助", importance=3)
        
        return graph
    
    def convert_graph_to_serializable(self, graph: nx.Graph) -> Dict[str, Any]:
        """将networkx的Graph对象转换为可序列化的字典"""
        serializable_graph = {
            "nodes": list(graph.nodes()),
            "edges": []
        }
        
        # 处理边，确保属性可以序列化
        for u, v, attrs in graph.edges(data=True):
            serializable_attrs = {}
            for key, value in attrs.items():
                try:
                    # 测试是否可以序列化
                    json.dumps(value)
                    serializable_attrs[key] = value
                except TypeError:
                    # 如果不能序列化，转换为字符串
                    serializable_attrs[key] = str(value)
            
            serializable_graph["edges"].append((u, v, serializable_attrs))
        
        return serializable_graph
    
    def generate_emergency_plan(self, fire_alarm_info: str, thread_id: str = "fire_emergency_thread") -> Dict[str, Any]:
        """生成火灾应急响应方案"""
        # 创建火灾应急响应图
        fire_graph = self.create_fire_emergency_graph()
        serializable_graph = self.convert_graph_to_serializable(fire_graph)
        
        # 准备输入数据
        inputs = {
            "multimodal_inputs": [
                {"type": "text", "data": fire_alarm_info}
            ],
            "graph_input_data": serializable_graph
        }
        
        # 运行智能体系统
        result = self.app.invoke(inputs, config={"configurable": {"thread_id": thread_id}})
        
        return result
    
    def format_response_as_json(self, fire_alarm_info: str) -> Dict[str, Any]:
        """将LLM输出格式化为要求的JSON结构"""
        try:
            # 导入必要的模块
            import requests
            from test import LLMProcessor
            
            # 创建LLM处理器
            llm_processor = LLMProcessor(self.model_config)
            
            # 构建详细的提示词，要求LLM基于火灾报警信息生成应急方案
            base_prompt = f"""
            你是一名经验丰富的应急响应专家。请根据以下火灾报警信息，为消防员、医生、保安、物业四个角色，
            按报警初期（0-5分钟）、救援中期（5-30分钟）、善后阶段（30分钟后）三个阶段制定应急响应方案。
            
            火灾报警信息：{fire_alarm_info}
            
            请严格按照以下JSON格式输出，只包含字符串数组，不要添加任何解释性文字：
            {{
              "阶段一": {{
                "消防员": ["任务1", "任务2", "任务3"],
                "医生": ["任务1", "任务2", "任务3"],
                "保安": ["任务1", "任务2", "任务3"],
                "物业": ["任务1", "任务2", "任务3"]
              }},
              "阶段二": {{
                "消防员": ["任务1", "任务2", "任务3"],
                "医生": ["任务1", "任务2", "任务3"],
                "保安": ["任务1", "任务2", "任务3"],
                "物业": ["任务1", "任务2", "任务3"]
              }},
              "阶段三": {{
                "消防员": ["任务1", "任务2", "任务3"],
                "医生": ["任务1", "任务2", "任务3"],
                "保安": ["任务1", "任务2", "任务3"],
                "物业": ["任务1", "任务2", "任务3"]
              }}
            }}
            """
            
            # 使用RAG系统增强提示词（如果可用）
            if self.rag_system:
                print("使用RAG系统增强提示词...")
                system_prompt = self.rag_system.enhance_prompt(base_prompt, top_k=10)
            else:
                system_prompt = base_prompt
            
            raw_result = llm_processor._call_llm_api(system_prompt)
            
            # 尝试直接解析LLM返回的JSON结果
            try:
                result = json.loads(raw_result)
                # 添加去重处理
                return self._remove_duplicates(result)
            except json.JSONDecodeError:
                # 如果解析失败，提取其中的JSON部分
                import re
                json_match = re.search(r'\{.*\}', raw_result, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group())
                        # 添加去重处理
                        return self._remove_duplicates(result)
                    except json.JSONDecodeError:
                        print("警告：提取的内容仍然不是有效的JSON格式")
                        # 返回基本结构作为最后备用
                        return self._get_basic_plan_structure()
                else:
                    print("警告：无法从LLM输出中提取有效的JSON格式")
                    # 返回基本结构作为最后备用
                    return self._get_basic_plan_structure()
        except Exception as e:
            print(f"生成方案时出错: {e}")
            # 返回基本结构作为最后备用
            return self._get_basic_plan_structure()
    
    def _remove_duplicates(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """去除方案中的重复任务"""
        cleaned_plan = {}
        for phase, roles in plan.items():
            cleaned_plan[phase] = {}
            for role, tasks in roles.items():
                # 去除重复任务并保持顺序
                seen = set()
                unique_tasks = []
                for task in tasks:
                    if task not in seen:
                        seen.add(task)
                        unique_tasks.append(task)
                cleaned_plan[phase][role] = unique_tasks
        return cleaned_plan
    
    def _get_basic_plan_structure(self) -> Dict[str, Any]:
        """返回基本的应急方案结构"""
        return {
            "阶段一": {
                "消防员": [],
                "医生": [],
                "保安": [],
                "物业": []
            },
            "阶段二": {
                "消防员": [],
                "医生": [],
                "保安": [],
                "物业": []
            },
            "阶段三": {
                "消防员": [],
                "医生": [],
                "保安": [],
                "物业": []
            }
        }

# 主函数
if __name__ == "__main__":
    if LANGGRAPH_AVAILABLE:
        # 火灾报警信息
        fire_alarm_info = """火灾发生地点：阳光花园小区，3号楼，5楼
火势等级：中度（黑烟明显，有明火）
报警来源：烟感报警器 + 居民电话报警
时间：2025年4月15日 08:32
天气：沙尘暴，风力8级
人员情况：楼内可能有10人被困"""
        
        # 文档目录路径（可根据实际情况修改）
        docs_dir = os.path.join(os.path.dirname(__file__), "fire_docs")
        
        # 初始化系统（带RAG功能）
        print("正在初始化火灾应急响应系统...")
        system = FireEmergencySystem(docs_dir if os.path.exists(docs_dir) else None)
        
        # 如果文档目录不存在，提示用户创建
        if not os.path.exists(docs_dir):
            print(f"文档目录不存在: {docs_dir}")
            print("请在该目录下添加火灾应急相关的TXT或JSON文档以启用RAG功能")
            
            # 示例：创建示例文档
            try:
                os.makedirs(docs_dir, exist_ok=True)
                # 创建示例文档
                example_doc_path = os.path.join(docs_dir, "fire_emergency_guide.txt")
                with open(example_doc_path, "w", encoding="utf-8") as f:
                    f.write("""
火灾应急指南

一、报警初期（0-5分钟）处理措施：
1. 立即拨打119报警，清晰说明火灾地点、火势大小和人员情况
2. 疏散人员时应遵循"小火快跑、浓烟关门"原则
3. 沙尘暴天气下，应注意关闭门窗防止火势蔓延
4. 高楼火灾应利用防烟楼梯间撤离，避免使用电梯

二、救援中期（5-30分钟）协调方案：
1. 消防员应优先搜救被困人员，尤其是老人、儿童和行动不便者
2. 医生应在安全区域设立临时医疗点，准备急救设备
3. 保安应设置警戒线，疏散周围群众，确保救援通道畅通
4. 物业应提供建筑平面图和消防设施分布情况

三、善后阶段（30分钟后）工作重点：
1. 持续监测火场，防止复燃
2. 对受伤人员进行分类救治
3. 统计人员伤亡和财产损失情况
4. 配合消防部门进行事故调查
                    """)
                print(f"已创建示例文档: {example_doc_path}")
                # 添加文档到RAG系统
                system.add_document_to_rag(example_doc_path)
            except Exception as e:
                print(f"创建示例文档失败: {e}")
        
        # 生成应急响应方案
        print("正在生成火灾应急响应方案...")
        result = system.generate_emergency_plan(fire_alarm_info)
        
        # 格式化响应为JSON
        print("LLM正在思考并生成方案...")
        formatted_plan = system.format_response_as_json(fire_alarm_info)
        
        # 打印最终结果（严格JSON格式）
        print("\n火灾应急响应方案生成完成:")
        print(json.dumps(formatted_plan, ensure_ascii=False, indent=2))
        
        # 保存结果到文件
        try:
            with open("fire_emergency_result.json", "w", encoding="utf-8") as f:
                json.dump(formatted_plan, f, ensure_ascii=False, indent=2)
            print(f"\n方案已保存到: fire_emergency_result.json")
        except Exception as e:
            print(f"保存方案失败: {e}")
    else:
        print("无法运行，缺少必要的依赖库")