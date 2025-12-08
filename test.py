# 1. 修复 import 顺序，确保 os 被正确导入
import os
import json
import requests
from typing import List, Dict, Any, Union, Optional
from pydantic import BaseModel, Field
import networkx as nx

# 确保正确导入langgraph相关库
LANGGRAPH_AVAILABLE = False
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    print("请安装 langgraph: pip install langgraph")
    # 取消注释以在导入失败时抛出异常
    raise ImportError("需要安装 langgraph: pip install langgraph")

# 配置文件路径 - 修改为使用当前目录下的配置文件
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_api_config.json')

# 加载模型配置
def load_model_config():
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载模型配置失败: {e}")
        # 返回默认配置
        return {
            "qwen-plus": {
                "api_key": "sk-42bcb40c4c5049ee914d21a17fd311d8",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
            },
            "deepseek-r1": {
                "api_key": "sk-42bcb40c4c5049ee914d21a17fd311d8",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
            }
        }

# 定义状态结构
class AgentState(BaseModel):
    # 允许 Pydantic 处理任意类型
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    # 多模态输入数据
    multimodal_inputs: List[Dict[str, Any]] = Field(default_factory=list)
    # 移除不可序列化的 Graph 类型字段
    # 改为可选的字典表示
    graph_input_data: Optional[Dict[str, Any]] = None
    # 感知结果
    perception_results: Dict[str, Any] = Field(default_factory=dict)
    # 关系建模结果
    graph_modeling_result: Dict[str, Any] = Field(default_factory=dict)
    # LLM 处理结果
    llm_output: str = ""
    # 最终规划/推理结果
    planning_result: Dict[str, Any] = Field(default_factory=dict)

# 多模态感知 Agent 类
class MultimodalPerceptionAgent:
    def __init__(self, model_config):
        self.model_config = model_config
        
    def perceive(self, state: AgentState) -> Dict[str, Any]:
        """处理多模态输入，提取特征和信息"""
        print("执行多模态感知...")
        perception_results = {}
        
        # 处理每种模态的输入
        for input_data in state.multimodal_inputs:
            modality_type = input_data.get('type', 'unknown')
            data = input_data.get('data', '')
            
            if modality_type == 'text':
                perception_results['text'] = self._process_text(data)
            elif modality_type == 'image':
                perception_results['image'] = self._process_image(data)
            elif modality_type == 'audio':
                perception_results['audio'] = self._process_audio(data)
            # 可以添加更多模态类型
        
        return {"perception_results": perception_results}
    
    def _process_text(self, text: str) -> Dict[str, Any]:
        """处理文本输入"""
        # 这里可以调用模型API进行文本分析
        return {"content": text, "processed": True}
    
    def _process_image(self, image_path: str) -> Dict[str, Any]:
        """处理图像输入"""
        # 这里可以实现图像分析逻辑
        return {"path": image_path, "processed": True, "analysis": "图像分析结果"}
    
    def _process_audio(self, audio_path: str) -> Dict[str, Any]:
        """处理音频输入"""
        # 这里可以实现音频分析逻辑
        return {"path": audio_path, "processed": True, "transcript": "音频转录结果"}

# 图建模组件
# 在 GraphModelingComponent 类中更新 model 方法
class GraphModelingComponent:
    def model(self, state: AgentState) -> Dict[str, Any]:
        """构建或处理图结构，进行关系建模"""
        print("执行图结构建模...")
        
        # 如果有输入图数据，尝试重建图；否则基于感知结果构建图
        if state.graph_input_data:
            # 从字典数据重建图
            graph = self._rebuild_graph_from_data(state.graph_input_data)
        else:
            graph = self._build_graph_from_perception(state.perception_results)
        
        # 执行图分析
        graph_stats = self._analyze_graph(graph)
        
        # 返回可序列化的数据
        return {"graph_modeling_result": {
            "stats": graph_stats,
            "nodes": list(graph.nodes()),
            "edges": list(graph.edges(data=True))
        }}
    
    def _rebuild_graph_from_data(self, graph_data: Dict[str, Any]) -> nx.Graph:
        """从字典数据重建图结构"""
        graph = nx.Graph()
        
        # 如果数据中包含 nodes 和 edges，重建图
        if "nodes" in graph_data:
            for node in graph_data["nodes"]:
                graph.add_node(node)
        
        if "edges" in graph_data:
            for edge in graph_data["edges"]:
                # 确保边的数据格式正确
                if len(edge) >= 2:
                    if len(edge) == 2:
                        # 无边属性的情况
                        graph.add_edge(edge[0], edge[1])
                    else:
                        # 有边属性的情况
                        graph.add_edge(edge[0], edge[1], **edge[2])
        
        return graph
    
    def _build_graph_from_perception(self, perception_results: Dict[str, Any]) -> nx.Graph:
        """从感知结果构建图结构"""
        graph = nx.Graph()
        
        # 从文本中提取实体和关系
        if 'text' in perception_results:
            text_content = perception_results['text'].get('content', '')
            # 实际应用中，这里应该使用NLP工具提取实体和关系
            # 这里仅作为示例
            entities = self._extract_entities_from_text(text_content)
            for entity in entities:
                graph.add_node(entity)
            
            # 添加实体之间的关系
            if len(entities) >= 2:
                for i in range(len(entities)-1):
                    graph.add_edge(entities[i], entities[i+1], relation="相关")
        
        # 整合其他模态的信息
        if 'image' in perception_results and 'text' in perception_results:
            # 假设图像分析结果与文本内容相关联
            image_analysis = perception_results['image'].get('analysis', '')
            graph.add_node(f"图像内容", type="image")
            # 将图像节点与文本中的第一个实体关联
            if len(list(graph.nodes())) > 0:
                first_entity = list(graph.nodes())[0]
                if first_entity != "图像内容":
                    graph.add_edge(first_entity, "图像内容", relation="包含")
        
        return graph
    
    def _extract_entities_from_text(self, text: str) -> List[str]:
        """从文本中提取实体（简化版）"""
        # 实际应用中，这里应该使用NLP工具如spaCy或jieba
        # 这里仅作为示例，简单分割文本
        if len(text) > 0:
            # 简化的实体提取逻辑
            return ["实体1", "实体2", "实体3"]
        return []
    
    def _analyze_graph(self, graph: nx.Graph) -> Dict[str, Any]:
        """分析图结构，提取特征"""
        # 计算图的基本特征
        return {
            "num_nodes": len(graph.nodes()),
            "num_edges": len(graph.edges()),
            "density": nx.density(graph),
            "connected_components": nx.number_connected_components(graph),
            "average_degree": sum(dict(graph.degree()).values()) / len(graph.nodes()) if graph.nodes() else 0
        }

# LLM 处理组件
class LLMProcessor:
    def __init__(self, model_config, model_choice: str = "qwen"):
        self.model_config = model_config
        self.model_choice = model_choice  # "qwen" 或 "deepseek"
    
    def process(self, state: AgentState) -> Dict[str, Any]:
        """使用LLM处理感知和图建模结果"""
        print(f"使用 {self.model_choice} 模型进行处理...")
        
        # 构建提示词
        prompt = self._build_prompt(state.perception_results, state.graph_modeling_result)
        
        # 调用LLM API
        llm_output = self._call_llm_api(prompt)
        
        return {"llm_output": llm_output}
    
    def _build_prompt(self, perception_results: Dict[str, Any], graph_modeling_result: Dict[str, Any]) -> str:
        """构建发送给LLM的提示词"""
        prompt = f"""根据以下信息进行分析和推理：

1. 多模态感知结果：
{json.dumps(perception_results, ensure_ascii=False, indent=2)}

2. 图结构分析：
节点数量: {graph_modeling_result['stats']['num_nodes']}
边数量: {graph_modeling_result['stats']['num_edges']}
节点列表: {', '.join(graph_modeling_result['nodes'])}

请基于以上信息，进行深度分析和推理，提供详细的见解。"""
        
        return prompt
    
    # 增强 LLM API 调用部分，使其更符合实际使用场景
    def _call_llm_api(self, prompt: str) -> str:
        """调用LLM API获取结果"""
        try:
            config = self.model_config.get(self.model_choice, {})
            api_key = config.get('api_key', '')
            base_url = config.get('base_url', '')
            
            # 验证必要的配置
            if not api_key or not base_url:
                raise ValueError(f"缺少 {self.model_choice} 模型的必要配置")
            
            # 根据模型类型设置正确的请求参数
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            # 根据不同模型设置不同的请求体结构
            if self.model_choice == "qwen":
                data = {
                    "model": "qwen-plus",  # 替换为您实际使用的通义千问模型
                    "messages": [{"role": "user", "content": prompt}]
                }
            elif self.model_choice == "deepseek":
                data = {
                    "model": "deepseek-r1",  # 替换为您实际使用的深度求索模型
                    "messages": [{"role": "user", "content": prompt}]
                }
            else:
                # 默认请求体结构
                data = {
                    "model": "qwen-plus",
                    "messages": [{"role": "user", "content": prompt}]
                }
            
            # 实际的API调用代码
            response = requests.post(base_url, headers=headers, json=data)
            response.raise_for_status()
            return response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
        except Exception as e:
            print(f"调用LLM API失败: {e}")
            return f"[错误] 无法获取LLM结果: {str(e)}"

# 规划/推理组件
class PlanningReasoningComponent:
    def __init__(self, model_config):
        self.model_config = model_config
        # 使用另一个模型进行规划推理
    
    def plan(self, state: AgentState) -> Dict[str, Any]:
        """基于LLM输出进行规划和推理"""
        print("执行规划和推理...")
        
        # 这里可以实现复杂的规划和推理逻辑
        # 简化示例：生成一个基于LLM结果的行动计划
        
        # 构建规划提示词
        planning_prompt = self._build_planning_prompt(state.llm_output, state.graph_modeling_result)
        
        # 调用另一个模型进行规划
        planner = LLMProcessor(self.model_config, model_choice="deepseek" if state.llm_output.startswith("[qwen]") else "qwen")
        plan_result = planner._call_llm_api(planning_prompt)
        
        # 解析规划结果
        parsed_plan = self._parse_plan_result(plan_result)
        
        return {"planning_result": parsed_plan}
    
    def _build_planning_prompt(self, llm_output: str, graph_modeling_result: Dict[str, Any]) -> str:
        """构建规划提示词"""
        prompt = f"""基于以下分析结果，制定详细的行动计划：

{llm_output}

同时考虑图结构中的关系：
{json.dumps(graph_modeling_result['stats'], ensure_ascii=False, indent=2)}

请生成一个结构化的行动计划，包括目标、步骤和预期结果。"""
        return prompt
    
    def _parse_plan_result(self, plan_result: str) -> Dict[str, Any]:
        """解析规划结果"""
        # 简化示例：实际应用中可以根据返回格式进行更复杂的解析
        return {
            "plan_text": plan_result,
            "steps": [
                {"id": 1, "description": "第一步行动"},
                {"id": 2, "description": "第二步行动"},
                {"id": 3, "description": "第三步行动"}
            ],
            "confidence": 0.85  # 模拟的置信度
        }

# 构建LangGraph图
def build_agent_graph(model_config):
    """构建完整的智能体图"""
    if not LANGGRAPH_AVAILABLE:
        raise ImportError("langgraph库未安装或导入失败")
        
    # 初始化各组件
    perception_agent = MultimodalPerceptionAgent(model_config)
    graph_modeler = GraphModelingComponent()
    llm_processor = LLMProcessor(model_config)
    planner = PlanningReasoningComponent(model_config)
    
    # 创建状态图
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("perception", perception_agent.perceive)
    workflow.add_node("graph_modeling", graph_modeler.model)
    workflow.add_node("llm_processing", llm_processor.process)
    workflow.add_node("planning", planner.plan)
    
    # 添加边
    workflow.set_entry_point("perception")
    workflow.add_edge("perception", "graph_modeling")
    workflow.add_edge("graph_modeling", "llm_processing")
    workflow.add_edge("llm_processing", "planning")
    workflow.add_edge("planning", END)
    
    # 添加检查点
    memory = MemorySaver()
    
    # 编译图
    app = workflow.compile(checkpointer=memory)
    
    return app

# 示例运行函数
def run_example():
    """运行示例"""
    # 加载模型配置
    model_config = load_model_config()
    
    # 构建图
    app = build_agent_graph(model_config)
    
    # 准备输入数据（多模态输入示例）
    print("\n=== 示例1：仅使用多模态输入 ===")
    inputs1 = {
        "multimodal_inputs": [
            {"type": "text", "data": "这是一个多模态场景理解的示例文本输入，包含了多个实体和它们之间的关系"},
            {"type": "image", "data": "path/to/image.jpg"}
        ]
    }
    
    # 运行图
    thread_id1 = "example_thread_1"
    result1 = app.invoke(inputs1, config={"configurable": {"thread_id": thread_id1}})
    
    # 打印结果
    print("\n===== 最终结果 =====")
    print(f"感知结果: {result1['perception_results']}")
    print(f"图建模结果: {result1['graph_modeling_result']['stats']}")
    print(f"LLM输出: {result1['llm_output']}")
    print(f"规划结果: {result1['planning_result']['steps']}")
    
    # 示例2：使用预构建的图结构
    print("\n=== 示例2：使用预构建的图结构 ===")
    # 创建一个预构建的图
    prebuilt_graph = nx.Graph()
    prebuilt_graph.add_node("用户", type="person")
    prebuilt_graph.add_node("任务", type="object")
    prebuilt_graph.add_node("资源", type="object")
    prebuilt_graph.add_edge("用户", "任务", relation="分配给")
    prebuilt_graph.add_edge("任务", "资源", relation="需要")
    
    inputs2 = {
        "multimodal_inputs": [
            {"type": "text", "data": "请根据已有关系图，规划完成任务的最佳路径"}
        ],
        "graph_input": prebuilt_graph  # 提供预构建的图结构
    }
    
    # 运行图
    thread_id2 = "example_thread_2"
    result2 = app.invoke(inputs2, config={"configurable": {"thread_id": thread_id2}})
    
    # 打印结果
    print("\n===== 使用预构建图的最终结果 =====")
    print(f"感知结果: {result2['perception_results']}")
    print(f"图建模结果: {result2['graph_modeling_result']['stats']}")
    print(f"LLM输出: {result2['llm_output']}")
    print(f"规划结果: {result2['planning_result']['steps']}")

if __name__ == "__main__":
    run_example()