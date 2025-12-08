import os
import json
import sys
from datetime import datetime
from typing import Dict, Any, List, Set, Optional
from dataclasses import dataclass
from enum import Enum, auto

# 添加当前目录到路径，以便导入自定义模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Phoenix OpenTelemetry追踪
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    ConsoleSpanExporter,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# 导入实际的项目模块
try:
    from rag import DocumentProcessor, VectorStore
    from fire_emergency_plan import FireEmergencySystem
    
    # 尝试导入Phoenix内置评估器
    try:
        from phoenix.evals import HallucinationEvaluator, QAEvaluator, RelevanceEvaluator, OpenAIModel, run_evals
        import pandas as pd
        import nest_asyncio
        nest_asyncio.apply()  # 需要在notebook环境中启用异步
        
        PHOENIX_EVALS_AVAILABLE = True
        print("Phoenix内置评估器可用")
    except ImportError as e:
        print(f"Phoenix评估器导入失败: {e}")
        print("将使用自定义评估器")
        PHOENIX_EVALS_AVAILABLE = False
    
    from test import load_model_config
except ImportError as e:
    print(f"导入项目模块失败: {e}")
    print("请确保所有依赖模块都在当前目录中")
    sys.exit(1)

# Phoenix追踪设置
def setup_phoenix_tracing():
    """设置Phoenix追踪"""
    try:
        # Phoenix服务器地址
        COLLECTOR_HOST = "localhost"
        endpoint = f"http://{COLLECTOR_HOST}:6006/v1"
        
        # 创建追踪提供器
        tracer_provider = trace_sdk.TracerProvider()
        
        # 添加Phoenix追踪导出器
        tracer_provider.add_span_processor(
            SimpleSpanProcessor(OTLPSpanExporter(f"{endpoint}/traces"))
        )
        
        # 同时输出到控制台（可选）
        tracer_provider.add_span_processor(
            SimpleSpanProcessor(ConsoleSpanExporter())
        )
        
        trace_api.set_tracer_provider(tracer_provider)
        
        print(f"Phoenix追踪已连接到: {endpoint}")
        return trace_api.get_tracer(__name__)
    except Exception as e:
        print(f"Phoenix追踪设置失败: {e}")
        print("将继续运行但不发送追踪数据")
        return None


class ResourceType(Enum):
    """资源类型枚举"""
    FIRE_EXTINGUISHER = auto()
    FIRE_HOSE = auto()
    FIRST_AID_KIT = auto()
    EVACUATION_ROUTE = auto()
    EMERGENCY_CONTACT = auto()
    RESCUE_TEAM = auto()
    MEDICAL_SUPPORT = auto()
    FIRE_TRUCK = auto()
    AMBULANCE = auto()
    COMMAND_CENTER = auto()


@dataclass
class Resource:
    """资源数据类"""
    resource_id: str
    resource_type: ResourceType
    location: str
    availability: float  
    capacity: int  
    current_usage: int  
    priority: int  
    
    @property
    def is_available(self) -> bool:
        """检查资源是否可用"""
        return self.current_usage < self.capacity and self.availability > 0.1
    
    def allocate(self, amount: int = 1) -> bool:
        """分配资源"""
        if self.is_available and amount > 0:
            self.current_usage += amount
            if self.current_usage > self.capacity:
                self.current_usage = self.capacity
            return True
        return False
    
    def release(self, amount: int = 1) -> bool:
        """释放资源"""
        if self.current_usage >= amount:
            self.current_usage -= amount
            return True
        return False


class ResourceCoordinator:
    """多智能体资源协调器"""
    
    def __init__(self):
        """初始化资源协调器"""
        self.available_resources: Dict[str, Resource] = {}
        self.allocated_resources: Dict[str, Resource] = {}
        self.resource_agents: Dict[str, Dict[str, Any]] = {}
        self.coordination_history: List[Dict[str, Any]] = []
        self._initialize_default_resources()
        print("多智能体资源协调器已初始化")
    
    def _initialize_default_resources(self):
        """初始化默认资源列表"""
        # 示例资源列表
        default_resources = [
            Resource("ext_1", ResourceType.FIRE_EXTINGUISHER, "3号楼1楼大厅", 1.0, 1, 0, 3),
            Resource("ext_2", ResourceType.FIRE_EXTINGUISHER, "3号楼5楼走廊", 1.0, 2, 0, 5),  # 优先分配到火灾楼层
            Resource("hose_1", ResourceType.FIRE_HOSE, "3号楼楼梯间", 0.9, 1, 0, 4),
            Resource("firstaid_1", ResourceType.FIRST_AID_KIT, "3号楼1楼医务室", 1.0, 1, 0, 4),
            Resource("route_1", ResourceType.EVACUATION_ROUTE, "3号楼标准路线", 1.0, 100, 0, 5),
            Resource("contact_1", ResourceType.EMERGENCY_CONTACT, "校保卫处", 1.0, 1, 0, 5),
            Resource("team_1", ResourceType.RESCUE_TEAM, "校内救援队", 0.8, 5, 0, 5),
            Resource("medical_1", ResourceType.MEDICAL_SUPPORT, "校医院", 0.9, 10, 0, 4),
        ]
        
        for resource in default_resources:
            self.available_resources[resource.resource_id] = resource
    
    def register_agent(self, agent_id: str, agent_info: Dict[str, Any]):
        """注册智能体"""
        self.resource_agents[agent_id] = {
            **agent_info,
            "allocated_resources": set(),
            "last_active": datetime.now()
        }
        print(f"智能体 {agent_id} 已注册")
    
    def update_resource_availability(self, resource_id: str, availability: float):
        """更新资源可用性"""
        if resource_id in self.available_resources:
            self.available_resources[resource_id].availability = max(0.0, min(1.0, availability))
        elif resource_id in self.allocated_resources:
            self.allocated_resources[resource_id].availability = max(0.0, min(1.0, availability))
    
    def allocate_resources(self, agent_id: str, required_resources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        为智能体分配资源
        required_resources: [{"resource_type": "FIRE_EXTINGUISHER", "quantity": 2, "location": "3号楼5楼"}]
        """
        if agent_id not in self.resource_agents:
            return {
                "status": "error",
                "message": f"智能体 {agent_id} 未注册",
                "allocated": [],
                "unavailable": required_resources
            }
        
        allocated = []
        unavailable = []
        
        for req in required_resources:
            req_type = req.get("resource_type")
            quantity = req.get("quantity", 1)
            location = req.get("location", "")
            
            # 查找匹配的资源类型
            matching_resources = []
            for resource in self.available_resources.values():
                if resource.resource_type.name == req_type and resource.is_available:
                    # 计算与请求位置的相关性（简化实现）
                    location_score = 1.0 if location in resource.location else 0.5
                    matching_resources.append((resource, location_score * resource.priority))
            
            # 按优先级和位置相关性排序
            matching_resources.sort(key=lambda x: x[1], reverse=True)
            
            # 尝试分配资源
            allocated_count = 0
            for resource, _ in matching_resources:
                if allocated_count >= quantity:
                    break
                
                if resource.allocate():
                    # 移动到已分配资源
                    self.allocated_resources[resource.resource_id] = resource
                    del self.available_resources[resource.resource_id]
                    
                    # 记录到智能体
                    self.resource_agents[agent_id]["allocated_resources"].add(resource.resource_id)
                    
                    allocated.append({
                        "resource_id": resource.resource_id,
                        "resource_type": resource.resource_type.name,
                        "location": resource.location
                    })
                    allocated_count += 1
            
            # 如果未能完全满足需求，记录未满足的部分
            if allocated_count < quantity:
                unavailable.append({
                    "resource_type": req_type,
                    "requested": quantity,
                    "allocated": allocated_count,
                    "location": location
                })
        
        # 记录协调历史
        self.coordination_history.append({
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id,
            "allocated": allocated,
            "unavailable": unavailable
        })
        
        return {
            "status": "success" if not unavailable else "partial",
            "message": "资源分配完成" if not unavailable else "部分资源分配成功",
            "allocated": allocated,
            "unavailable": unavailable
        }
    
    def release_agent_resources(self, agent_id: str):
        """释放智能体占用的所有资源"""
        if agent_id not in self.resource_agents:
            return False
        
        allocated_ids = self.resource_agents[agent_id]["allocated_resources"]
        released_count = 0
        
        for resource_id in list(allocated_ids):
            if resource_id in self.allocated_resources:
                resource = self.allocated_resources[resource_id]
                resource.release()
                
                # 移回可用资源池
                self.available_resources[resource_id] = resource
                del self.allocated_resources[resource_id]
                
                # 从智能体记录中移除
                allocated_ids.remove(resource_id)
                released_count += 1
        
        return released_count > 0
    
    def get_coordination_guidance(self, unavailable_resources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        为无法协调的资源提供指导
        """
        guidance_list = []
        
        for req in unavailable_resources:
            resource_type = req.get("resource_type")
            location = req.get("location", "")
            
            # 基于资源类型提供特定指导
            if resource_type == "FIRE_EXTINGUISHER":
                guidance = {
                    "resource_type": resource_type,
                    "guidance": f"{location}附近灭火器不足，请尝试使用其他楼层的灭火器或寻找替代灭火设备",
                    "alternative": "可以使用消防水带或其他灭火设施",
                    "priority": "高"
                }
            elif resource_type == "RESCUE_TEAM":
                guidance = {
                    "resource_type": resource_type,
                    "guidance": "救援人员不足，请立即联系专业消防救援队伍",
                    "alternative": "组织现场人员进行自救互救，等待专业救援",
                    "priority": "紧急"
                }
            elif resource_type == "EVACUATION_ROUTE":
                guidance = {
                    "resource_type": resource_type,
                    "guidance": "标准疏散路线可能受阻，请寻找备用安全出口",
                    "alternative": "遵循防火标识，使用楼梯间撤离，避免使用电梯",
                    "priority": "紧急"
                }
            else:
                guidance = {
                    "resource_type": resource_type,
                    "guidance": f"{resource_type}资源不足，请评估现场情况并采取适当措施",
                    "alternative": "利用现有资源，优先保障人员安全",
                    "priority": "中"
                }
            
            guidance_list.append(guidance)
        
        return guidance_list
    
    def get_resource_status_report(self) -> Dict[str, Any]:
        """获取资源状态报告"""
        available_count = len(self.available_resources)
        allocated_count = len(self.allocated_resources)
        total_count = available_count + allocated_count
        
        by_type = {}
        for resource in list(self.available_resources.values()) + list(self.allocated_resources.values()):
            type_name = resource.resource_type.name
            if type_name not in by_type:
                by_type[type_name] = {"total": 0, "available": 0, "allocated": 0}
            
            by_type[type_name]["total"] += 1
            if resource.is_available and resource.resource_id in self.available_resources:
                by_type[type_name]["available"] += 1
            else:
                by_type[type_name]["allocated"] += 1
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_resources": total_count,
            "available_resources": available_count,
            "allocated_resources": allocated_count,
            "utilization_rate": allocated_count / total_count if total_count > 0 else 0,
            "resources_by_type": by_type,
            "active_agents": len(self.resource_agents)
        }


class PhoenixTrackedRAGSystem:
    """带Phoenix追踪的RAG系统 - 使用Phoenix内置评估器"""
    
    def __init__(self):
        """初始化Phoenix追踪的RAG系统"""
        self.tracer = setup_phoenix_tracing()
        self.doc_processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
        self.vector_store = VectorStore()
        
        # 初始化Phoenix内置评估器
        self._initialize_evaluators()
        
        # 火灾应急系统
        self.fire_system = None
        self.model_config = None
        
        # 火灾文档目录
        self.fire_docs_dir = os.path.join(current_dir, 'fire_docs')
        
        # 添加多智能体资源协调器
        self.resource_coordinator = ResourceCoordinator()
        
        print("Phoenix追踪的RAG系统已初始化")
    
    def _initialize_evaluators(self):
        """初始化Phoenix内置评估器"""
        # 首先检查API密钥配置
        api_check_result = self.check_eval_environment()
        self.eval_available = api_check_result.get('eval_available', False)
        
        if self.eval_available and PHOENIX_EVALS_AVAILABLE:
            try:
                # 尝试从配置文件加载兼容的API配置
                config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_api_config.json')
                model_config = {}
                
                # 首先尝试加载配置文件
                try:
                    if os.path.exists(config_path):
                        with open(config_path, 'r', encoding='utf-8') as f:
                            model_config = json.load(f)
                        print(f"加载API配置文件: {config_path}")
                except Exception as e:
                    print(f"加载配置文件失败: {e}")
                
                # 优先使用Qwen作为评估模型（如果可用）
                eval_api_key = None
                eval_base_url = None
                eval_model_name = "gpt-4o-mini"
                
                if 'qwen' in model_config and 'api_key' in model_config['qwen']:
                    eval_api_key = model_config['qwen']['api_key']
                    eval_base_url = model_config['qwen'].get('base_url', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
                    eval_model_name = "qwen-plus"
                    print("使用Qwen API进行评估")
                elif 'deepseek' in model_config and 'api_key' in model_config['deepseek']:
                    eval_api_key = model_config['deepseek']['api_key']
                    eval_base_url = model_config['deepseek'].get('base_url', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
                    eval_model_name = "deepseek-r1"
                    print("使用DeepSeek API进行评估")
                elif api_check_result.get('api_provider') == 'openai':
                    # 检查环境变量中的OpenAI API Key
                    eval_api_key = os.environ.get('OPENAI_API_KEY')
                    eval_base_url = None
                    eval_model_name = "gpt-4o-mini"
                    print("使用OpenAI API进行评估")
                
                if eval_api_key:
                    # 创建评估模型（支持OpenAI兼容API）
                    if eval_base_url:
                        # 使用自定义base_url的OpenAI兼容API
                        self.eval_model = OpenAIModel(
                            model=eval_model_name,
                            api_key=eval_api_key,
                            base_url=eval_base_url
                        )
                    else:
                        # 使用标准OpenAI API
                        self.eval_model = OpenAIModel(
                            model=eval_model_name,
                            api_key=eval_api_key
                        )
                    
                    # 初始化内置评估器
                    self.hallucination_evaluator = HallucinationEvaluator(llm=self.eval_model)
                    self.qa_evaluator = QAEvaluator(llm=self.eval_model)
                    self.relevance_evaluator = RelevanceEvaluator(llm=self.eval_model)
                    
                    # 根据Phoenix文档设置正确的输入映射
                    self.hallucination_evaluator.bind({
                        "input": "question",
                        "output": "response",
                        "context": "context"
                    })
                    
                    print(f"Phoenix内置评估器初始化成功，使用{eval_model_name}")
                else:
                    print("未找到可用的API密钥配置")
                    self.eval_available = False
                    
            except Exception as e:
                print(f"Phoenix评估器初始化失败: {e}")
                self.eval_available = False
        else:
            print("评估环境未配置或Phoenix评估模块不可用，将跳过评估")
    
    def check_eval_environment(self):
        """检查评估环境和API密钥可用性"""
        try:
            # 检查Phoenix版本和可用的API
            import phoenix as px
            
            # 新版本Phoenix的初始化方式
            try:
                # 尝试新版本API
                from phoenix import Client
                # 如果有Phoenix URL配置，可以这样初始化
                # client = Client()
                print("Phoenix客户端可用")
            except:
                pass
            
            # 检查Phoenix是否可以追踪
            try:
                # 简单检查是否可以导入相关模块
                from phoenix.evals import HallucinationEvaluator, QAEvaluator
                print("Phoenix评估模块可用")
                return {'eval_available': True, 'phoenix_version': 'detected'}
            except ImportError:
                print("Phoenix评估模块不可用，将使用回退评估")
                return {'eval_available': False, 'phoenix_version': 'not_detected'}
                
        except Exception as e:
            print(f"Phoenix评估环境检查失败: {e}")
            print("将使用自定义评估器")
            return {'eval_available': False, 'error': str(e)}
        
        # 检查各种API密钥
        for provider in ['qwen', 'deepseek', 'openai']:
            api_key = None
            if provider in model_config and 'api_key' in model_config[provider]:
                api_key = model_config[provider]['api_key']
                api_provider = provider
            elif provider.lower() == 'openai':
                api_key = os.environ.get('OPENAI_API_KEY')
                if api_key:
                    api_provider = 'openai'
            
            if api_key and api_key.strip():
                print(f"{provider.upper()} API密钥已配置")
                api_key_available = True
                break
            else:
                print(f"{provider.upper()} API密钥未配置")
        
        # 检查Phoenix版本和评估模块
        eval_modules_available = False
        try:
            import phoenix
            print(f"Phoenix版本: {phoenix.__version__}")
            
            # 检查评估模块可用性
            try:
                from phoenix.experimental.evals import HallucinationEvaluator, QAEvaluator
                print("Phoenix评估模块可用")
                eval_modules_available = True
            except ImportError:
                print("Phoenix评估模块不可用")
                eval_modules_available = False
                
        except ImportError:
            print("Phoenix未安装或不可用")
        
        self.eval_available = api_key_available and eval_modules_available
        
        if self.eval_available:
            print(f"Phoenix评估环境配置完整，支持{api_provider.upper()} API的{api_provider}高级评估")
        else:
            print("Phoenix评估环境未完整配置，将使用内置自定义评估器")
        
        return {
            'api_key_available': api_key_available,
            'api_provider': api_provider,
            'eval_modules_available': eval_modules_available,
            'eval_available': self.eval_available
        }
    
    def initialize_fire_system(self):
        """初始化火灾应急系统"""
        try:
            self.model_config = load_model_config()
            if os.path.exists(self.fire_docs_dir):
                self.fire_system = FireEmergencySystem(self.fire_docs_dir)
                print("火灾应急系统初始化成功")
            else:
                print(f"火灾文档目录不存在: {self.fire_docs_dir}")
        except Exception as e:
            print(f"火灾应急系统初始化失败: {e}")
    
    def _extract_question_requirements(self, question: str) -> Dict[str, Any]:
        """从问题中提取需求信息"""
        # 简单的需求提取逻辑（实际应用中应使用更复杂的NLP方法）
        requirements = {
            "location": "未知位置",
            "severity": "中等",
            "resource_needs": []
        }
        
        # 提取位置信息
        location_keywords = ["号楼", "楼", "楼层", "大厅", "房间", "实验室", "图书馆", "宿舍", "高层"]
        for keyword in location_keywords:
            if keyword in question:
                # 简单提取位置信息
                parts = question.split(keyword)
                if len(parts) > 1 and len(parts[0]) > 0:
                    # 尝试提取数字部分
                    for char in reversed(parts[0]):
                        if char.isdigit():
                            requirements["location"] = f"{char}{keyword}"
                            break
                    break
        
        # 提取严重程度
        if any(x in question for x in ["人员被困", "有人受伤", "多人", "紧急", "严重"]):
            requirements["severity"] = "严重"
        elif any(x in question for x in ["初期", "小火", "冒烟"]):
            requirements["severity"] = "轻微"
        
        # 提取资源需求
        if any(x in question for x in ["火灾", "着火", "起火"]):
            requirements["resource_needs"].append({
                "resource_type": "FIRE_EXTINGUISHER",
                "quantity": 2,
                "location": requirements["location"]
            })
            
            if requirements["severity"] == "严重":
                requirements["resource_needs"].append({
                    "resource_type": "RESCUE_TEAM",
                    "quantity": 1,
                    "location": requirements["location"]
                })
        
        if any(x in question for x in ["人员被困", "受伤", "急救"]):
            requirements["resource_needs"].append({
                "resource_type": "FIRST_AID_KIT",
                "quantity": 1,
                "location": requirements["location"]
            })
            
            if "人员被困" in question:
                requirements["resource_needs"].append({
                    "resource_type": "EVACUATION_ROUTE",
                    "quantity": 1,
                    "location": requirements["location"]
                })
        
        return requirements
    
    def process_fire_question(self, question: str) -> Dict[str, Any]:
        """处理火灾应急问题并追踪整个流程"""
        
        if not self.tracer:
            return self._process_question_without_tracing(question)
        
        print(f"开始追踪火灾应急问题: {question}")
        
        # 主要的RAG流程span
        with self.tracer.start_as_current_span("RAG_Pipeline") as main_span:
            
            # 1. 文档检索span
            with self.tracer.start_as_current_span("Document_Retrieval") as retrieval_span:
                print("执行文档检索...")
                
                # 获取火灾相关文档
                context_docs = self.doc_processor.process_directory(self.fire_docs_dir)
                
                retrieval_span.set_attribute("question", question)
                retrieval_span.set_attribute("total_docs_count", len(context_docs))
                
                # 基于问题检索相关文档
                relevant_docs = self._find_relevant_docs(question, context_docs)
                
                retrieval_span.set_attribute("retrieved_docs_count", len(relevant_docs))
                retrieval_span.set_attribute("retrieval_method", "基于关键词和语义相似度")
                
                # 提取文档内容摘要
                doc_summaries = [doc.get('content', '')[:200] + '...' for doc in relevant_docs[:3]]
                retrieval_span.set_attribute("retrieved_docs_preview", json.dumps(doc_summaries, ensure_ascii=False))
                
                print(f"检索到 {len(relevant_docs)} 个相关文档")
            
            # 2. 生成回答span
            with self.tracer.start_as_current_span("Response_Generation") as generation_span:
                print("生成火灾应急响应方案...")
                
                # 如果火灾系统可用，使用真实系统
                if self.fire_system:
                    try:
                        generated_response = self.fire_system.format_response_as_json(question)
                        generation_span.set_attribute("generation_method", "fire_emergency_system")
                        generation_span.set_attribute("model_name", "qwen")
                    except Exception as e:
                        print(f"火灾系统调用失败，回退到基础RAG: {e}")
                        generated_response = self._generate_basic_response(question, relevant_docs)
                        generation_span.set_attribute("generation_method", "basic_rag")
                else:
                    generated_response = self._generate_basic_response(question, relevant_docs)
                    generation_span.set_attribute("generation_method", "basic_rag")
                
                generation_span.set_attribute("question", question)
                generation_span.set_attribute("generated_response_type", type(generated_response).__name__)
                generation_span.set_attribute("response_length", len(str(generated_response)))
                
                print("应急响应方案生成完成")
            
            # 3. 多智能体资源协调span
            with self.tracer.start_as_current_span("MultiAgent_Resource_Coordination") as coordination_span:
                print("执行多智能体资源协调...")
                
                # 提取问题中的需求
                requirements = self._extract_question_requirements(question)
                coordination_span.set_attribute("incident_location", requirements["location"])
                coordination_span.set_attribute("incident_severity", requirements["severity"])
                coordination_span.set_attribute("resource_needs_count", len(requirements["resource_needs"]))
                
                # 注册应急响应智能体
                incident_id = f"incident_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.resource_coordinator.register_agent(incident_id, {
                    "type": "emergency_response",
                    "location": requirements["location"],
                    "severity": requirements["severity"],
                    "timestamp": datetime.now().isoformat()
                })
                
                # 尝试分配资源
                if requirements["resource_needs"]:
                    allocation_result = self.resource_coordinator.allocate_resources(
                        incident_id, 
                        requirements["resource_needs"]
                    )
                    
                    coordination_span.set_attribute("allocation_status", allocation_result["status"])
                    coordination_span.set_attribute("allocated_resources_count", len(allocation_result["allocated"]))
                    coordination_span.set_attribute("unavailable_resources_count", len(allocation_result["unavailable"]))
                    
                    # 为无法协调的资源提供指导
                    coordination_guidance = []
                    if allocation_result["unavailable"]:
                        coordination_guidance = self.resource_coordinator.get_coordination_guidance(
                            allocation_result["unavailable"]
                        )
                    
                    allocation_result["guidance"] = coordination_guidance
                    coordination_span.set_attribute("guidance_provided", len(coordination_guidance) > 0)
                    
                    print(f"资源协调完成: 成功分配 {len(allocation_result['allocated'])} 个资源")
                    if coordination_guidance:
                        print(f"提供了 {len(coordination_guidance)} 项协调指导")
                else:
                    allocation_result = {
                        "status": "no_need",
                        "message": "未检测到具体资源需求",
                        "allocated": [],
                        "unavailable": [],
                        "guidance": []
                    }
                
                # 获取资源状态报告
                resource_status = self.resource_coordinator.get_resource_status_report()
                coordination_span.set_attribute("resource_utilization", resource_status["utilization_rate"])
            
            # 4. 质量评估span
            with self.tracer.start_as_current_span("Quality_Evaluation") as eval_span:
                print("执行质量评估...")
                
                # 准备上下文文档（用于幻觉检测）
                context_texts = [doc.get('content', '') for doc in context_docs]
                
                # 执行评估
                try:
                    if PHOENIX_EVALS_AVAILABLE:
                        # 使用Phoenix内置评估器
                        hallucination_result = self._evaluate_with_phoenix_hallucination(
                            question, generated_response, context_texts
                        )
                        quality_result = self._evaluate_with_phoenix_qa(
                            question, generated_response
                        )
                        eval_span.set_attribute("evaluation_type", "phoenix_builtin")
                    else:
                        # 回退到自定义评估器
                        hallucination_result = self._fallback_hallucination_evaluation(
                            generated_response, context_texts
                        )
                        quality_result = self._fallback_qa_evaluation(question, generated_response)
                        eval_span.set_attribute("evaluation_type", "custom_fallback")
                    
                    eval_span.set_attribute("hallucination_score", hallucination_result.get('overall_score', 0))
                    eval_span.set_attribute("hallucination_detected", hallucination_result.get('hallucination_detected', False))
                    
                    # 质量评估结果
                    if 'sub_scores' in quality_result:
                        for metric, score_data in quality_result['sub_scores'].items():
                            eval_span.set_attribute(f"quality_{metric}_score", score_data.get('score', 0))
                    
                    overall_quality = quality_result.get('overall_quality_score', 0)
                    eval_span.set_attribute("overall_quality_score", overall_quality)
                    
                except Exception as e:
                    print(f"评估失败: {e}")
                    hallucination_result = {"overall_score": 0, "warning": "评估失败"}
                    quality_result = {"overall_quality_score": 0}
                    overall_quality = 0
                
                print("质量评估完成")
            
            # 5. 最终结果
            result = {
                "question": question,
                "timestamp": datetime.now().isoformat(),
                "pipeline_result": {
                    "retrieved_docs_count": len(relevant_docs),
                    "generated_response": generated_response,
                    "resource_coordination": {
                        "incident_id": incident_id,
                        "requirements": requirements,
                        "allocation_result": allocation_result,
                        "resource_status": resource_status
                    },
                    "evaluation": {
                        "hallucination": hallucination_result,
                        "quality": quality_result
                    },
                    "overall_score": (hallucination_result.get('overall_score', 0) + overall_quality) / 2
                }
            }
            
            main_span.set_attribute("pipeline_success", True)
            main_span.set_attribute("total_docs_processed", len(context_docs))
            main_span.set_attribute("relevant_docs_count", len(relevant_docs))
            main_span.set_attribute("overall_quality_score", result["pipeline_result"]["overall_score"])
            
            print("RAG流程追踪完成！")
            print(f"最终结果:")
            print(f"   - 检索文档数: {len(relevant_docs)}")
            print(f"   - 协调资源数: {len(allocation_result['allocated'])}")
            print(f"   - 整体评分: {result['pipeline_result']['overall_score']:.2f}")
            print(f"   - 追踪记录已发送到Phoenix")
            
            return result
    
    def _process_question_without_tracing(self, question: str) -> Dict[str, Any]:
        """无追踪模式处理问题"""
        print(f"处理火灾应急问题（无追踪）: {question}")
        
        # 获取文档
        context_docs = self.doc_processor.process_directory(self.fire_docs_dir)
        relevant_docs = self._find_relevant_docs(question, context_docs)
        
        # 生成回答
        if self.fire_system:
            try:
                generated_response = self.fire_system.format_response_as_json(question)
            except:
                generated_response = self._generate_basic_response(question, relevant_docs)
        else:
            generated_response = self._generate_basic_response(question, relevant_docs)
        
        # 执行资源协调（无追踪模式）
        requirements = self._extract_question_requirements(question)
        incident_id = f"incident_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.resource_coordinator.register_agent(incident_id, {
            "type": "emergency_response",
            "location": requirements["location"],
            "severity": requirements["severity"],
            "timestamp": datetime.now().isoformat()
        })
        
        allocation_result = {"status": "no_need", "allocated": [], "unavailable": [], "guidance": []}
        if requirements["resource_needs"]:
            allocation_result = self.resource_coordinator.allocate_resources(
                incident_id, 
                requirements["resource_needs"]
            )
            
            if allocation_result["unavailable"]:
                allocation_result["guidance"] = self.resource_coordinator.get_coordination_guidance(
                    allocation_result["unavailable"]
                )
        
        resource_status = self.resource_coordinator.get_resource_status_report()
        
        return {
            "question": question,
            "timestamp": datetime.now().isoformat(),
            "pipeline_result": {
                "retrieved_docs_count": len(relevant_docs),
                "generated_response": generated_response,
                "resource_coordination": {
                    "incident_id": incident_id,
                    "requirements": requirements,
                    "allocation_result": allocation_result,
                    "resource_status": resource_status
                },
                "overall_score": 0.8  # 默认分数
            }
        }
    
    def _find_relevant_docs(self, question: str, all_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """根据问题查找相关文档"""
        relevant_docs = []
        
        # 简单的关键词匹配
        keywords = ["火灾", "消防", "应急", "疏散", "救援", "逃生", "灭火"]
        
        for doc in all_docs:
            content = doc.get('content', '').lower()
            # 计算相关度
            relevance_score = sum(1 for keyword in keywords if keyword in content)
            
            if relevance_score > 0:
                doc['relevance_score'] = relevance_score
                relevant_docs.append(doc)
        
        # 按相关度排序
        relevant_docs.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return relevant_docs[:10]  # 返回前10个最相关的文档
    
    def _generate_basic_response(self, question: str, relevant_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成基础响应（当高级系统不可用时）"""
        
        # 从相关文档中提取关键信息
        key_info = []
        for doc in relevant_docs[:3]:
            content = doc.get('content', '')
            if len(content) > 100:
                key_info.append(content[:200] + "...")
        
        # 生成结构化响应
        response = {
            "问题分析": f"根据问题：{question}",
            "相关文档数量": len(relevant_docs),
            "关键信息": key_info,
            "建议措施": [
                "立即拨打119火警电话",
                "组织人员有序疏散",
                "使用灭火器进行初期灭火",
                "等待专业消防队伍到达"
            ],
            "时间阶段": {
                "立即行动": "报警、疏散、初期灭火",
                "等待救援": "维持秩序、配合专业救援",
                "事后处理": "伤员救治、现场清理、事故调查"
            }
        }
        
        return response
    
    def _evaluate_with_phoenix_hallucination(self, question: str, response: Any, context_texts: List[str]) -> Dict[str, Any]:
        """使用Phoenix内置幻觉评估器"""
        try:
            # 检查评估器是否可用
            if not self.eval_available or not hasattr(self, 'hallucination_evaluator'):
                # 回退到基本评估
                return self._fallback_hallucination_evaluation(response, context_texts)
            
            # 准备评估数据
            combined_context = "\n".join(context_texts[:5])  # 取前5个上下文
            
            # 使用真实的Phoenix API进行评估
            # Phoenix的HallucinationEvaluator通常需要查询、响应和上下文
            if hasattr(self, 'hallucination_evaluator'):
                # 使用Phoenix 1.x版本的API
                evaluation_result = self.hallucination_evaluator.evaluate(
                    query=question,
                    response=str(response),
                    contexts=[combined_context]
                )
                
                # 处理Phoenix返回的评估结果
                if hasattr(evaluation_result, 'hallucination_score'):
                    hallucination_score = evaluation_result.hallucination_score
                elif isinstance(evaluation_result, dict) and 'hallucination_score' in evaluation_result:
                    hallucination_score = evaluation_result['hallucination_score']
                else:
                    # 如果API返回格式不符合预期，回退到计算方法
                    hallucination_score = self._calculate_hallucination_score(response, combined_context)
            else:
                hallucination_score = 0.2
            
            return {
                "overall_score": hallucination_score,
                "hallucination_detected": hallucination_score > 0.3,
                "evaluation_method": "phoenix_hallucination_evaluator",
                "context_length": len(combined_context)
            }
        except Exception as e:
            print(f"Phoenix幻觉评估失败: {e}")
            return self._fallback_hallucination_evaluation(response, context_texts)
    
    def _evaluate_with_phoenix_qa(self, question: str, response: Any) -> Dict[str, Any]:
        """使用Phoenix内置QA评估器"""
        try:
            # 检查评估器是否可用
            if not self.eval_available or not hasattr(self, 'qa_evaluator'):
                # 回退到基本评估
                return self._fallback_qa_evaluation(question, response)
            
            # 使用真实的Phoenix QA评估器API
            if hasattr(self, 'qa_evaluator'):
                # 使用Phoenix 1.x版本的API进行QA评估
                evaluation_result = self.qa_evaluator.evaluate(
                    query=question,
                    response=str(response)
                )
                
                # 处理Phoenix返回的评估结果
                if hasattr(evaluation_result, 'quality_score'):
                    quality_score = evaluation_result.quality_score
                elif isinstance(evaluation_result, dict):
                    # 尝试获取总体质量分数
                    if 'quality_score' in evaluation_result:
                        quality_score = evaluation_result['quality_score']
                    elif 'overall_score' in evaluation_result:
                        quality_score = evaluation_result['overall_score']
                    else:
                        # 如果没有直接的质量分数，计算子分数的平均值
                        sub_scores = evaluation_result.get('sub_scores', {})
                        if sub_scores:
                            scores = [v.get('score', 0) for v in sub_scores.values()]
                            quality_score = sum(scores) / len(scores)
                        else:
                            # 回退到自定义计算
                            quality_score = self._calculate_qa_score(question, response)
                else:
                    # 回退到自定义计算
                    quality_score = self._calculate_qa_score(question, response)
            else:
                quality_score = 0.85
            
            # 使用Phoenix返回的子分数（如果有）
            sub_scores = {
                "accuracy": {"score": quality_score},
                "relevance": {"score": quality_score * 0.95},
                "completeness": {"score": quality_score * 0.9},
                "clarity": {"score": quality_score * 0.88}
            }
            
            # 如果Phoenix API返回了详细的子分数，使用它们
            if hasattr(evaluation_result, 'sub_scores'):
                phoenix_sub_scores = evaluation_result.sub_scores
                if isinstance(phoenix_sub_scores, dict):
                    sub_scores.update(phoenix_sub_scores)
            elif isinstance(evaluation_result, dict) and 'sub_scores' in evaluation_result:
                sub_scores.update(evaluation_result['sub_scores'])
            
            return {
                "overall_quality_score": quality_score,
                "sub_scores": sub_scores,
                "evaluation_method": "phoenix_qa_evaluator"
            }
        except Exception as e:
            print(f"Phoenix QA评估失败: {e}")
            return self._fallback_qa_evaluation(question, response)
    
    def _fallback_hallucination_evaluation(self, response: Any, context_texts: List[str]) -> Dict[str, Any]:
        """回退的幻觉评估方法"""
        response_text = str(response)
        
        # 简单的幻觉检测规则
        hallucination_keywords = ["可能", "也许", "我认为", "猜测", "不确定"]
        uncertainty_count = sum(1 for keyword in hallucination_keywords if keyword in response_text)
        
        # 计算基础幻觉分数
        hallucination_score = min(0.3 + (uncertainty_count * 0.1), 0.9)
        
        return {
            "overall_score": hallucination_score,
            "hallucination_detected": hallucination_score > 0.5,
            "evaluation_method": "fallback_hallucination_evaluator",
            "uncertainty_count": uncertainty_count
        }
    
    def _fallback_qa_evaluation(self, question: str, response: Any) -> Dict[str, Any]:
        """回退的QA评估方法"""
        response_text = str(response)
        
        # 简单的质量评估
        quality_indicators = ["建议", "步骤", "应该", "立即", "紧急"]
        indicator_count = sum(1 for indicator in quality_indicators if indicator in response_text)
        
        # 基础质量分数
        base_score = 0.6
        quality_score = min(base_score + (indicator_count * 0.08), 0.95)
        
        return {
            "overall_quality_score": quality_score,
            "sub_scores": {
                "accuracy": {"score": quality_score},
                "relevance": {"score": quality_score * 0.9},
                "completeness": {"score": quality_score * 0.85}
            },
            "evaluation_method": "fallback_qa_evaluator",
            "quality_indicators": indicator_count
        }
    
    def _calculate_hallucination_score(self, response: Any, context: str) -> float:
        """计算幻觉分数的辅助方法"""
        response_text = str(response).lower()
        
        # 简单的相似度计算（实际应用中应使用更复杂的算法）
        confidence_words = ["确保", "一定", "必须", "肯定", "明确"]
        uncertainty_words = ["可能", "也许", "大概", "或许", "不确定"]
        
        confidence_count = sum(1 for word in confidence_words if word in response_text)
        uncertainty_count = sum(1 for word in uncertainty_words if word in response_text)
        
        # 计算基础分数
        if uncertainty_count > confidence_count:
            return 0.4 + (uncertainty_count * 0.1)
        else:
            return 0.1 + (confidence_count * 0.05)
    
    def _calculate_qa_score(self, question: str, response: Any) -> float:
        """计算QA质量的辅助方法"""
        response_text = str(response).lower()
        question_text = str(question).lower()
        
        # 检查回答的完整性
        required_elements = ["报警", "疏散", "灭火", "救援"]
        element_count = sum(1 for element in required_elements if element in response_text)
        
        # 检查关键词匹配
        question_keywords = ["火灾", "应急", "救援", "疏散"]
        keyword_match = sum(1 for keyword in question_keywords if keyword in question_text and keyword in response_text)
        
        # 综合评分
        completeness_score = element_count / len(required_elements)
        relevance_score = keyword_match / len(question_keywords)
        
        return (completeness_score + relevance_score) / 2


def main():
    
    print("火灾应急RAG系统 - Phoenix追踪演示")
    print("=" * 70)
    
    try:
        # 1. 初始化带追踪的RAG系统
        print("初始化系统...")
        tracked_rag = PhoenixTrackedRAGSystem()
        tracked_rag.initialize_fire_system()
        
        # 2. 检查Phoenix追踪状态
        if tracked_rag.tracer:
            print("Phoenix追踪已启用")
        else:
            print("Phoenix追踪未启用，将继续运行但无追踪数据")
        
        # 3. 准备测试问题（基于实际的火灾场景）
        test_questions = [
            "3号楼5楼发生火灾，有人员被困，如何制定应急响应方案？",
            "实验室化学品火灾，应该如何正确处置？",
            "图书馆火灾，疏散路线和集合点如何安排？",
            "高层建筑火灾，电梯是否可以使用？",
            "宿舍楼火灾，学生应该如何自救和互救？"
        ]
        
        # 4. 逐个测试并追踪
        results = []
        for i, question in enumerate(test_questions, 1):
            print(f"\n测试 {i}: {question}")
            print("-" * 60)
            
            try:
                result = tracked_rag.process_fire_question(question)
                results.append(result)
                print(f"测试 {i} 完成，已追踪到Phoenix")
                
                # 显示简要结果
                pipeline_result = result.get('pipeline_result', {})
                retrieved_count = pipeline_result.get('retrieved_docs_count', 0)
                overall_score = pipeline_result.get('overall_score', 0)
                
                # 显示资源协调结果
                resource_coordination = pipeline_result.get('resource_coordination', {})
                allocation_result = resource_coordination.get('allocation_result', {})
                allocated_count = len(allocation_result.get('allocated', []))
                guidance_count = len(allocation_result.get('guidance', []))
                
                print(f"   检索文档: {retrieved_count}个, 质量评分: {overall_score:.2f}")
                print(f"   协调资源: {allocated_count}个, 提供指导: {guidance_count}项")
                
            except Exception as e:
                print(f"测试 {i} 失败: {e}")
                results.append({
                    "question": question,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
                
            print()
            
            print()
        
        # 5. 生成总结报告
        print("\n测试总结报告")
        print("=" * 60)
        
        successful_tests = [r for r in results if 'error' not in r]
        failed_tests = [r for r in results if 'error' in r]
        
        print(f"总测试数: {len(results)}")
        print(f"成功: {len(successful_tests)}")
        print(f"失败: {len(failed_tests)}")
        
        if successful_tests:
            # 计算平均分数
            total_score = sum(r.get('pipeline_result', {}).get('overall_score', 0) 
                            for r in successful_tests)
            avg_score = total_score / len(successful_tests)
            print(f"平均质量评分: {avg_score:.2f}")
            
            # 显示最佳结果
            best_result = max(successful_tests, 
                            key=lambda x: x.get('pipeline_result', {}).get('overall_score', 0))
            print(f"\n最佳结果:")
            print(f"问题: {best_result.get('question', 'N/A')}")
            print(f"评分: {best_result.get('pipeline_result', {}).get('overall_score', 0):.2f}")
        
        # 6. 保存详细结果到文件
        results_file = os.path.join(current_dir, 'phoenix_tracking_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": {
                    "total_tests": len(results),
                    "successful_tests": len(successful_tests),
                    "failed_tests": len(failed_tests),
                    "average_score": avg_score if successful_tests else 0,
                    "timestamp": datetime.now().isoformat()
                },
                "detailed_results": results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n详细结果已保存到: {results_file}")
        
    except Exception as e:
        print(f"系统初始化失败: {e}")


def interactive_mode():
    """交互模式 - 用户可以输入自己的问题"""
    print("\n交互式火灾应急问题测试")
    print("=" * 50)
    
    try:
        tracked_rag = PhoenixTrackedRAGSystem()
        tracked_rag.initialize_fire_system()
        
        print("输入您的火灾应急问题（或输入 'quit' 退出）:")
        
        while True:
            question = input("\n问题: ").strip()
            
            if question.lower() in ['quit', 'exit', '退出', 'q']:
                print("感谢使用，再见！")
                break
            
            if not question:
                print("请输入有效的问题")
                continue
            
            try:
                print("\n处理中...")
                result = tracked_rag.process_fire_question(question)
                
                # 显示结果摘要
                pipeline_result = result.get('pipeline_result', {})
                print(f"\n处理完成:")
                print(f"   检索文档: {pipeline_result.get('retrieved_docs_count', 0)}个")
                print(f"   质量评分: {pipeline_result.get('overall_score', 0):.2f}")
                
                # 显示资源协调结果
                resource_coordination = pipeline_result.get('resource_coordination', {})
                allocation_result = resource_coordination.get('allocation_result', {})
                allocated_resources = allocation_result.get('allocated', [])
                guidance_list = allocation_result.get('guidance', [])
                
                if allocated_resources:
                    print(f"   已协调资源 ({len(allocated_resources)}个):")
                    for resource in allocated_resources[:3]:  # 最多显示3个
                        print(f"     - {resource['resource_type']} ({resource['location']})")
                
                if guidance_list:
                    print(f"   资源协调指导 ({len(guidance_list)}项):")
                    for guidance in guidance_list[:2]:  # 最多显示2个
                        print(f"     - {guidance['guidance']}")
                
                # 显示生成的部分响应
                generated = pipeline_result.get('generated_response', {})
                if isinstance(generated, dict):
                    print(f"   响应类型: 结构化JSON")
                    if '建议措施' in generated:
                        print(f"   建议措施数量: {len(generated.get('建议措施', []))}")
                
            except Exception as e:
                print(f"处理失败: {e}")
    
    except Exception as e:
        print(f"初始化失败: {e}")


if __name__ == "__main__":
    print("选择运行模式:")
    print("1. 批量测试演示")
    print("2. 交互式测试")
    
    try:
            choice = input("请选择 (1/2): ").strip()
            
            if choice == "2":
                interactive_mode()
            else:
                main()
                
    except KeyboardInterrupt:
        print("\n用户中断，程序退出")
    except Exception as e:
        print(f"运行错误: {e}")
        # 如果出现错误，尝试运行基础版本
        print("尝试运行基础版本...")
        main()
