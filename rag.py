import os
import re
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import jieba

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """
    文档处理器 - 负责加载、解析和分块文档（使用LangChain）
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        初始化文档处理器
        
        Args:
            chunk_size: 文档块的大小（字符数）
            chunk_overlap: 文档块之间的重叠大小
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 初始化LangChain的文本分块器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "", "\s", "", ""],
        )
    
    def load_document(self, file_path: str) -> Optional[List[Dict[str, Any]]]:
        """
        使用LangChain加载单个文件并分块
        
        Args:
            file_path: 文件路径
            
        Returns:
            文档块列表
        """
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return None
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            # 根据文件类型选择合适的加载器
            if file_extension == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
                documents = loader.load()
                # 分块处理
                chunked_documents = self.text_splitter.split_documents(documents)
                
                # 转换为我们需要的格式
                chunks = []
                for i, doc in enumerate(chunked_documents):
                    chunks.append({
                        'content': doc.page_content,
                        'source': file_path,
                        'metadata': doc.metadata,
                        'chunk_id': i
                    })
                return chunks
            elif file_extension == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 将JSON数据序列化为字符串
                    content = json.dumps(data, ensure_ascii=False, indent=2)
                    
                    # 手动分块
                    chunks = []
                    start = 0
                    text_length = len(content)
                    
                    while start < text_length:
                        end = min(start + self.chunk_size, text_length)
                        
                        # 尝试在句子边界处分割
                        if end < text_length:
                            while end > start and content[end] not in ['.', '。', '\n', '\r', '!', '！', '?', '？']:
                                end -= 1
                            
                            if end == start:
                                end = min(start + self.chunk_size, text_length)
                        
                        chunk_text = content[start:end].strip()
                        if chunk_text:
                            chunks.append({
                                'content': chunk_text,
                                'source': os.path.basename(file_path),
                                'chunk_id': len(chunks)
                            })
                        
                        start = end - self.chunk_overlap
                        if start >= end:
                            start = end
                    
                    return chunks
            else:
                print(f"警告: 不支持的文件类型 {file_extension}")
                return None
        except Exception as e:
            print(f"加载文档失败: {str(e)}")
            return None
    
    def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        使用LangChain加载目录中的所有支持的文件
        
        Args:
            directory_path: 目录路径
            
        Returns:
            所有文档块的列表
        """
        all_chunks = []
        
        if not os.path.exists(directory_path):
            print(f"目录不存在: {directory_path}")
            return all_chunks
        
        # 支持的文件扩展名
        supported_extensions = ['.txt', '.json']
        
        # 逐个加载文件
        loaded_count = 0
        
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_extension = os.path.splitext(file)[1].lower()
                if file_extension in supported_extensions:
                    file_path = os.path.join(root, file)
                    print(f"正在加载文件: {file_path}")
                    try:
                        chunks = self.load_document(file_path)
                        if chunks:
                            loaded_count += 1
                            print(f"成功加载文件 {file_path}，获得 {len(chunks)} 个文档块")
                            # 确保添加时间戳
                            for chunk in chunks:
                                if 'timestamp' not in chunk:
                                    chunk['timestamp'] = datetime.now().isoformat()
                            all_chunks.extend(chunks)
                    except Exception as e:
                        print(f"加载文件 {file_path} 时出错: {str(e)}")
        
        print(f"总共加载了 {len(all_chunks)} 个文档块")
        return all_chunks


class ChineseTokenizer:
    """
    中文分词器，用于TF-IDF向量化
    """
    def __init__(self):
        """
        初始化中文分词器
        """
        import re
        self.re = re
        
    def tokenize(self, text):
        """
        使用jieba对中文文本进行分词
        
        Args:
            text: 输入文本
            
        Returns:
            分词后的词列表
        """
        # 移除特殊字符和数字
        text = self.re.sub(r'[^\u4e00-\u9fa5a-zA-Z]', ' ', text)
        # 使用jieba分词
        words = jieba.lcut(text)
        # 过滤掉空格
        words = [word for word in words if word.strip()]
        return words


class VectorStore:
    """
    向量存储 - 负责文档块的向量化和索引管理
    """
    
    def __init__(self):
        """
        初始化向量存储
        """
        self.vectorizer = None
        self.vectorized_docs = None
        self.document_chunks = None
        self.documents = None  # 为了兼容RAGSystem中的引用
        self.tokenizer = ChineseTokenizer()
    
    def add_documents(self, documents):
        """
        添加文档到向量存储
        
        Args:
            documents: 文档块列表
        """
        self.document_chunks = documents
        self.documents = documents
        return True
    
    def build_index(self, document_chunks: Optional[List[Dict[str, Any]]] = None):
        """
        构建文档块索引
        
        Args:
            document_chunks: 文档块列表（可选）
        """
        if document_chunks is not None:
            self.document_chunks = document_chunks
        
        self.documents = self.document_chunks
        
        if not self.document_chunks:
            print("没有文档块可以索引")
            return False
        
        try:
            # 过滤掉内容太短的文档块
            valid_chunks = [chunk for chunk in self.document_chunks if len(chunk['content']) > 10]
            self.document_chunks = valid_chunks
            self.documents = valid_chunks
            
            print(f"过滤后有效文档块数量: {len(valid_chunks)}")
            
            # 初始化TF-IDF向量化器
            self.vectorizer = TfidfVectorizer(
                tokenizer=self.tokenizer.tokenize,
                lowercase=False,
                max_df=0.95,
                min_df=1,
                use_idf=True,
                smooth_idf=True,
                sublinear_tf=True
            )
            
            # 向量化文档块
            chunk_contents = [chunk['content'] for chunk in valid_chunks]
            self.vectorized_docs = self.vectorizer.fit_transform(chunk_contents)
            
            # 输出词汇表信息
            feature_names = self.vectorizer.get_feature_names_out()
            print(f"词汇表大小: {len(feature_names)}")
            if len(feature_names) > 0:
                print(f"前10个词汇: {feature_names[:10]}")
            
            return True
        except Exception as e:
            print(f"索引建立失败: {e}")
            return False
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        搜索相关文档块
        
        Args:
            query: 搜索查询
            top_k: 返回前k个结果
            
        Returns:
            搜索结果列表
        """
        if self.vectorizer is None or self.vectorized_docs is None:
            print("索引尚未建立")
            return []
        
        try:
            # 向量化查询
            vectorized_query = self.vectorizer.transform([query])
            
            # 计算余弦相似度
            similarities = cosine_similarity(vectorized_query, self.vectorized_docs).flatten()
            
            # 获取排序后的索引
            sorted_indices = similarities.argsort()[::-1][:top_k]
            
            # 构建结果
            results = []
            
            for idx in sorted_indices:
                chunk = self.document_chunks[idx]
                similarity = similarities[idx]
                
                if similarity > 0:
                    results.append({
                        'source': chunk['source'],
                        'content': chunk['content'],
                        'similarity': float(similarity),
                        'metadata': chunk.get('metadata', {}),
                        'chunk_id': chunk.get('chunk_id', -1)
                    })
                    
                    # 记录匹配到的内容摘要
                    content_preview = chunk['content'][:200] + '...' if len(chunk['content']) > 200 else chunk['content']
                    print(f"匹配到文档块: 来源={chunk['source']}, 相似度={similarity:.4f}, 内容摘要={content_preview}")
            
            return results
        except Exception as e:
            print(f"搜索失败: {e}")
            return []
    
    def save_index(self, file_path: str):
        """
        保存索引到文件
        
        Args:
            file_path: 保存路径
        """
        try:
            with open(file_path, 'wb') as f:
                pickle.dump({
                    'vectorizer': self.vectorizer,
                    'vectorized_docs': self.vectorized_docs,
                    'document_chunks': self.document_chunks,
                    'documents': self.documents
                }, f)
            return True
        except Exception as e:
            print(f"保存索引失败: {e}")
            return False
    
    def load_index(self, file_path: str):
        """
        从文件加载索引
        
        Args:
            file_path: 加载路径
        """
        try:
            if not os.path.exists(file_path):
                print(f"索引文件不存在: {file_path}")
                return False
            
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                
            self.vectorizer = data['vectorizer']
            self.vectorized_docs = data['vectorized_docs']
            self.document_chunks = data['document_chunks']
            self.documents = data.get('documents', self.document_chunks)
            
            print(f"成功加载索引，包含 {len(self.document_chunks)} 个文档块")
            return True
        except Exception as e:
            print(f"加载索引失败: {e}")
            return False


class RAGEnhancer:
    """
    RAG增强器 - 负责利用检索结果增强提示词
    """
    
    def __init__(self, vector_store: VectorStore):
        """
        初始化RAG增强器
        
        Args:
            vector_store: 向量存储实例
        """
        self.vector_store = vector_store
    
    def enhance_prompt(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        增强提示词
        
        Args:
            query: 原始查询
            top_k: 返回前k个检索结果
            
        Returns:
            包含增强提示词和检索结果的字典
        """
        # 搜索相关文档块
        search_results = self.vector_store.search(query, top_k=top_k)
        
        # 构建增强提示词
        if search_results:
            enhanced_prompt = f"基于以下参考信息，请回答问题：\n\n"
            
            for i, result in enumerate(search_results, 1):
                enhanced_prompt += f"参考信息 {i}:\n{result['content']}\n\n"
            
            enhanced_prompt += f"问题: {query}"
        else:
            enhanced_prompt = query
        
        return {
            'enhanced_prompt': enhanced_prompt,
            'search_results': search_results
        }


class RAGSystem:
    """
    RAG系统 - 整合文档处理、向量存储和增强功能
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        初始化RAG系统
        
        Args:
            chunk_size: 文档分块大小
            chunk_overlap: 块之间的重叠大小
        """
        self.document_processor = DocumentProcessor(chunk_size, chunk_overlap)
        self.vector_store = VectorStore()
        self.rag_enhancer = RAGEnhancer(self.vector_store)
        self.all_chunks = []
    
    def add_documents_from_directory(self, directory_path: str):
        """
        从目录添加文档到RAG系统
        
        Args:
            directory_path: 文档目录路径
        """
        # 处理目录中的所有文档
        documents = self.document_processor.process_directory(directory_path)
        
        if documents:
            # 添加到向量存储和本地存储
            self.all_chunks.extend(documents)
            self.vector_store.add_documents(documents)
            # 构建索引
            self.vector_store.build_index()
            return True
        return False
    
    def enhance_prompt(self, query: str, top_k: int = 3) -> str:
        """
        增强查询
        
        Args:
            query: 原始查询
            top_k: 返回前k个检索结果
            
        Returns:
            增强后的提示词
        """
        result = self.rag_enhancer.enhance_prompt(query, top_k=top_k)
        
        # 记录检索结果统计信息
        if result['search_results']:
            print(f"为查询 '{query}' 找到了 {len(result['search_results'])} 个相关文档块")
        
        return result['enhanced_prompt']


# 工具函数
def create_rag_system(docs_dir: Optional[str] = None, chunk_size: int = 500, chunk_overlap: int = 50) -> RAGSystem:
    """
    创建并初始化RAG系统
    
    Args:
        docs_dir: 可选的文档目录路径
        chunk_size: 文档块大小
        chunk_overlap: 文档块重叠大小
    
    Returns:
        初始化的RAG系统实例
    """
    rag_system = RAGSystem(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    if docs_dir and os.path.exists(docs_dir):
        rag_system.add_documents_from_directory(docs_dir)
    
    return rag_system