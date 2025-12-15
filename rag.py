import os
import re
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 引入 DashScope
import dashscope
from http import HTTPStatus


class DocumentProcessor:
    """文档处理器 - 负责加载、解析和分块文档"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "", "\s", "", ""],
        )
    
    def load_document(self, file_path: str) -> Optional[List[Dict[str, Any]]]:
        if not os.path.exists(file_path):
            return None
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
                documents = loader.load()
                chunked_documents = self.text_splitter.split_documents(documents)
                
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
                    content = json.dumps(data, ensure_ascii=False, indent=2)
                    chunks = []
                    # 简单分块逻辑
                    start = 0
                    text_length = len(content)
                    while start < text_length:
                        end = min(start + self.chunk_size, text_length)
                        chunk_text = content[start:end].strip()
                        if chunk_text:
                            chunks.append({
                                'content': chunk_text,
                                'source': os.path.basename(file_path),
                                'chunk_id': len(chunks)
                            })
                        start = end - self.chunk_overlap
                        if start >= end: start = end
                    return chunks
            return None
        except Exception as e:
            print(f"加载文档失败 {file_path}: {str(e)}")
            return None
    
    def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        all_chunks = []
        if not os.path.exists(directory_path):
            return all_chunks
        
        supported_extensions = ['.txt', '.json']
        
        for root, _, files in os.walk(directory_path):
            for file in files:
                if os.path.splitext(file)[1].lower() in supported_extensions:
                    file_path = os.path.join(root, file)
                    chunks = self.load_document(file_path)
                    if chunks:
                        for chunk in chunks:
                            chunk['timestamp'] = datetime.now().isoformat()
                        all_chunks.extend(chunks)
        return all_chunks


class VectorStore:
    """向量存储 - 修复 API Key 问题"""
    def __init__(self, api_key: str = None):
        self.documents = []
        self.vectors = None
        # 优先使用传入的 key，否则读取环境变量
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        
        # [关键修复] 强制设置 dashscope 全局变量，防止 SDK 某些版本读取不到
        if self.api_key:
            dashscope.api_key = self.api_key
            # 调试打印 (隐藏部分key)
            masked_key = self.api_key[:6] + "***" + self.api_key[-4:] if len(self.api_key) > 10 else "***"
            print(f"VectorStore 已初始化，API Key: {masked_key}")
        else:
            print("警告: VectorStore 初始化时未检测到 API Key！向量化将失败。")

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        
        if not self.api_key:
            print("错误: 缺少 API Key，无法生成向量。")
            return np.random.rand(len(texts), 1536) # 防止程序崩溃
            
        try:
            # 显式传递 api_key
            resp = dashscope.TextEmbedding.call(
                model=dashscope.TextEmbedding.Models.text_embedding_v2,
                input=texts,
                api_key=self.api_key
            )
            if resp.status_code == HTTPStatus.OK:
                embeddings = [item['embedding'] for item in resp.output['embeddings']]
                return np.array(embeddings)
            else:
                print(f"Embedding API Error: {resp}")
                return np.random.rand(len(texts), 1536)
        except Exception as e:
            print(f"获取向量异常: {e}")
            return np.random.rand(len(texts), 1536)

    def add_documents(self, documents):
        self.documents = documents

    def build_index(self, documents: Optional[List[Dict[str, Any]]] = None):
        if documents:
            self.documents = documents
        
        if not self.documents:
            return False
        
        texts = [doc['content'] for doc in self.documents]
        batch_size = 10
        vector_list = []
        
        print(f"正在为 {len(texts)} 个文档块生成向量索引...")
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_vecs = self._get_embeddings(batch_texts)
            if len(batch_vecs) > 0:
                vector_list.append(batch_vecs)
        
        if vector_list:
            self.vectors = np.vstack(vector_list)
            print("索引构建完成")
            return True
        return False

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if self.vectors is None or not self.documents:
            return []
            
        # 查询向量化
        query_vec = self._get_embeddings([query])[0]
        
        # 计算相似度
        norm_docs = np.linalg.norm(self.vectors, axis=1)
        norm_query = np.linalg.norm(query_vec)
        
        if norm_query == 0: return []
            
        similarities = np.dot(self.vectors, query_vec) / (norm_docs * norm_query)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = similarities[idx]
            if score > 0.25: # 降低一点阈值确保能召回
                doc = self.documents[idx].copy()
                doc['similarity'] = float(score)
                results.append(doc)
                
        return results

class RAGSystem:
    def __init__(self, api_key=None, docs_dir=None, chunk_size=500, chunk_overlap=50):
        self.doc_processor = DocumentProcessor(chunk_size, chunk_overlap)
        # 传递 api_key
        self.vector_store = VectorStore(api_key=api_key)
        self.rag_enhancer = None # 简化版不需要单独的Enhancer类，逻辑集成在enhance_prompt
        
        if docs_dir:
            self.initialize(docs_dir)
            
    def initialize(self, docs_dir):
        docs = self.doc_processor.process_directory(docs_dir)
        if docs:
            self.vector_store.build_index(docs)
            
    def add_documents_from_directory(self, docs_dir):
        self.initialize(docs_dir)

    def enhance_prompt(self, query: str, top_k: int = 3) -> str:
        results = self.vector_store.search(query, top_k)
        if not results:
            return query
        context = "\n".join([f"资料{i+1}: {r['content']}" for i, r in enumerate(results)])
        return f"基于以下参考资料回答：\n{context}\n\n问题：{query}"

def create_rag_system(docs_dir=None, api_key=None, chunk_size=500, chunk_overlap=50):
    return RAGSystem(api_key, docs_dir, chunk_size, chunk_overlap)