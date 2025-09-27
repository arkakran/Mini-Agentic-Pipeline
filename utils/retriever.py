import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Tuple, Dict
import logging

class VectorRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", vector_store_path: str = "./data/faiss_index"):
        self.model = SentenceTransformer(model_name)
        self.vector_store_path = vector_store_path
        self.index = None
        self.documents = []
        self.metadata = []
        self.dimension = None
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
        os.makedirs("./logs", exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_and_chunk_documents(self, file_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict]:
        #Load documents and split them into chunks
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Split by document sections
            sections = text.split('## Document')
            chunks = []
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " "]
            )
            
            for i, section in enumerate(sections[1:], 1):  # Skip first empty section
                if section.strip():
                    section_chunks = text_splitter.split_text(section.strip())
                    for j, chunk in enumerate(section_chunks):
                        chunks.append({
                            'content': chunk,
                            'doc_id': i,
                            'chunk_id': j,
                            'source': f"Document_{i}"
                        })
            
            self.logger.info(f"Created {len(chunks)} chunks from {len(sections)-1} documents")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error loading documents: {e}")
            return []
    
    def build_index(self, file_path: str):
        try:
            chunks = self.load_and_chunk_documents(file_path)
            if not chunks:
                raise ValueError("No chunks created from documents")
            
            # Extract texts & metadata
            texts = [chunk['content'] for chunk in chunks]
            self.metadata = chunks
            
            self.logger.info("Generating embeddings...")
            embeddings = self.model.encode(texts, show_progress_bar=True)
            embeddings = np.array(embeddings).astype('float32')
            
            self.dimension = embeddings.shape[1]
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)  
            self.index.hnsw.efConstruction = 200  
            
            self.index.add(embeddings)
            self.documents = texts
            
            self.save_index()
            self.logger.info(f"Built index with {len(texts)} chunks, dimension {self.dimension}")
            
        except Exception as e:
            self.logger.error(f"Error building index: {e}")
            raise
    
    def save_index(self):
        try:
            faiss.write_index(self.index, f"{self.vector_store_path}.faiss")
            
            with open(f"{self.vector_store_path}_metadata.pkl", 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'metadata': self.metadata,
                    'dimension': self.dimension
                }, f)
            
            self.logger.info("Index saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving index: {e}")
    
    def load_index(self):
        try:
            index_file = f"{self.vector_store_path}.faiss"
            metadata_file = f"{self.vector_store_path}_metadata.pkl"
            
            if os.path.exists(index_file) and os.path.exists(metadata_file):
                self.index = faiss.read_index(index_file)
                
                with open(metadata_file, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.metadata = data['metadata']
                    self.dimension = data['dimension']
                
                self.logger.info(f"Loaded index with {len(self.documents)} documents")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error loading index: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> Tuple[List[Dict], float]:
        try:
            if self.index is None:
                return [], 0.0
            
            query_embedding = self.model.encode([query]).astype('float32')
            
            # Search
            distances, indices = self.index.search(query_embedding, top_k)
            
            similarities = 1 / (1 + distances[0])
            max_similarity = float(np.max(similarities)) if len(similarities) > 0 else 0.0
            
            results = []
            for i, (idx, sim) in enumerate(zip(indices[0], similarities)):
                if idx < len(self.documents):  # Valid index
                    results.append({
                        'content': self.documents[idx],
                        'similarity': float(sim),
                        'metadata': self.metadata[idx],
                        'rank': i + 1
                    })
            
            self.logger.info(f"Found {len(results)} results for query, max similarity: {max_similarity:.3f}")
            return results, max_similarity
            
        except Exception as e:
            self.logger.error(f"Error in search: {e}")
            return [], 0.0
    
    def get_stats(self) -> Dict:
        return {
            'total_documents': len(self.documents) if self.documents else 0,
            'dimension': self.dimension,
            'index_type': 'HNSW' if self.index else None,
            'model_name': self.model.get_sentence_embedding_dimension()
        }