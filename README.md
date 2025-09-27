# 🤖 Mini Agentic Pipeline

A production-ready AI question answering system that intelligently combines **knowledge base retrieval** with **real-time web search**. The system automatically decides whether to answer from its internal knowledge base or search the web for the most current information.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Flask](https://img.shields.io/badge/flask-v3.0.0-green.svg)

## 🎯 Demo

**Try these queries to see the intelligent routing:**

| Query Type | Example | Expected Source |
|------------|---------|-----------------|
| **Product Info** | "What is the pricing for CloudSync Pro?" | Knowledge Base |
| **Current Events** | "Latest AI developments 2024" | Web Search |
| **Support Info** | "What support options are available?" | Knowledge Base |
| **Real-time Data** | "Current weather in New York" | Web Search |

## 🌟 Key Features

### 🧠 **Intelligent Decision Making**
- Automatically routes queries to best information source
- Confidence-based decision with 0.7 similarity threshold
- LLM-powered validation for edge cases
- Complete execution tracing for transparency

### 🔍 **Advanced Retrieval System**
- **FAISS HNSW** vector search for sub-second performance
- **Sentence Transformers** for semantic understanding
- **20 TechCorp documents** as sample knowledge base
- Configurable similarity thresholds and chunk sizes

### 🌐 **Real-time Web Search**
- **Tavily API** integration for current information
- Advanced search with content extraction
- Source ranking and relevance scoring
- Automatic fallback when KB is insufficient

### 🎨 **Clean Web Interface**
- Responsive design with modern UI
- Real-time query processing with loading states
- Interactive trace visualization
- Built-in performance evaluation tools

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
Internet connection for web search
```

### 1. Clone & Install
```bash
git clone repo_link
cd mini-agentic-pipeline
pip install -r requirements.txt
```

### 2. Get API Keys (Free Tiers Available)
- **Groq API**: [console.groq.com](https://console.groq.com) - Free tier: 14,400 requests/day
- **Tavily API**: [tavily.com](https://tavily.com) - Free tier: 1,000 searches/month

### 3. Configure Environment
```bash
# Create .env file
GROQ_API_KEY=gsk_your_groq_api_key_here
TAVILY_API_KEY=tvly-your_tavily_api_key_here
FLASK_ENV=development
FLASK_DEBUG=True
```

### 4. Run the Application
```bash
python app.py
```

### 5. Open in Browser
```
http://localhost:5000
```

That's it! The system will auto-build the vector index and start serving queries.

## 📁 Project Structure

```
mini-agentic-pipeline/
├── 📄 app.py                     # Main Flask app & pipeline controller
├── 📄 requirements.txt           # Python dependencies  
├── 📄 my_data.txt               # Knowledge base (20 documents)
├── 🗂️ utils/
│   ├── 📄 retriever.py          # FAISS vector search
│   ├── 📄 reasoner.py           # LLM decision making  
│   ├── 📄 actor.py              # Tavily web search
│   └── 📄 evaluator.py          # Performance testing
├── 🗂️ templates/
│   └── 📄 index.html            # Web interface
├── 🗂️ static/  
│   └── 📄 style.css             # Responsive styling
├── 🗂️ tests/
│   └── 📄 test_queries.py       # Automated tests
├── 🗂️ data/                     # Auto-created
│   └── 🗂️ faiss_index/          # Vector store files
└── 🗂️ logs/                     # Auto-created
    ├── 📄 pipeline.log          # System logs
    ├── 📄 traces_*.jsonl        # Query traces
    └── 📄 evaluation_*.json     # Performance reports
```

## 🎯 How It Works

### 1. **Query Flow**
```
User Query → Vector Search → LLM Decision → Action → Response
     ↑                                         ↓
     └─────────── Logging & Tracing ←─────────┘
```

### 2. **Decision Logic**
```python
if similarity_score > 0.7 and relevant_kb_results:
    return "use_knowledge_base"
else:
    return "search_web"
```

### 3. **Example Processing**

**Knowledge Base Query:**
```
Query: "What does CloudSync Pro cost?"
  ↓
Vector Search: similarity=0.95, found pricing info
  ↓  
Decision: use_kb (high confidence)
  ↓
Answer: "CloudSync Pro costs $29.99/month..."
```

**Web Search Query:**
```
Query: "Latest AI news today"
  ↓
Vector Search: similarity=0.3, no current news
  ↓
Decision: web_search (need current info)
  ↓
Tavily Search: finds recent AI articles
  ↓
Answer: "Recent AI developments include..." + sources
```

## 🧪 Testing & Evaluation

### Web Interface Testing
1. Visit `http://localhost:5000`
2. Try example queries
3. Click "🧪 Run Evaluation" for performance metrics
4. View "📊 System Stats" for component status

### Performance Metrics
- **Average Latency**: ~2.5 seconds per query
- **KB Queries**: ~1.5 seconds (local processing)
- **Web Queries**: ~3.0 seconds (network + API calls)
- **Decision Accuracy**: >90% appropriate source selection
- **Success Rate**: >95% successful query completion

## ⚙️ Configuration

### Environment Variables
```bash
# Required
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key

# Optional (with defaults)
CONFIDENCE_THRESHOLD=0.7          # KB similarity threshold
EMBEDDING_MODEL=all-MiniLM-L6-v2  # Sentence transformer model
LLM_MODEL=llama-3.3-70b-versatile # Groq model
CHUNK_SIZE=500                    # Document chunk size
CHUNK_OVERLAP=50                  # Chunk overlap
MAX_DOCUMENTS=20                  # KB document limit
```


## 🔧 Technical Details

### **Vector Search Engine**
- **Model**: all-MiniLM-L6-v2 (384 dimensions)
- **Index**: FAISS HNSW for O(log n) search
- **Chunking**: Recursive character splitter
- **Similarity**: Cosine similarity with L2 normalization

### **Language Model**
- **Provider**: Groq (fast inference)
- **Model**: Llama-3.3-70B-Versatile
- **Temperature**: 0.3 (consistent responses)
- **Context**: 8K tokens max

### **Web Search**
- **Provider**: Tavily (AI-optimized search)
- **Features**: Content extraction, ranking, filtering
- **Rate Limits**: 1000 searches/month (free tier)
- **Timeout**: 30 seconds per search

## 📊 Architecture Decisions

### **Why FAISS?**
- ✅ **Performance**: Sub-second search on 100K+ documents
- ✅ **Memory Efficient**: Compressed vector storage
- ✅ **Scalable**: Handles millions of vectors
- ✅ **Production Ready**: Used by Facebook, Microsoft

### **Why Groq + Llama?**
- ✅ **Speed**: 2-5x faster than OpenAI GPT-4
- ✅ **Quality**: Excellent reasoning capabilities
- ✅ **Cost**: More economical for high usage
- ✅ **Reliability**: Consistent response times

### **Why Tavily?**
- ✅ **AI-Optimized**: Built for LLM applications
- ✅ **Quality**: Better content extraction than Google
- ✅ **Reliability**: 99.9% uptime SLA
- ✅ **Integration**: Simple, clean API

## 🎥 Demo Video

**[🎬 Watch Demo Video](https://drive.google.com/file/d/190XzTVKaV0dxCihZV7szrPCfviz5fQVK/view?usp=drivesdk)** - Sorry for the long explaination 


## 🙏 Acknowledgments

- **[Groq](https://groq.com)** for lightning-fast LLM inference
- **[Tavily](https://tavily.com)** for intelligent web search
- **[Facebook AI](https://github.com/facebookresearch/faiss)** for FAISS vector search
- **[Sentence Transformers](https://www.sbert.net/)** for semantic embeddings
- **[Flask](https://flask.palletsprojects.com/)** for the web framework

## ThankYou
