# 🤖 Mini Agentic Pipeline

A production-ready AI question answering system that intelligently combines **knowledge base retrieval** with **real-time web search**. The system automatically decides whether to answer from its internal knowledge base or search the web for the most current information.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Flask](https://img.shields.io/badge/flask-v3.0.0-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

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
- No complex JavaScript frameworks

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
Internet connection for web search
```

### 1. Clone & Install
```bash
git clone <your-repo-url>
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

### Run Automated Tests
```bash
python tests/test_queries.py
```

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

### Customizing Knowledge Base
Replace `my_data.txt` with your own documents:
```
# Your Knowledge Base

## Document 1: Product Information
Your product details here...

## Document 2: Service Information  
Your service details here...
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

## 🚧 Known Limitations

1. **Knowledge Base Size**: Optimized for 20-100 documents
2. **Language Support**: Best performance with English queries
3. **Context Window**: Long documents may be truncated
4. **API Dependencies**: Requires internet for LLM and web search
5. **Rate Limits**: Free tiers have usage restrictions

## 🔮 Future Enhancements

- [ ] **Multi-modal Search**: Image and PDF processing
- [ ] **Larger Knowledge Bases**: Support for 1000+ documents
- [ ] **Conversation Memory**: Multi-turn dialogue support
- [ ] **Custom Embeddings**: Domain-specific models
- [ ] **Caching Layer**: Redis for performance optimization
- [ ] **REST API**: Programmatic access endpoints
- [ ] **Multi-language**: Support for non-English queries

## 🎥 Demo Video

**[🎬 Watch 7-minute Demo](your-video-link-here)**

The demo covers:
- **Architecture Overview** (2 minutes)
- **Live Query Examples** (3 minutes) 
- **Performance Analysis** (2 minutes)

## 🤝 Contributing

We welcome contributions! Here's how:

1. **Fork** the repository
2. **Create** feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** changes: `git commit -m 'Add amazing feature'`
4. **Push** to branch: `git push origin feature/amazing-feature`
5. **Open** Pull Request

### Development Setup
```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black . && isort .
```

## 📄 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[Groq](https://groq.com)** for lightning-fast LLM inference
- **[Tavily](https://tavily.com)** for intelligent web search
- **[Facebook AI](https://github.com/facebookresearch/faiss)** for FAISS vector search
- **[Sentence Transformers](https://www.sbert.net/)** for semantic embeddings
- **[Flask](https://flask.palletsprojects.com/)** for the web framework

## 🆘 Support

**Having issues?**

1. **Check** the [troubleshooting guide](#🔧-troubleshooting)
2. **Search** existing [issues](../../issues)
3. **Create** a [new issue](../../issues/new) with:
   - Python version
   - Error messages
   - Steps to reproduce

## 🔧 Troubleshooting

### Common Issues

**"Pipeline initialization failed"**
```bash
# Check API keys in .env file
cat .env | grep API_KEY

# Test Groq connection
python -c "import requests; print('Groq:', requests.get('https://api.groq.com').status_code)"
```

**"FAISS import error"**
```bash
pip uninstall faiss-cpu faiss-gpu
pip install faiss-cpu==1.7.4
```

**"Port 5000 already in use"**
```python
# In app.py, change:
app.run(debug=True, host='0.0.0.0', port=8080)  # Use different port
```

**"Tavily search failed"**
```bash
# Check API key and quota
curl -X POST "https://api.tavily.com/search" \
  -H "Content-Type: application/json" \
  -d '{"api_key": "your_key", "query": "test", "max_results": 1}'
```

---

**Built with ❤️ for intelligent question answering**

*Star ⭐ this repo if it helped you build something awesome!*
