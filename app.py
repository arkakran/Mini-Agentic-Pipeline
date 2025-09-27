import os
import json
import time
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

from utils.retriever import VectorRetriever
from utils.reasoner import LLMReasoner
from utils.actor import WebSearchActor
from utils.evaluator import PipelineEvaluator

load_dotenv()

class MiniAgenticPipeline:
    def __init__(self):
        self.retriever = VectorRetriever(
            model_name=os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2'),
            vector_store_path=os.getenv('VECTOR_STORE_PATH', './data/faiss_index')
        )
        
        self.reasoner = LLMReasoner(
            api_key=os.getenv('GROQ_API_KEY'),
            model=os.getenv('LLM_MODEL', 'llama-3.3-70b-versatile')
        )
        
        self.actor = WebSearchActor(
            tavily_api_key=os.getenv('TAVILY_API_KEY')
        )
        
        self.confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', '0.7'))
        
        # Setup logging
        os.makedirs('./logs', exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('./logs/pipeline.log'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.traces = []
        
        self.setup_retriever()
    
    def setup_retriever(self):
        if not self.retriever.load_index():
            self.logger.info("Building new vector index...")
            if os.path.exists('my_data.txt'):
                self.retriever.build_index('my_data.txt')
            else:
                self.logger.error("my_data.txt not found! Please create the knowledge base file.")
                raise FileNotFoundError("Knowledge base file my_data.txt not found")
        else:
            self.logger.info("Loaded existing vector index")
    
    def process_query(self, query: str) -> tuple:
        trace = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'steps': []
        }
        
        start_time = time.time()
        
        try:
            # Step 1: Retrieve from knowledge base
            self.logger.info(f"Processing query: {query}")
            trace['steps'].append("Step 1: Searching knowledge base...")
            
            kb_results, max_similarity = self.retriever.search(query, top_k=5)
            trace['kb_results'] = kb_results
            trace['max_similarity'] = max_similarity
            
            # Step 2: Reason about next action
            trace['steps'].append("Step 2: Reasoning about best approach...")
            decision = self.reasoner.decide_action(query, kb_results, max_similarity, self.confidence_threshold)
            trace['decision'] = decision
            
            # Step 3: Execute action
            if decision['decision'] == 'use_kb':
                trace['steps'].append("Step 3: Generating answer from knowledge base...")
                answer = self.reasoner.generate_answer_from_kb(query, kb_results)
                trace['source'] = 'knowledge_base'
                trace['web_results'] = []
            else:
                trace['steps'].append("Step 3: Searching web for current information...")
                web_results, web_latency = self.actor.search_web(query)
                trace['web_results'] = web_results
                trace['web_search_latency'] = web_latency
                
                answer = self.reasoner.generate_answer_from_web(query, web_results)
                trace['source'] = 'web_search'
            
            trace['answer'] = answer
            trace['total_latency'] = time.time() - start_time
            trace['status'] = 'success'
            
            # Save trace
            self.traces.append(trace)
            self.save_trace(trace)
            
            return answer, trace
            
        except Exception as e:
            trace['error'] = str(e)
            trace['status'] = 'error'
            trace['total_latency'] = time.time() - start_time
            self.logger.error(f"Pipeline error: {e}")
            return f"Error processing query: {e}", trace
    
    def save_trace(self, trace):
        try:
            timestamp = datetime.now().strftime("%Y%m%d")
            filename = f"./logs/traces_{timestamp}.jsonl"
            
            with open(filename, 'a') as f:
                f.write(json.dumps(trace) + '\n')
                
        except Exception as e:
            self.logger.error(f"Error saving trace: {e}")
    
    def get_system_stats(self):
        #Get pipeline statistics
        return {
            'retriever_stats': self.retriever.get_stats(),
            'total_traces': len(self.traces),
            'reasoner_model': self.reasoner.model,
            'search_service': self.actor.get_search_stats()
        }

app = Flask(__name__)
pipeline = None


@app.route('/')
def index():
    """Main interface"""
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def handle_query():
    """Handle user queries"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        if 'pipeline' not in globals() or pipeline is None:
            return jsonify({'error': 'Pipeline not initialized'}), 500
        
        answer, trace = pipeline.process_query(query)
        
        return jsonify({
            'answer': answer,
            'trace': {
                'decision': trace.get('decision', {}),
                'source': trace.get('source', ''),
                'latency': trace.get('total_latency', 0),
                'steps': trace.get('steps', []),
                'kb_results_count': len(trace.get('kb_results', [])),
                'web_results_count': len(trace.get('web_results', [])),
                'max_similarity': trace.get('max_similarity', 0.0)
            }
        })
        
    except Exception as e:
        app.logger.error(f"Error handling query: {e}")
        return jsonify({'error': f'Internal server error: {e}'}), 500

@app.route('/stats')
def get_stats():
    #Get system statistics
    try:
        if not pipeline:
            return jsonify({'error': 'Pipeline not initialized'}), 500
        
        stats = pipeline.get_system_stats()
        return jsonify(stats)
        
    except Exception as e:
        app.logger.error(f"Error getting stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/evaluate')
def run_evaluation():
    #Run pipeline evaluation
    try:
        if not pipeline:
            return jsonify({'error': 'Pipeline not initialized'}), 500
        
        evaluator = PipelineEvaluator()
        results = evaluator.evaluate_pipeline(pipeline)
        report = evaluator.generate_report(results)
        
        return jsonify({
            'results': results,
            'report': report
        })
        
    except Exception as e:
        app.logger.error(f"Error running evaluation: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ensure req dir exist
    os.makedirs('./data', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./templates', exist_ok=True)
    os.makedirs('./static', exist_ok=True)
    
    # Check for required environment variables
    required_vars = ['GROQ_API_KEY', 'TAVILY_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {missing_vars}")
        print("Please create a .env file with your API keys")
        exit(1)
    
    # Initialize pipeline
    try:
        pipeline = MiniAgenticPipeline()
        print("Pipeline initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        exit(1)
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)