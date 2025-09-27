import json
import time
import statistics
from typing import List, Dict, Any
from datetime import datetime
import logging

class PipelineEvaluator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.test_queries = [
            "What is the pricing for CloudSync Pro?",
            "How much does DataGuard Security Suite cost?",
            "What features are included in WebBuilder Studio?",
            "Tell me about the Analytics Dashboard Pro",
            "What support options are available?",
            "How does the Mobile App Suite work?",
            "What is the latest news about artificial intelligence?",
            "Current weather in New York",
            "Recent developments in quantum computing",
            "What are the best practices for cloud security in 2024?",
            "Latest updates in machine learning frameworks",
            "Current stock market trends today"
        ]
        
        self.evaluation_results = []
    
    def evaluate_pipeline(self, pipeline, save_results: bool = True) -> Dict:
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_queries': len(self.test_queries),
            'query_results': [],
            'performance_metrics': {}
        }
        
        total_latency = 0
        kb_used_count = 0
        web_search_count = 0
        successful_responses = 0
        
        for i, query in enumerate(self.test_queries):
            self.logger.info(f"Evaluating query {i+1}/{len(self.test_queries)}: {query}")
            
            start_time = time.time()
            try:
                # Process query through pipeline
                response, trace = pipeline.process_query(query)
                total_time = time.time() - start_time
                
                # Count decision types
                if trace.get('decision', {}).get('decision') == 'use_kb':
                    kb_used_count += 1
                elif trace.get('decision', {}).get('decision') == 'web_search':
                    web_search_count += 1
                
                if response and len(response) > 10:  # Basic quality check
                    successful_responses += 1
                
                total_latency += total_time
                
                query_result = {
                    'query': query,
                    'response': response,
                    'latency': total_time,
                    'decision': trace.get('decision', {}),
                    'kb_results_count': len(trace.get('kb_results', [])),
                    'web_results_count': len(trace.get('web_results', [])),
                    'max_similarity': trace.get('max_similarity', 0.0),
                    'success': len(response) > 10 if response else False
                }
                
                results['query_results'].append(query_result)
                
            except Exception as e:
                total_time = time.time() - start_time
                self.logger.error(f"Error evaluating query '{query}': {e}")
                
                results['query_results'].append({
                    'query': query,
                    'response': f"Error: {e}",
                    'latency': total_time,
                    'success': False
                })
        
        # Calculate performance metrics
        latencies = [r['latency'] for r in results['query_results'] if 'latency' in r]
        
        results['performance_metrics'] = {
            'average_latency': statistics.mean(latencies) if latencies else 0,
            'median_latency': statistics.median(latencies) if latencies else 0,
            'max_latency': max(latencies) if latencies else 0,
            'min_latency': min(latencies) if latencies else 0,
            'total_time': total_latency,
            'success_rate': successful_responses / len(self.test_queries),
            'kb_usage_rate': kb_used_count / len(self.test_queries),
            'web_search_rate': web_search_count / len(self.test_queries)
        }
        
        if save_results:
            self.save_evaluation_results(results)
        
        return results
    
    def save_evaluation_results(self, results: Dict):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"./logs/evaluation_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            
            self.logger.info(f"Evaluation results saved to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving evaluation results: {e}")
    
    def generate_report(self, results: Dict) -> str:
        """Generate a formatted evaluation report"""
        metrics = results['performance_metrics']
        
        report = f"""
            # Mini Agentic Pipeline Evaluation Report

            **Evaluation Timestamp:** {results['timestamp']}
            **Total Queries Tested:** {results['total_queries']}

            ## Performance Metrics

            - **Average Latency:** {metrics['average_latency']:.3f} seconds
            - **Median Latency:** {metrics['median_latency']:.3f} seconds
            - **Max Latency:** {metrics['max_latency']:.3f} seconds
            - **Min Latency:** {metrics['min_latency']:.3f} seconds
            - **Success Rate:** {metrics['success_rate']:.1%}

            ## Decision Distribution

            - **Knowledge Base Usage:** {metrics['kb_usage_rate']:.1%}
            - **Web Search Usage:** {metrics['web_search_rate']:.1%}

            ## Query Results Summary

            | Query | Decision | Latency (s) | Success |
            |-------|----------|-------------|---------|
            """
        
        for result in results['query_results'][:10]:  # First 10 queries
            decision = result.get('decision', {}).get('decision', 'unknown')
            latency = result.get('latency', 0)
            success = '✓' if result.get('success', False) else '✗'
            query_short = result['query'][:50] + '...' if len(result['query']) > 50 else result['query']
            
            report += f"| {query_short} | {decision} | {latency:.3f} | {success} |\n"
        
        return report