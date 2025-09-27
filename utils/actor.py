import os
import requests
from tavily import TavilyClient
from typing import List, Dict, Any, Optional, Tuple
import logging
import time

class WebSearchActor:
    def __init__(self, tavily_api_key: str):
        self.tavily_client = TavilyClient(api_key=tavily_api_key)
        self.logger = logging.getLogger(__name__)
        
    def search_web(self, query: str, max_results: int = 5) -> Tuple[List[Dict], float]:
        start_time = time.time()
        try:
            # Perform search
            response = self.tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                include_answer=True,
                include_raw_content=False
            )
            
            latency = time.time() - start_time
            
            # Process results
            results = []
            if 'results' in response:
                for result in response['results']:
                    results.append({
                        'title': result.get('title', ''),
                        'url': result.get('url', ''),
                        'content': result.get('content', ''),
                        'score': result.get('score', 0.0),
                        'published_date': result.get('published_date', '')
                    })
            
            self.logger.info(f"Web search completed in {latency:.2f}s, found {len(results)} results")
            return results, latency
            
        except Exception as e:
            latency = time.time() - start_time
            self.logger.error(f"Web search failed after {latency:.2f}s: {e}")
            return [], latency
    
    def get_search_stats(self) -> Dict:
        return {
            'service': 'Tavily',
            'features': ['advanced_search', 'real_time', 'content_extraction']
        }