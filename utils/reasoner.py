import os
import json
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime
import requests

class LLMReasoner:
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key
        self.model = model
        self.logger = logging.getLogger(__name__)
        
        # Initialize client with simple approach
        self.client = self._initialize_client()
        
        # Test the client
        try:
            self._test_client()
            self.logger.info(f"LLM client initialized successfully with model: {model}")
        except Exception as e:
            self.logger.error(f"Client test failed: {e}")
            raise
        
        # Versioned prompt templates
        self.prompt_templates = {
            "v1": {
                "system": """You are an intelligent assistant that helps users find information. You have access to a knowledge base and web search capabilities.

                Your job is to:
                1. Analyze the user's query
                2. Determine if the knowledge base results are sufficient to answer the query
                3. Decide whether to use knowledge base information or search the web
                4. Provide a clear, helpful answer

                Guidelines:
                - If knowledge base results have good similarity (>0.7) and contain relevant info, use them
                - If knowledge base results are insufficient or outdated, recommend web search
                - Always be honest about the source of your information
                - Provide specific, actionable answers when possible""",
                
            "decision": """Based on the user query and knowledge base search results, decide the next action:

                User Query: {query}

                Knowledge Base Results:
                {kb_results}

                Maximum Similarity Score: {max_similarity}

                Decision Options:
                1. "use_kb" - if knowledge base results are sufficient and relevant
                2. "web_search" - if need fresh information or KB results are insufficient

                Respond with JSON format:
                {{
                    "decision": "use_kb" or "web_search",
                    "reasoning": "explanation of why this decision was made",
                    "confidence": 0.0-1.0
                }}""",
                
            "answer_kb": """Answer the user's query using the provided knowledge base information.

                User Query: {query}

                Relevant Information from Knowledge Base:
                {kb_results}

                Instructions:   
                - Provide a comprehensive answer based on the knowledge base
                - Include specific details like pricing, features, etc. when available
                - If information is partial, mention what's covered and what might need additional research
                - Be clear and concise""",
                
            "answer_web": """Answer the user's query using the web search results.

                User Query: {query}

                Web Search Results:
                {web_results}

                Instructions:
                - Synthesize information from web search results
                - Provide current, accurate information
                - Cite sources when relevant
                - If results are insufficient, say so clearly"""
            }
        }
        
        self.current_version = "v1"
    
    def _initialize_client(self):
        
        # Method 1: Try importing and using Groq normally
        try:
            from groq import Groq
            return Groq(api_key=self.api_key)
        except Exception as e1:
            self.logger.warning(f"Standard Groq initialization failed: {e1}")
            
            # Method 2: Try direct HTTP client
            try:
                return self._create_http_client()
            except Exception as e2:
                self.logger.error(f"HTTP client creation failed: {e2}")
                raise Exception(f"All client initialization methods failed. Groq: {e1}, HTTP: {e2}")
    
    def _create_http_client(self):
        
        class SimpleGroqClient:
            def __init__(self, api_key: str, model: str):
                self.api_key = api_key
                self.model = model
                self.base_url = "https://api.groq.com/openai/v1"
                
                # Test API key immediately
                self._test_api_key()
            
            def _test_api_key(self):
                """Test if API key is valid"""
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                test_data = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 1
                }
                
                try:
                    response = requests.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=test_data,
                        timeout=10
                    )
                    
                    if response.status_code == 401:
                        raise Exception("Invalid API key")
                    elif response.status_code == 429:
                        raise Exception("Rate limit exceeded")
                    elif response.status_code >= 400:
                        raise Exception(f"API error: {response.status_code} - {response.text}")
                    
                except requests.exceptions.RequestException as e:
                    raise Exception(f"Network error: {e}")
            
            def create_chat_completion(self, messages: List[Dict], max_tokens: int = 1000, temperature: float = 0.3) -> str:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
                
                try:
                    response = requests.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=data,
                        timeout=30
                    )
                    
                    if response.status_code != 200:
                        raise Exception(f"API call failed: {response.status_code} - {response.text}")
                    
                    result = response.json()
                    return result['choices'][0]['message']['content']
                    
                except requests.exceptions.RequestException as e:
                    raise Exception(f"Network error during API call: {e}")
                except KeyError as e:
                    raise Exception(f"Unexpected API response format: {e}")
        
        return SimpleGroqClient(self.api_key, self.model)
    
    def _test_client(self):
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'OK' if you can hear me."}
        ]
        
        try:
            response = self._call_llm(test_messages, max_tokens=10)
            if response and len(response.strip()) > 0:
                self.logger.info("Client test successful")
            else:
                raise Exception("Empty response from test")
        except Exception as e:
            raise Exception(f"Client test failed: {e}")
    
    def _call_llm(self, messages: List[Dict], max_tokens: int = 1000) -> str:
        #Api call 
        try:
            # Handle different client types
            if hasattr(self.client, 'chat') and hasattr(self.client.chat, 'completions'):
                # Standard Groq SDK client
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.3
                )
                return response.choices[0].message.content.strip()
                
            elif hasattr(self.client, 'create_chat_completion'):
                # Custom HTTP client
                return self.client.create_chat_completion(messages, max_tokens, 0.3)
            else:
                raise Exception("Unknown client type")
            
        except Exception as e:
            self.logger.error(f"LLM API call failed: {e}")
            return f"I apologize, but I'm having trouble processing your request due to a technical issue: {str(e)}. Please try again."
    
    def decide_action(self, query: str, kb_results: List[Dict], max_similarity: float, 
                     threshold: float = 0.7) -> Dict:
        #Decide web or vdb
        try:
            # Format KB results for prompt
            kb_text = ""
            if kb_results:
                for i, result in enumerate(kb_results[:3]):  # Top 3 results
                    kb_text += f"Result {i+1} (similarity: {result['similarity']:.3f}):\n"
                    content_preview = result['content'][:300] if len(result['content']) > 300 else result['content']
                    kb_text += f"{content_preview}\n\n"
            else:
                kb_text = "No relevant results found in knowledge base."
            
            # Create decision prompt
            prompt = self.prompt_templates[self.current_version]["decision"].format(
                query=query,
                kb_results=kb_text,
                max_similarity=max_similarity
            )
            
            messages = [
                {"role": "system", "content": self.prompt_templates[self.current_version]["system"]},
                {"role": "user", "content": prompt}
            ]
            
            response = self._call_llm(messages, max_tokens=300)
            
            # Try to parse JSON response
            try:
                # Clean response 
                response_clean = response.strip()
                if "```json" in response_clean:
                    response_clean = response_clean.split("```json")[1].split("```")[0].strip()
                elif "```" in response_clean:
                    lines = response_clean.split("```")
                    if len(lines) >= 2:
                        response_clean = lines[1].strip()
                
                # Try to find JSON in the response
                if not response_clean.startswith('{'):
                    # Look for JSON-like structure
                    start = response_clean.find('{')
                    end = response_clean.rfind('}')
                    if start != -1 and end != -1:
                        response_clean = response_clean[start:end+1]
                
                decision_data = json.loads(response_clean)
                
                # Validate the decision
                if decision_data.get("decision") not in ["use_kb", "web_search"]:
                    raise ValueError("Invalid decision value")
                
                return decision_data
                
            except (json.JSONDecodeError, ValueError) as e:
                self.logger.warning(f"Could not parse LLM decision response: {e}. Response: {response[:100]}...")
                # Fallback decision logic
                if max_similarity > threshold and kb_results:
                    return {
                        "decision": "use_kb",
                        "reasoning": f"Knowledge base has good similarity ({max_similarity:.3f}) above threshold ({threshold})",
                        "confidence": max_similarity
                    }
                else:
                    return {
                        "decision": "web_search",
                        "reasoning": f"Knowledge base similarity ({max_similarity:.3f}) below threshold ({threshold}) or no results",
                        "confidence": 0.5
                    }
            
        except Exception as e:
            self.logger.error(f"Error in decision making: {e}")
            # Fallback to simple threshold-based decision
            if max_similarity > threshold and kb_results:
                return {
                    "decision": "use_kb",
                    "reasoning": f"Fallback decision: KB similarity ({max_similarity:.3f}) > threshold ({threshold})",
                    "confidence": max_similarity
                }
            else:
                return {
                    "decision": "web_search",
                    "reasoning": f"Fallback decision: Need web search (similarity: {max_similarity:.3f})",
                    "confidence": 0.5
                }
    
    def generate_answer_from_kb(self, query: str, kb_results: List[Dict]) -> str:
        try:
            # Format KB results
            kb_text = ""
            for i, result in enumerate(kb_results[:5]):
                source_info = result.get('metadata', {}).get('source', f'Document {i+1}')
                similarity = result.get('similarity', 0.0)
                kb_text += f"Source {i+1} ({source_info}, similarity: {similarity:.3f}):\n"
                kb_text += f"{result['content']}\n\n"
            
            prompt = self.prompt_templates[self.current_version]["answer_kb"].format(
                query=query,
                kb_results=kb_text
            )
            
            messages = [
                {"role": "system", "content": self.prompt_templates[self.current_version]["system"]},
                {"role": "user", "content": prompt}
            ]
            
            return self._call_llm(messages, max_tokens=800)
            
        except Exception as e:
            self.logger.error(f"Error generating KB answer: {e}")
            # Fallback to simple response
            if kb_results:
                return f"Based on our knowledge base regarding '{query}': {kb_results[0]['content'][:300]}..."
            else:
                return f"I don't have specific information about '{query}' in my knowledge base."
    
    def generate_answer_from_web(self, query: str, web_results: List[Dict]) -> str:
        try:
            # Format web results
            web_text = ""
            for i, result in enumerate(web_results[:5]):
                web_text += f"Source {i+1}: {result.get('title', 'N/A')}\n"
                web_text += f"URL: {result.get('url', 'N/A')}\n"
                content = result.get('content', '')
                content_preview = content[:300] if len(content) > 300 else content
                web_text += f"Content: {content_preview}\n\n"
            
            prompt = self.prompt_templates[self.current_version]["answer_web"].format(
                query=query,
                web_results=web_text
            )
            
            messages = [
                {"role": "system", "content": self.prompt_templates[self.current_version]["system"]},
                {"role": "user", "content": prompt}
            ]
            
            return self._call_llm(messages, max_tokens=800)
            
        except Exception as e:
            self.logger.error(f"Error generating web answer: {e}")
            # Fallback to simple response
            if web_results:
                return f"Based on current web search for '{query}': {web_results[0].get('content', 'Information found but could not be processed')[:200]}..."
            else:
                return f"I searched the web for '{query}' but couldn't find relevant information at this time."
    
    def set_prompt_version(self, version: str):
        if version in self.prompt_templates:
            self.current_version = version
            self.logger.info(f"Switched to prompt version: {version}")
        else:
            self.logger.warning(f"Unknown prompt version: {version}")
    
    def get_available_versions(self) -> List[str]:
        return list(self.prompt_templates.keys())
    
    def get_client_info(self) -> Dict:
        return {
            "model": self.model,
            "client_type": type(self.client).__name__,
            "api_key_set": bool(self.api_key and len(self.api_key) > 10),
            "prompt_version": self.current_version
        }
