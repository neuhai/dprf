"""
PersonaRefiner class for the Generalized DPRF framework.
Implements the two-step persona refinement process.
"""

from typing import Dict, List, Any, Optional, Tuple, Callable
import re
import os
import logging
import platform
import time
import random
import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer  
import torch  
            
# Try to import vLLM, but don't fail if it's not available
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

# Check if running on Apple Silicon
IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine().startswith("arm")

import openai

# Try to import llama.cpp integration
try:
    from .llama_cpp_integration import LlamaCppModel, SamplingParams as LlamaCppSamplingParams, is_llama_cpp_supported
    LLAMA_CPP_AVAILABLE = is_llama_cpp_supported()
except ImportError:
    LLAMA_CPP_AVAILABLE = False

# Import the utility functions from local utils.py
from .utils import format_analysis_prompt, format_refinement_prompt
from .token_usage import (
    TokenUsageTracker,
    record_bedrock_usage,
    record_openai_usage,
)

def is_likely_valid_api_key(key):
    """Check if a string looks like a valid OpenAI API key format"""
    if not key:
        return False
        
    # Accept both standard and project API keys
    if key.startswith("sk-proj-"):
        # Project API keys are valid
        return True
        
    # Traditional OpenAI keys usually start with 'sk-' and are 51 characters
    valid_format = re.match(r'^sk-[A-Za-z0-9]{48}$', key)
    return bool(valid_format)


def normalize_openai_base_url(base_url: Optional[str]) -> Optional[str]:
    """Normalize OpenAI-compatible base URL (strip trailing /responses)."""
    if not base_url:
        return None
    normalized = base_url.strip().rstrip("/")
    if normalized.endswith("/responses"):
        normalized = normalized[: -len("/responses")]
    return normalized or None


def is_gpt_5_family(model_name: str) -> bool:
    return model_name.lower().startswith("gpt-5")


GPT5_MIN_COMPLETION_TOKENS = 8000


def openai_max_completion_tokens(model_name: str, max_tokens: int) -> int:
    if is_gpt_5_family(model_name):
        return max(max_tokens, GPT5_MIN_COMPLETION_TOKENS)
    return max_tokens

try:
    import boto3
    from botocore.config import Config
    BEDROCK_AVAILABLE = True
except ImportError:
    BEDROCK_AVAILABLE = False

try:
    import aioboto3
    import asyncio
    BEDROCK_ASYNC_AVAILABLE = True
except ImportError:
    BEDROCK_ASYNC_AVAILABLE = False

class PersonaRefiner:
    """
    Class for refining personas based on comparison between generated responses and ground truth.
    Uses a two-step process: first analyze differences, then refine based on analysis.
    
    Supports OpenAI API, vLLM, and llama.cpp as backends.
    """
    
    def __init__(
        self, 
        model_name: str = "gpt-4o",
        model_type: str = "openai",
        openai_api_key: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        bedrock_region_name: str = 'us-east-1', 
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the persona refiner.
        
        Args:
            model_name: Name of the model to use for persona refinement
            model_type: Type of model to use ('vllm', 'openai', 'llama.cpp')
            openai_api_key: OpenAI API key (defaults to environment variable if None)
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling threshold
            model_kwargs: Additional arguments for model initialization
        """
        # Set up logging
        self.logger = logging.getLogger("DPRF.PersonaRefiner")
        
        self.model_name = model_name
        self.model_type = model_type
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.bedrock_region_name = bedrock_region_name 
        self.model_kwargs = model_kwargs or {}
        self.token_usage = self.model_kwargs.get("shared_token_usage")
        if self.token_usage is None:
            self.token_usage = TokenUsageTracker()
            self.model_kwargs["shared_token_usage"] = self.token_usage
        self.openai_base_url = normalize_openai_base_url(
            os.environ.get("OPENAI_BASE_URL") or os.environ.get("AZURE_OPENAI_BASE_URL")
        )
        
        # Use provided API key or the environment variable
        provided_api_key = openai_api_key or openai.api_key or os.environ.get("OPENAI_API_KEY", "")
        
        # Validate the API key
        if model_type == "openai" and not self.openai_base_url and not is_likely_valid_api_key(provided_api_key):
            self.logger.error("The provided OpenAI API key appears to be invalid.")
            raise ValueError("Invalid OpenAI API key format. Please provide a valid API key.")
            
        actual_api_key = provided_api_key
        
        # Determine if we're running on Apple Silicon and log it
        if IS_APPLE_SILICON:
            self.logger.info("Running on Apple Silicon (M-series chip).")
        
        # Setup the appropriate model
        if model_type == "vllm":
            self._setup_vllm_model(model_name)
        elif model_type == "sglang":
            self._setup_sglang_model(model_name)
        elif model_type == "openai":
            self._setup_openai_model(model_name, actual_api_key)
        elif model_type in ["hf", "hf_8bit"]:  
            # Log HuggingFace model initialization  
            self.model_name = model_name  
            self.model_type = model_type  
            self.max_tokens = max_tokens  
            self.temperature = temperature  
            self.top_p = top_p  
            self.model_kwargs = model_kwargs or {}  
            # Do not load the model immediately, load on first use  
            self.hf_model = None  
            self.hf_tokenizer = None  
        elif model_type == "bedrock":
            self._setup_bedrock_model(model_name, bedrock_region_name)
        else:  
            raise ValueError(f"Unknown model type: {model_type}")  


    def _setup_bedrock_model(self, model_id: str, region_name: str):
        """Set up AWS Bedrock client."""
        if 'shared_bedrock_client' in self.model_kwargs and 'shared_bedrock_model_id' in self.model_kwargs:
            self.logger.info(f"Using shared Bedrock session for model {model_id}")
            # Handle both sync and async clients
            shared_client = self.model_kwargs['shared_bedrock_client']
            if hasattr(shared_client, 'client'):  # aioboto3.Session
                self.bedrock_session = shared_client
                self.bedrock_config = self.model_kwargs.get('shared_bedrock_config', None)
                self.is_async_bedrock = True
            else:  # boto3.client
                self.bedrock_runtime = shared_client
                self.is_async_bedrock = False
            self.bedrock_model_id = self.model_kwargs['shared_bedrock_model_id']
            return
            
        # Try async first, fallback to sync
        if BEDROCK_ASYNC_AVAILABLE:
            try:
                self.logger.info(f"Initializing AWS Bedrock async session for model {model_id} in region {region_name}")
                from botocore.config import Config
                boto_config = Config(
                    region_name=region_name,
                    retries={
                        'max_attempts': self.model_kwargs.get('bedrock_max_attempts', 200), 
                        'mode': 'adaptive'
                    }
                )
                self.bedrock_session = aioboto3.Session()
                self.bedrock_config = boto_config
                self.bedrock_model_id = model_id
                self.is_async_bedrock = True
                self.logger.info(f"Successfully initialized AWS Bedrock async session for model {self.bedrock_model_id}")
                return
            except Exception as e:
                self.logger.warning(f"Failed to initialize async Bedrock session: {e}, falling back to sync")
        
        # Fallback to sync bedrock
        if not BEDROCK_AVAILABLE:
            self.logger.error("Neither aioboto3 nor boto3 are installed. Please install one to use Bedrock models.")
            raise ImportError("Bedrock dependencies are required for Bedrock model type but not found.")

        try:
            self.logger.info(f"Initializing AWS Bedrock sync client for model {model_id} in region {region_name}")
            from botocore.config import Config
            boto_config = Config(
                region_name=region_name,
                retries={
                    'max_attempts': self.model_kwargs.get('bedrock_max_attempts', 200), 
                    'mode': 'adaptive'
                }
            )
            self.bedrock_runtime = boto3.client('bedrock-runtime', config=boto_config)
            self.bedrock_model_id = model_id
            self.is_async_bedrock = False
            self.logger.info(f"Successfully initialized AWS Bedrock sync client for model {self.bedrock_model_id}")
        except Exception as e:
            self.logger.error(f"Error initializing AWS Bedrock client: {e}")
            raise RuntimeError(f"Failed to initialize Bedrock client: {e}")
    
    def _setup_openai_model(self, model_name: str, api_key: str):
        """Set up OpenAI async client."""
        self.logger.info(f"Using OpenAI API for refiner model {model_name}")
        client_kwargs = {"api_key": api_key}
        if self.openai_base_url:
            client_kwargs["base_url"] = self.openai_base_url
            self.logger.info(f"Using custom OpenAI-compatible base URL: {self.openai_base_url}")
        self.openai_client = openai.AsyncClient(**client_kwargs)

    def _setup_vllm_model(self, model_name: str):
        """Set up vLLM model, preferring shared instances."""
        if 'shared_vllm_model' in self.model_kwargs and 'shared_vllm_sampling_params' in self.model_kwargs:
            self.logger.info(f"Using shared vLLM model instance for refiner: {model_name}")
            self.model = self.model_kwargs['shared_vllm_model']
            self.sampling_params = self.model_kwargs['shared_vllm_sampling_params']
            return

        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not installed, and no shared instance was provided.")

        self.logger.warning(f"No shared vLLM model instance found for {model_name}. Initializing a new one for the refiner. This may lead to VRAM issues if a task model is already loaded.")
        
        try:
            refiner_model_kwargs = self.model_kwargs.copy()
            if 'gpu_memory_utilization' not in refiner_model_kwargs:
                refiner_model_kwargs['gpu_memory_utilization'] = 0.45

            # Current vLLM version's EngineArgs may not support trust_remote_code parameter, so actively remove it
            if 'trust_remote_code' in refiner_model_kwargs:
                self.logger.warning("Detected 'trust_remote_code' in refiner_model_kwargs but current vLLM version does not support it; ignoring the parameter.")
                refiner_model_kwargs.pop('trust_remote_code')

            common_kwargs = {
                'model': model_name,
                'tensor_parallel_size': 1,  # Default to 1 GPU for safety
                **refiner_model_kwargs
            }
            
            self.model = LLM(**common_kwargs)
            
            is_qwen_model = 'qwen' in model_name.lower()
            is_deepseek_model = 'deepseek' in model_name.lower()
            
            stop_tokens = []
            if is_qwen_model:
                stop_tokens = ["<|im_end|>"]
            elif not is_deepseek_model:
                stop_tokens = ["</s>"]

            self.sampling_params = SamplingParams(
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                stop=stop_tokens
            )
            self.logger.info(f"Successfully initialized a new vLLM refiner model instance.")
        except Exception as e:
            self.logger.error(f"Error initializing new vLLM instance for refiner: {e}")
            raise

    def _setup_sglang_model(self, model_name: str):
        """Set up SGLang client for SGLang models from a shared client."""
        if 'shared_sglang_client' in self.model_kwargs and 'shared_sglang_sampling_params' in self.model_kwargs:
            self.logger.info(f"Using shared SGLang client for refiner model {model_name}")
            self.sglang_client = self.model_kwargs['shared_sglang_client']
            self.sglang_sampling_params = self.model_kwargs['shared_sglang_sampling_params']
            
            # Configure retry attempts for PersonaRefiner SGLang API calls
            self.sglang_max_attempts = self.model_kwargs.get('sglang_max_attempts', 200)
            self.logger.info(f"PersonaRefiner SGLang max retry attempts set to {self.sglang_max_attempts}")
            return

        self.logger.warning(f"SGLang refiner model {model_name} requested but no shared client instance provided. Refinement will likely fail.")

    def format_qwen_messages(self, messages):  
        """  
        Format messages into ChatML format required by Qwen models  
        
        Args:  
            messages: List of messages [{"role": "...", "content": "..."}, ...]  
            
        Returns:  
            Formatted text  
        """  
        formatted_text = ""  
        for message in messages:  
            role = message["role"]  
            content = message["content"]  
            
            if role == "system":  
                formatted_text += f"<|im_start|>system\n{content}<|im_end|>\n"  
            elif role == "user":  
                formatted_text += f"<|im_start|>user\n{content}<|im_end|>\n"  
            elif role == "assistant":  
                formatted_text += f"<|im_start|>assistant\n{content}<|im_end|>\n"  
            else:  
                # Handle other roles  
                formatted_text += f"<|im_start|>{role}\n{content}<|im_end|>\n"  
        
        # Add assistant marker to prompt model to generate a reply, but do not add end marker  
        formatted_text += "<|im_start|>assistant\n"  
        
        return formatted_text  


    
    async def refine_persona(
        self, 
        persona: str, 
        content: str, 
        generated_response: str, 
        ground_truth: str,
        analysis_formatter: Optional[Callable[[str, str, str, str], str]] = None,
        refinement_formatter: Optional[Callable[[str, str], str]] = None
    ) -> Tuple[str, str]:
        """
        Refine the persona based on differences between generated response and ground truth.
        This is a two-step process:
        1. Analyze differences between responses
        2. Refine the persona based on analysis
        
        Args:
            persona: Current description of the persona
            content: Text content processed
            generated_response: Response generated by the agent
            ground_truth: Ground truth response
            
        Returns:
            Tuple of (refined persona, analysis)
        """
        # Step 1: Generate analysis of differences
        analysis = await self._generate_analysis(
            persona=persona,
            content=content,
            generated_response=generated_response,
            ground_truth=ground_truth,
            analysis_formatter=analysis_formatter
        )
        
        # Step 2: Generate refined persona based on analysis
        refined_persona = await self._generate_refined_persona(
            persona=persona,
            analysis=analysis,
            refinement_formatter=refinement_formatter
        )
        
        return refined_persona, analysis
    
    async def _generate_analysis(
        self,
        persona: str,
        content: str,
        generated_response: str,
        ground_truth: str,
        analysis_formatter: Optional[Callable[[str, str, str, str], str]] = None
    ) -> str:
        print(f"DEBUG: _generate_analysis called with analysis_formatter = {analysis_formatter}")
        print(f"DEBUG: analysis_formatter is not None: {analysis_formatter is not None}")
        
        # Use custom formatter if provided
        if analysis_formatter:
            print("DEBUG: Using custom analysis_formatter")
            formatted_prompt = analysis_formatter(
                persona,
                content,
                generated_response,
                ground_truth
            )
            return await self._generate_text(formatted_prompt)
        
        # Fallback: Check if a custom formatter has been injected as class member (legacy)
        elif hasattr(self, '_custom_analysis_formatter'):
            print("DEBUG: Using legacy _custom_analysis_formatter")
            formatted_prompt = self._custom_analysis_formatter(
                persona=persona,
                content=content,
                generated_response=generated_response,
                ground_truth=ground_truth
            )
            return await self._generate_text(formatted_prompt)
        
        # Use the default formatter
        print("DEBUG: Using default analysis formatter")
        analysis_prompt = format_analysis_prompt(
            persona=persona,
            content=content,
            generated_response=generated_response,
            ground_truth=ground_truth
        )
        
        analysis = await self._generate_text(analysis_prompt)
        analysis_tokens = self.count_tokens(analysis)
        # print("step 2-analysis_tokens: ", analysis_tokens)
        return analysis
    
    async def _generate_refined_persona(
        self, 
        persona: str, 
        analysis: str,
        refinement_formatter: Optional[Callable[[str, str], str]] = None
    ) -> str:
        """
        Generate refined persona based on analysis.
        
        Args:
            persona: Current description of the persona
            analysis: Analysis of differences
            refinement_formatter: Optional custom formatter for refinement prompt
            
        Returns:
            Refined persona
        """
        # Use custom formatter if provided
        if refinement_formatter:
            formatted_prompt = refinement_formatter(persona, analysis)
            refined_persona_text = await self._generate_text(formatted_prompt)
            refined_persona = self._extract_refined_persona(refined_persona_text)
            return refined_persona if refined_persona else persona
        
        # Fallback: Check if a custom formatter has been injected as class member (legacy)
        elif hasattr(self, '_custom_refinement_formatter'):
            formatted_prompt = self._custom_refinement_formatter(
                persona=persona,
                analysis=analysis
            )
            refined_persona_text = await self._generate_text(formatted_prompt)
            refined_persona = self._extract_refined_persona(refined_persona_text)
            return refined_persona if refined_persona else persona
            
        # Use the default formatter
        refinement_prompt = format_refinement_prompt(
            persona=persona,
            analysis=analysis
        )
        refinement_tokens = self.count_tokens(refinement_prompt)
        # print("step 3-refinement_input_tokens: ", refinement_tokens)

        refined_persona_text = await self._generate_text(refinement_prompt)
        refined_persona_tokens = self.count_tokens(refined_persona_text)
        # print("step 3-refined_persona_output_tokens: ", refined_persona_tokens)
        
        # Extract the persona from the response
        refined_persona = self._extract_refined_persona(refined_persona_text)
        
        # If extraction failed, return the original persona
        if not refined_persona:
            return persona
        
        return refined_persona
        
    def _post_process_deepseek_response(self, text: str) -> str:  
        """Post-process the response generated by the DeepSeek model to remove the thinking prefix"""  
        # Try to find the </think> marker  
        think_end_marker = "</think>"  
        think_end_pos = text.find(think_end_marker)  
        
        if think_end_pos != -1:  
            # Marker found, extract content after it and remove possible leading whitespace  
            content_start = think_end_pos + len(think_end_marker)  
            return text[content_start:].lstrip()  
        
        # If no </think>, keep the original text  
        return text  


    async def _generate_text(self, prompt: str) -> str:  
        if self.model_type == "vllm":
            import asyncio
            is_deepseek_model = 'deepseek' in self.model_name.lower()
            
            # vLLM's generate is synchronous, run in a thread to avoid blocking asyncio event loop
            outputs = await asyncio.to_thread(self.model.generate, [prompt], self.sampling_params)
            
            response_text = outputs[0].outputs[0].text.strip()
            
            if is_deepseek_model:
                response_text = self._post_process_deepseek_response(response_text)
                
            return response_text

        elif self.model_type == "sglang":
            is_deepseek_model = 'deepseek' in self.model_name.lower()
            
            if not hasattr(self, 'sglang_client'):
                 self.logger.error("SGLang client not initialized. Make sure a shared client is passed.")
                 return "Error: SGLang client not available."

            messages = [
                {"role": "system", "content": "You are an AI assistant helping with persona refinement."},
                {"role": "user", "content": prompt}
            ]
            
            # Use retry mechanism for SGLang API calls
            response_text = await self._call_sglang_with_retry(messages)
            # Token usage recorded inside _call_sglang_with_retry when response is returned
            
            if is_deepseek_model:
                response_text = self._post_process_deepseek_response(response_text)
                
            return response_text
        
  
        elif self.model_type == "openai":  
            request_kwargs = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": openai_max_completion_tokens(self.model_name, self.max_tokens),
            }
            if not is_gpt_5_family(self.model_name):
                request_kwargs["temperature"] = self.temperature
                request_kwargs["top_p"] = self.top_p
            response = await self.openai_client.chat.completions.create(**request_kwargs)
            record_openai_usage(self.token_usage, response, source="refiner_openai")
            content = response.choices[0].message.content
            if content is None:
                content = ""
            return content.strip()
        elif self.model_type in ["hf", "hf_8bit"]:  
            # Import torch at the method level  
            if 'shared_hf_model' in self.model_kwargs and 'shared_hf_tokenizer' in self.model_kwargs:
                self.hf_model = self.model_kwargs['shared_hf_model']
                self.hf_tokenizer = self.model_kwargs['shared_hf_tokenizer']
                self.logger.info(f"Using shared HuggingFace model instance for {self.model_name}")
            elif not hasattr(self, 'hf_model') or self.hf_model is None:
                try:  
                    print(f"Loading HuggingFace model: {self.model_name}")  
                    tokenizer = AutoTokenizer.from_pretrained(self.model_name)  
                      
                    if self.model_type == "hf_8bit":  
                        model = AutoModelForCausalLM.from_pretrained(  
                            self.model_name,  
                            device_map="auto",  
                            load_in_8bit=True,  
                            trust_remote_code=True  
                        )  
                    else:  
                        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32  
                        
                        model = AutoModelForCausalLM.from_pretrained(  
                            self.model_name,  
                            device_map="auto",  
                            torch_dtype=dtype,  
                            trust_remote_code=True  
                        )  
                    
                    self.hf_model = model  
                    self.hf_tokenizer = tokenizer  
                    print(f"HuggingFace model {self.model_name} loaded successfully")  
                except Exception as e:  
                    print(f"Error loading HuggingFace model: {e}")  
                    return "Error generating response due to model loading failure."  
            
            try:  
                inputs = self.hf_tokenizer(prompt, return_tensors="pt").to(self.hf_model.device)  
                
                generation_kwargs = {  
                    "max_new_tokens": self.max_tokens,  
                    "temperature": self.temperature,  
                    "top_p": self.top_p,  
                    "do_sample": True,  
                    "repetition_penalty": 1.1,  
                    "pad_token_id": self.hf_tokenizer.eos_token_id  
                }  
                
                with torch.no_grad():  
                    outputs = self.hf_model.generate(**inputs, **generation_kwargs)  
                
                response = self.hf_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)  
                
                return response  
            except Exception as e:  
                print(f"Error generating with HuggingFace model: {e}")  
                return "Error generating response."  
            
        elif self.model_type == "bedrock":
            messages = [{"role": "user", "content": prompt}]
            
            # If the prompt itself is in a multi-turn dialogue format (uncommon in this function), more complex parsing is needed
            # Here it is assumed that the prompt structure of PersonaRefiner is relatively simple

            try:
                # Convert message format
                transformed_messages = [self._transform_bedrock_message(msg) for msg in messages]
                
                # Extract Claude-specific parameters from model_kwargs, or use default values
                inference_config = {
                    "maxTokens": self.model_kwargs.get('claude_max_tokens', self.max_tokens),
                    "temperature": self.model_kwargs.get('claude_temperature', self.temperature),
                    "topP": self.model_kwargs.get('claude_top_p', self.top_p),
                    # "stopSequences": self.model_kwargs.get('claude_stop_sequences', []) # Claude's stop sequences
                }
                api_messages = transformed_messages
                system_prompt_text = self.model_kwargs.get('bedrock_system_prompt', None) # Allow passing system prompts through model_kwargs

                request_body = {
                    "modelId": self.bedrock_model_id,
                    "messages": api_messages,
                    "inferenceConfig": inference_config
                }
                if system_prompt_text:
                    request_body["system"] = [{"text": system_prompt_text}]

                # Support both async and sync bedrock calls
                if getattr(self, 'is_async_bedrock', False):
                    response = await self._call_bedrock_async(**request_body)
                else:
                    response = self.bedrock_runtime.converse(**request_body)
                    record_bedrock_usage(self.token_usage, response, source="refiner_bedrock")
                
                # Extract generated text from the response
                # Claude's Converse API response structure is usually response['output']['message']['content'][0]['text']
                if response and 'output' in response and 'message' in response['output'] and \
                'content' in response['output']['message'] and \
                isinstance(response['output']['message']['content'], list) and \
                len(response['output']['message']['content']) > 0 and \
                'text' in response['output']['message']['content'][0]:
                    return response['output']['message']['content'][0]['text'].strip()
                else:
                    self.logger.error(f"Unexpected Bedrock response format: {response}")
                    return "Error: Could not parse Bedrock response."
            except Exception as e:
                self.logger.error(f"Error calling Bedrock model {self.bedrock_model_id}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                return f"Error generating response from Bedrock: {e}"
        else:
            raise ValueError(f"Unknown model type for generation: {self.model_type}")

    async def _call_bedrock_async(self, **request_body) -> Dict[str, Any]:
        """
        Asynchronously call AWS Bedrock Converse API.
        
        Args:
            **request_body: Request body for Bedrock API
            
        Returns:
            Response from Bedrock API
        """
        if not hasattr(self, 'bedrock_session'):
            raise ValueError("Async bedrock session not initialized")
            
        async with self.bedrock_session.client('bedrock-runtime', config=self.bedrock_config) as bedrock_runtime:
            response = await bedrock_runtime.converse(**request_body)
            record_bedrock_usage(self.token_usage, response, source="refiner_bedrock")
            return response

    def _transform_bedrock_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform message format from 'content': 'xxx' to 'content': [{'text': 'xxx'}]
        for Bedrock Converse API.
        """
        transformed_msg = message.copy() # Avoid modifying the original message
        if 'content' in transformed_msg and isinstance(transformed_msg['content'], str):
            transformed_msg['content'] = [{'text': transformed_msg['content']}]
        # If content is already in the format required by Bedrock, do not convert
        elif 'content' in transformed_msg and isinstance(transformed_msg['content'], list) and \
            all(isinstance(item, dict) and 'text' in item for item in transformed_msg['content']):
            pass # Already in Bedrock format
        else:
            # For other complex formats, more detailed processing or error throwing may be needed
            self.logger.warning(f"Message content format for Bedrock might be incorrect: {transformed_msg.get('content')}")
        return transformed_msg

    async def _call_sglang_with_retry(self, messages: List[Dict[str, Any]], max_attempts: Optional[int] = None) -> str:
        """
        Call SGLang API with automatic retry and exponential backoff for PersonaRefiner.
        
        Args:
            messages: List of messages for the conversation
            max_attempts: Maximum number of retry attempts (uses configured value if None)
            
        Returns:
            Response text from SGLang API
            
        Raises:
            Exception: If all retry attempts fail
        """
        # Use configured retry attempts if not specified
        if max_attempts is None:
            max_attempts = getattr(self, 'sglang_max_attempts', 200)
        
        last_exception = None
        
        for attempt in range(1, max_attempts + 1):
            try:
                response = await self.sglang_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    stop=getattr(self.sglang_sampling_params, 'stop', [])
                )
                record_openai_usage(self.token_usage, response, source="refiner_sglang")
                
                # If successful, return the response text
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"PersonaRefiner SGLang API call attempt {attempt}/{max_attempts} failed: {e}")
                
                # If this is the last attempt, don't wait
                if attempt == max_attempts:
                    break
                
                # Calculate exponential backoff with jitter
                base_delay = min(2 ** (attempt - 1), 60)  # Cap at 60 seconds
                jitter = random.uniform(0.1, 0.5)  # Add 10-50% jitter
                delay = base_delay + jitter
                
                self.logger.info(f"Retrying PersonaRefiner SGLang API call in {delay:.2f} seconds...")
                await asyncio.sleep(delay)
        
        # If we get here, all attempts failed
        error_msg = f"PersonaRefiner SGLang API call failed after {max_attempts} attempts. Last error: {last_exception}"
        self.logger.error(error_msg)
        raise Exception(error_msg)
        
    def _extract_refined_persona(self, text: str) -> str:
        """
        Extract the refined persona from the text.
        
        Args:
            text: Text to extract persona from
            
        Returns:
            Extracted persona
        """
        # Try to find a section labeled "REFINED PERSONA" or similar
        match = re.search(r"(?:REFINED PERSONA|PERSONA):(.*?)(?:\Z|EXPLANATION:)", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # If not found, just return the whole text as the persona
        return text.strip() 