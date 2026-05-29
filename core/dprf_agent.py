"""  
DPRFAgent class for the Generalized DPRF framework.  
Handles running of Dynamic Persona Refinement iterations.  
"""  

from typing import Dict, List, Any, Optional, Tuple, Union, Callable  
import re  
import os  
import json  
import logging  
import datetime  
import platform  
import asyncio
import time
import random
import aioboto3
from botocore.config import Config
import tiktoken
import atexit  
# Optional SGLang utilities (only needed for task/refiner type "sglang")
try:
    from sglang.utils import (
        launch_server_cmd as sglang_launch_server_cmd,
        wait_for_server as sglang_wait_for_server,
        terminate_process as sglang_terminate_process,
    )
    SGLANG_AVAILABLE = True
except ImportError:
    sglang_launch_server_cmd = None
    sglang_wait_for_server = None
    sglang_terminate_process = None
    SGLANG_AVAILABLE = False

def _safe_terminate_process(process):
    """Gracefully terminate a subprocess even if SGLang helpers are unavailable."""
    if process is None:
        return
    try:
        if sglang_terminate_process:
            sglang_terminate_process(process)
        else:
            process.terminate()
            try:
                process.wait(timeout=10)
            except Exception:
                process.kill()
    except Exception as termination_error:
        logging.getLogger("DPRF.Agent").warning(f"Failed to terminate process cleanly: {termination_error}")
# Check if running on Apple Silicon  
IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine().startswith("arm")  

import openai  

# Try to import llama.cpp integration  
try:  
    from .llama_cpp_integration import LlamaCppModel, SamplingParams as LlamaCppSamplingParams, is_llama_cpp_supported  
    LLAMA_CPP_AVAILABLE = is_llama_cpp_supported()  
except ImportError:  
    LLAMA_CPP_AVAILABLE = False  

# Try to import vLLM
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

# Import from local modules  
from .persona_refinement import PersonaRefiner  
from .utils import (  
    ensure_directory,   
    load_json,   
    save_json,  
    format_persona_prompt,  
    format_peer_review_instruction  
)
from .token_usage import (
    TokenUsageTracker,
    record_bedrock_usage,
    record_estimated_usage,
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
    """GPT-5 models may spend the entire budget on reasoning tokens; reserve headroom for content."""
    if is_gpt_5_family(model_name):
        return max(max_tokens, GPT5_MIN_COMPLETION_TOKENS)
    return max_tokens

try:
    import aioboto3
    import asyncio
    from botocore.config import Config
    BEDROCK_AVAILABLE_AGENT = True 
except ImportError:
    BEDROCK_AVAILABLE_AGENT = False

# Create a simple parameter class for SGLang compatibility
class SGLangParams:
    """Simple parameter class for SGLang API calls"""
    def __init__(self, max_tokens=2500, temperature=0.7, top_p=0.9, stop=None):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.stop = stop or []
    
    def to_dict(self):
        """Convert parameters to dictionary for compatibility"""
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop": self.stop
        }

class DPRFAgent:  
    """  
    Dynamic Persona Refinement Framework (DPRF) Agent.  
    
    Uses a two-stage process to iteratively refine a persona:  
    1. Generate a response using the current persona  
    2. Compare the response to ground truth and refine the persona  
    
    Supports OpenAI API, vLLM, and llama.cpp for task model and refinement.  
    """  
    
    def __init__(  
        self,  
        task_model_name: str = "gpt-4o",  
        task_model_type: str = "openai",  
        refiner_model_name: str = "gpt-4o",  
        refiner_model_type: str = "openai",  
        max_iterations: int = 5,  
        openai_api_key: Optional[str] = None,  
        max_tokens: int = 2500,  
        temperature: float = 0.7,  
        top_p: float = 0.9,  
        save_logs: bool = False,  
        log_dir: str = "logs",  
        model_kwargs: Optional[Dict[str, Any]] = None ,
        bedrock_region_name: str = 'us-east-1' 
    ):  
        """  
        Initialize the DPRF agent.  
        
        Args:  
            task_model_name: Name of the model for task performance  
            task_model_type: Type of model for task ('openai', 'vllm', 'llama.cpp')  
            refiner_model_name: Name of the model for persona refinement  
            refiner_model_type: Type of model for refiner ('openai', 'vllm', 'llama.cpp')  
            max_iterations: Maximum number of iterations for refinement  
            openai_api_key: OpenAI API key (defaults to env var if None)  
            max_tokens: Maximum tokens to generate per response  
            temperature: Sampling temperature for generation  
            top_p: Top-p sampling threshold for generation  
            save_logs: Whether to save logs of the refinement process  
            log_dir: Directory to save logs (relative to cwd)  
            model_kwargs: Optional additional arguments for model initialization  
        """  
        # Set up logging  
        self.logger = logging.getLogger("DPRF.Agent")  
        
        # Store parameters  
        self.task_model_name = task_model_name  
        self.task_model_type = task_model_type  
        self.refiner_model_name = refiner_model_name
        self.refiner_model_type = refiner_model_type
        self.max_iterations = max_iterations  
        self.max_tokens = max_tokens  
        self.temperature = temperature  
        self.top_p = top_p  
        self.save_logs = save_logs  
        self.log_dir = log_dir  
        self.model_kwargs = model_kwargs or {}
        self.token_usage = self.model_kwargs.get("shared_token_usage")
        if self.token_usage is None:
            self.token_usage = TokenUsageTracker()
        self.model_kwargs["shared_token_usage"] = self.token_usage
        self.bedrock_region_name = bedrock_region_name 
        
        # Initialize model dictionaries  
        self.hf_models = {}  # Add HuggingFace model dictionary  
        
        # Initialize process tracking for cleanup
        self.server_process = None
        self.port = None
        
        # Ensure task_model attribute exists even if initialization fails
        self.task_model = None  # Will be overwritten by specific model setup if successful
        
        # Initialize tiktoken encoding for token counting
        self.tiktoken_encoding = None
        try:
            import tiktoken
            self.tiktoken_encoding = tiktoken.encoding_for_model(task_model_name)
        except Exception as e:
            self.logger.warning(f"Could not initialize tiktoken encoding for {task_model_name}: {e}")
            # Try to use GPT-4o encoding as fallback for unknown models
            try:
                import tiktoken
                self.tiktoken_encoding = tiktoken.encoding_for_model("gpt-4o")
                self.logger.info(f"Using GPT-4o tokenizer as fallback for {task_model_name}")
            except Exception as e2:
                self.logger.warning(f"Could not initialize fallback GPT-4o tokenizer: {e2}")
                self.tiktoken_encoding = None
        
        # Use provided API key or the environment variable  
        provided_api_key = openai_api_key or openai.api_key or os.environ.get("OPENAI_API_KEY", "")  
        
        self.openai_base_url = normalize_openai_base_url(
            os.environ.get("OPENAI_BASE_URL") or os.environ.get("AZURE_OPENAI_BASE_URL")
        )

        # Validate the API key if using OpenAI.
        # Skip strict key-format validation for custom OpenAI-compatible endpoints (e.g. Azure).
        if (
            (task_model_type == "openai" or refiner_model_type == "openai")
            and not self.openai_base_url
            and not is_likely_valid_api_key(provided_api_key)
        ):
            self.logger.error("The provided OpenAI API key appears to be invalid.")  
            raise ValueError("Invalid OpenAI API key format. Please provide a valid API key.")  
            
        self.openai_api_key = provided_api_key  
        
        # Determine if we're running on Apple Silicon and log it  
        if IS_APPLE_SILICON:  
            self.logger.info("Running on Apple Silicon (M-series chip).")  
        
        # Set up task model  
        if task_model_type == "vllm":
            self._setup_vllm_task_model(task_model_name)
        elif task_model_type == "sglang":
            self._setup_sglang_task_model(task_model_name, "")
        elif task_model_type == "openai":  
            self._setup_openai_task_model(task_model_name, self.openai_api_key)  
        elif task_model_type in ["hf", "hf_8bit"]:  
            # No need to initialize imme
            # tely, will be loaded on first use  
            self.logger.info(f"Will use Hugging Face model for task: {task_model_name}")  
        elif task_model_type == "bedrock":
            self._setup_bedrock_task_model(task_model_name, bedrock_region_name)
        else:  
            raise ValueError(f"Unknown task model type: {task_model_type}")  
        
        if (task_model_type == refiner_model_type and task_model_name == refiner_model_name):
            self.logger.info(f"Task and refiner models are identical ({task_model_name}), sharing model instance")
            refiner_model_kwargs = self.model_kwargs.copy()
            
            # Handle different model types for sharing
            if task_model_type == "sglang":
                refiner_model_kwargs['shared_sglang_client'] = self.sglang_client
                refiner_model_kwargs['shared_sglang_sampling_params'] = self.task_sampling_params
            elif task_model_type == "vllm":
                # Only share the vLLM model if initialization succeeded
                if self.task_model is not None:
                    refiner_model_kwargs['shared_vllm_model'] = self.task_model
                    refiner_model_kwargs['shared_vllm_sampling_params'] = self.task_sampling_params
                else:
                    self.logger.warning("vLLM task model was not initialized successfully; skipping model sharing with refiner.")
            elif task_model_type == "openai":
                # For OpenAI models, we don't need to share a client since it's API-based
                # Just pass the model type information
                pass
            elif task_model_type == "llama.cpp":
                refiner_model_kwargs['shared_llama_cpp_model'] = self.task_model
                refiner_model_kwargs['shared_llama_cpp_sampling_params'] = self.task_sampling_params
            elif task_model_type in ["hf", "hf_8bit"] and task_model_name in self.hf_models:
                refiner_model_kwargs['shared_hf_model'] = self.hf_models[task_model_name]["model"] 
                refiner_model_kwargs['shared_hf_tokenizer'] = self.hf_models[task_model_name]["tokenizer"]
            elif task_model_type == "bedrock":
                refiner_model_kwargs['shared_bedrock_client'] = self.bedrock_session
                refiner_model_kwargs['shared_bedrock_model_id'] = self.bedrock_task_model_id
                refiner_model_kwargs['shared_bedrock_config'] = self.bedrock_config
        else:
            self.logger.info(f"Using separate model instances for task ({task_model_name}) and refiner ({refiner_model_name})")
            refiner_model_kwargs = self.model_kwargs.copy()

        refiner_model_kwargs["shared_token_usage"] = self.token_usage
            
        # Add vLLM specific parameters if needed
        if refiner_model_type == "vllm":  
            if 'gpu_memory_utilization' not in refiner_model_kwargs:  
                refiner_model_kwargs['gpu_memory_utilization'] = 0.45  

        # Initialize the persona refiner  
        self.persona_refiner = PersonaRefiner(  
            model_name=refiner_model_name,  
            model_type=refiner_model_type,  
            openai_api_key=self.openai_api_key,  
            max_tokens=max_tokens,  
            temperature=temperature,  
            top_p=top_p,  
            model_kwargs=refiner_model_kwargs,  # Use refiner_model_kwargs instead of self.model_kwargs
            bedrock_region_name=bedrock_region_name 
        )  
        
        # Create log directory if saving logs  
        if save_logs:  
            ensure_directory(log_dir)  

        # Initialize AWS Bedrock if needed
        if self.task_model_type == "bedrock":
            self._setup_bedrock_task_model(task_model_name, bedrock_region_name)
        
        # Add SGLang concurrency control
        if self.task_model_type == "sglang":
            # Set concurrency: 3 requests per GPU (optimized for H200)
            gpu_count = self._get_gpu_count()
            max_concurrent = gpu_count * 3  # 3 concurrent requests per GPU
            self.sglang_semaphore = asyncio.Semaphore(max_concurrent)
            self.logger.info(f"SGLang concurrency limit set to {max_concurrent} ({gpu_count} GPUs × 3 requests/GPU)")
        else:
            self.sglang_semaphore = None

        # Register cleanup function to run when program exits
        atexit.register(self.cleanup)

    def cleanup(self):

        if hasattr(self, 'server_process') and self.server_process is not None:
            try:
                self.logger.info(f"Terminating SGLang server process (PID: {self.server_process.pid})")
                _safe_terminate_process(self.server_process)
                self.logger.info("SGLang server process terminated successfully")
                self.server_process = None
                self.port = None
            except Exception as e:
                self.logger.error(f"Error terminating SGLang server process: {e}")
        else:
            self.logger.debug("No SGLang server process to clean up")

    def __del__(self):
        self.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def _setup_bedrock_task_model(self, model_id: str, region_name: str):
        """Set up AWS Bedrock client with support for asynchronous concurrent calls."""
        if not BEDROCK_AVAILABLE_AGENT: 
            self.logger.error("Boto3 is not installed. Please install 'aioboto3' to use Bedrock models.")
            raise ImportError("Using Bedrock task model requires aioboto3 but it was not found.")

        try:
            self.logger.info(f"Initializing AWS Bedrock asynchronous client, model {model_id}, region {region_name}")
            boto_config = Config(
                region_name=region_name,
                retries={
                    'max_attempts': self.model_kwargs.get('bedrock_max_attempts', 200),
                    'mode': 'adaptive'
                }
            )
            # Create an asynchronous session instead of direct client
            self.bedrock_session = aioboto3.Session()
            self.bedrock_task_model_id = model_id
            self.bedrock_config = boto_config
            
            # Create a semaphore to control concurrency
            self.bedrock_semaphore = asyncio.Semaphore(
                self.model_kwargs.get('bedrock_max_concurrency', 20)
            )
            
            self.logger.info(f"Successfully initialized AWS Bedrock asynchronous client, model ID {self.bedrock_task_model_id}")
        except Exception as e:
            self.logger.error(f"Error initializing AWS Bedrock asynchronous client: {e}")
            raise RuntimeError(f"Failed to initialize Bedrock asynchronous client: {e}")

    def _transform_bedrock_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Transforms message for Bedrock Converse API (similar to PersonaRefiner's method)."""
        transformed_msg = message.copy()
        if 'content' in transformed_msg and isinstance(transformed_msg['content'], str):
            transformed_msg['content'] = [{'text': transformed_msg['content']}]
        return transformed_msg
        
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
        
    def format_llama_messages(self, messages):  
        """  
        Format messages into the format required by Llama models  
        
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
                formatted_text += f"<|system|>\n{content}\n"  
            elif role == "user":  
                formatted_text += f"<|user|>\n{content}\n"  
            elif role == "assistant":  
                formatted_text += f"<|assistant|>\n{content}\n"  
            else:  
                # Handle other roles  
                formatted_text += f"<|{role}|>\n{content}\n"  
        
        # Add final assistant marker to prompt model to generate a reply  
        formatted_text += "<|assistant|>\n"  
        
        return formatted_text  

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
    
    async def generate_with_model(self, messages, model_name, model_type):  
        """Generate text using the specified model."""  
        if model_type == "openai":  
            # Existing OpenAI code  
            request_kwargs = {
                "model": model_name,
                "messages": messages,
                "max_completion_tokens": openai_max_completion_tokens(self.task_model_name, self.max_tokens),
            }
            if not is_gpt_5_family(model_name):
                request_kwargs["temperature"] = self.temperature
                request_kwargs["top_p"] = self.top_p
            response = await openai.AsyncClient().chat.completions.create(**request_kwargs)
            record_openai_usage(self.token_usage, response, source="task_openai")
            return response.choices[0].message.content.strip()
        
        elif model_type in ["hf", "hf_8bit"]:  
            # Check if model is already loaded  
            if model_name not in self.hf_models:  
                try:  
                    from transformers import AutoModelForCausalLM, AutoTokenizer  
                    import torch  
                    
                    print(f"Loading HuggingFace model: {model_name}")  
                    tokenizer = AutoTokenizer.from_pretrained(model_name)  
                    
                    # Add support for 8-bit quantization  
                    if model_type == "hf_8bit":  
                        model = AutoModelForCausalLM.from_pretrained(  
                            model_name,   
                            device_map="auto",  
                            load_in_8bit=True,  
                            trust_remote_code=True  
                        )  
                    else:  
                        # Use torch.bfloat16 or torch.float16 to reduce memory usage  
                        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32  
                        
                        model = AutoModelForCausalLM.from_pretrained(  
                            model_name,   
                            device_map="auto",  
                            torch_dtype=dtype,  
                            trust_remote_code=True  
                        )  
                    
                    self.hf_models[model_name] = {  
                        "model": model,  
                        "tokenizer": tokenizer  
                    }  
                    print(f"HuggingFace model {model_name} loaded successfully")  
                except Exception as e:  
                    print(f"Error loading HuggingFace model: {e}")  
                    return "Error generating response due to model loading failure."  
            
            model_data = self.hf_models[model_name]  
            model = model_data["model"]  
            tokenizer = model_data["tokenizer"]  
            
            try:  
                # Format messages into Llama format  
                formatted_input = self.format_llama_messages(messages)  
                
                # Set generation parameters  
                generation_kwargs = {  
                    "max_new_tokens": self.max_tokens,  
                    "temperature": self.temperature,  
                    "top_p": self.top_p,  
                    "do_sample": True,  
                    "repetition_penalty": 1.1,  
                    "pad_token_id": tokenizer.eos_token_id  
                }  
                
                # Encode input  
                inputs = tokenizer(formatted_input, return_tensors="pt").to(model.device)  
                
                # Generate output  
                with torch.no_grad():  
                    outputs = model.generate(**inputs, **generation_kwargs)  
                
                # Decode output, skip input part  
                response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)  
                
                # Clean up response, remove possible residual markers  
                if "<|assistant|>" in response:  
                    response_parts = response.split("<|assistant|>")  
                    if len(response_parts) > 1:  
                        response = response_parts[-1].strip()  
                
                return response  
                
            except Exception as e:  
                print(f"Error generating with HuggingFace model: {e}")  
                return "Error generating response."  
                
        else:  
            # Other model types  
            return "Model type not supported for direct generation."  
    
    def _get_gpu_count(self) -> int:
        """Get the number of available GPUs dynamically."""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                self.logger.info(f"Detected {gpu_count} GPU(s) using torch.cuda")
                return gpu_count
            else:
                self.logger.info("CUDA not available, using 1 as fallback")
                return 1
        except ImportError:
            # Fallback to nvidia-smi if torch is not available
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                                      capture_output=True, text=True, check=True)
                gpu_count = len([line for line in result.stdout.strip().split('\n') if line.strip()])
                self.logger.info(f"Detected {gpu_count} GPU(s) using nvidia-smi")
                return gpu_count
            except (subprocess.CalledProcessError, FileNotFoundError):
                self.logger.warning("Could not detect GPU count, using 1 as fallback")
                return 1

    def _setup_openai_task_model(self, task_model_name: str, actual_api_key: str):
        """Set up OpenAI task model."""
        self.logger.info("Using OpenAI API directly")
        client_kwargs = {"api_key": actual_api_key}
        if self.openai_base_url:
            client_kwargs["base_url"] = self.openai_base_url
            self.logger.info(f"Using custom OpenAI-compatible base URL: {self.openai_base_url}")
        self.openai_client = openai.AsyncClient(**client_kwargs)
        self.stop_tokens = []
        self.logger.info("OpenAI API client initialized")

    def _setup_llama_cpp_task_model(self, model_name: str):
        """Set up Llama.cpp task model."""
        if not LLAMA_CPP_AVAILABLE:
            raise RuntimeError("Llama.cpp is not available on this system.")
        
        from .llama_cpp_integration import LlamaCppModel, SamplingParams as LlamaCppSamplingParams
        
        self.logger.info(f"Initializing llama.cpp model: {model_name}")
        
        # Default parameters for llama.cpp, can be overridden by model_kwargs
        llama_cpp_params = {
            "model_path": model_name,
            "n_gpu_layers": -1 if not IS_APPLE_SILICON else 1, # Use Metal on Apple Silicon
            "n_ctx": self.model_kwargs.get("n_ctx", 4096),
            **self.model_kwargs
        }
        
        self.task_model = LlamaCppModel(**llama_cpp_params)
        
        self.task_sampling_params = LlamaCppSamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=[] # Llama.cpp handles stopping differently, can be set here if needed
        )
        self.logger.info("Llama.cpp model initialized successfully.")

    def _setup_vllm_task_model(self, task_model_name: str):
        """Set up vLLM task model with appropriate configurations."""
        if not VLLM_AVAILABLE:
            self.logger.error("vLLM is not installed. Please install it with 'pip install vllm' or use a different model type.")
            self.logger.info("Automatically falling back to OpenAI API mode for task model...")
            self.task_model_type = "openai"
            return self._setup_openai_task_model(task_model_name, self.openai_api_key)
        
        try:
            # Safely get the number of GPUs
            try:
                requested_gpus = int(os.environ.get("SLURM_GPUS_ON_NODE", 1))
                self.logger.info(f"Setting tensor_parallel_size to {requested_gpus} based on requested GPUs.")
            except (ValueError, TypeError):
                requested_gpus = 1
                self.logger.info(f"Could not determine GPU count from environment. Using default: {requested_gpus}")
            
            # Prepare vLLM configuration
            task_model_kwargs = self.model_kwargs.copy()
            if 'gpu_memory_utilization' not in task_model_kwargs:
                task_model_kwargs['gpu_memory_utilization'] = 0.45
            
            # Remove trust_remote_code parameter for vLLM compatibility
            if 'trust_remote_code' in task_model_kwargs:
                self.logger.warning("Detected 'trust_remote_code' in model_kwargs but current vLLM version does not support it; ignoring the parameter.")
                task_model_kwargs.pop('trust_remote_code')

            # Ensure required parameters are in kwargs
            common_kwargs = {
                'model': task_model_name,
                'tensor_parallel_size': requested_gpus,
                **task_model_kwargs
            }
            
            # Choose different initialization strategies based on platform
            if IS_APPLE_SILICON:
                # Set environment variable for Apple Silicon
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
                self.logger.info("Configuring vLLM with MPS support for Apple Silicon.")
                
                # Try to detect MPS availability
                try:
                    import torch
                    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        self.logger.info("MPS is available. Using MPS device where supported by vLLM.")
                    else:
                        self.logger.info("MPS not available. Falling back to CPU.")
                except Exception as mps_e:
                    self.logger.warning(f"Error checking MPS availability: {mps_e}")
                
                # Use the same parameters for initialization regardless of MPS availability (vLLM will handle device selection automatically)
                self.task_model = LLM(**common_kwargs)
                
            else:
                # Some versions of vLLM do not accept the 'max_model_len' argument.
                # Attempt initialization without it to maintain compatibility.
                self.task_model = LLM(**common_kwargs)
            
            # Set model-specific stop tokens
            is_qwen_model = 'qwen' in task_model_name.lower()
            is_deepseek_model = 'deepseek' in task_model_name.lower()
            is_llama_3_2_model = 'llama3.2' in task_model_name.lower() or 'llama-3.2' in task_model_name.lower() or 'llama-3-2' in task_model_name.lower()
            
            stop_tokens = []
            if is_qwen_model:
                stop_tokens = ["<|im_end|>"]
                self.logger.info("Using Qwen-specific stop tokens: <|im_end|>")
            elif is_deepseek_model:
                # DeepSeek models do not require special stop tokens as we will post-process outputs
                # Add other special requirements here if needed
                self.logger.info("Using DeepSeek model, will post-process outputs")
            elif is_llama_3_2_model:
                stop_tokens = ["---", "</s>", 
                    "Please let me know if",
                    "Please let me know"]
                self.logger.info(f"Using Llama 3.2 specific stop tokens: {stop_tokens}")
            else:
                # Default stop tokens for other models
                stop_tokens = ["</s>"]
                self.logger.info("Using default stop token: </s>")
            
            # Create sampling parameters
            self.task_sampling_params = SamplingParams(
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                stop=stop_tokens
            )
            
            # Create separate stop tokens for DeepSeek analysis refinement
            if is_deepseek_model:
                refinement_stop_tokens = stop_tokens.copy()
                refinement_stop_tokens.extend(["\nANALYSIS OF DIFFERENCES:", "\n\nANALYSIS OF DIFFERENCES:"])
                self.logger.info("Created DeepSeek-specific refinement stop tokens")
                
                # Save to instance variable for later use
                self.refinement_sampling_params = SamplingParams(
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    stop=refinement_stop_tokens
                )
            
            self.logger.info(f"Successfully initialized vLLM with model {task_model_name} using tensor_parallel_size={requested_gpus}")
            
        except Exception as e:
            self.logger.error(f"Error initializing vLLM for task model: {e}")
            
            # Fallback strategy
            if IS_APPLE_SILICON and LLAMA_CPP_AVAILABLE:
                self.logger.info("Falling back to llama.cpp (recommended for Apple Silicon)...")
                self.task_model_type = "llama.cpp"
                return self._setup_llama_cpp_task_model(task_model_name)
            else:
                self.logger.info("Falling back to OpenAI API mode for task model...")
                self.task_model_type = "openai"
                return self._setup_openai_task_model(task_model_name, self.openai_api_key)

    def _setup_sglang_task_model(self, task_model_name: str, actual_api_key: str):  
        """Set up SGLang API service for SGLang models."""  
        if not SGLANG_AVAILABLE:
            raise ImportError(
                "SGLang support is requested but the 'sglang' package is not installed. "
                "Install sglang (and sgl-kernel if running locally) to use task_model_type='sglang'."
            )
        try:  
            self.logger.info(f"Initializing SGLang API service for model {task_model_name}")  
            
            # Get GPU count dynamically
            gpu_count = self._get_gpu_count()
            
            # Get context length from model_kwargs, default to a reasonable value for memory efficiency
            context_length = self.model_kwargs.get('context_length', 20000)  # Much smaller default
            mem_fraction = self.model_kwargs.get('mem_fraction_static', 0.8)  # Reduce memory fraction
            
            # Construct SGLang server command with memory-efficient settings
            server_cmd = (
                f"uv run --active python -m sglang.launch_server "
                f"--model-path {task_model_name} "
                f"--data-parallel-size {gpu_count} "
                f"--context-length {context_length} "
                f"--mem-fraction-static {mem_fraction}"
            )
            
            self.logger.info(f"Starting SGLang server with context_length={context_length}, mem_fraction={mem_fraction}")
            
            self.server_process, self.port = sglang_launch_server_cmd(server_cmd)  
            sglang_wait_for_server(f"http://localhost:{self.port}")  
            self.logger.info(f"SGLang server started on port {self.port} with {gpu_count} GPU(s)")  
            
            # Configure retry attempts for SGLang API calls
            self.sglang_max_attempts = self.model_kwargs.get('sglang_max_attempts', 200)
            self.logger.info(f"SGLang max retry attempts set to {self.sglang_max_attempts}")
            
            # Configure timeout settings for SGLang client - 1 hour timeout with limited concurrency
            import httpx
            timeout_config = httpx.Timeout(
                connect=600.0,    # 10 minutes to connect
                read=3600.0,     # 1 hour read timeout
                write=600.0,      # 10 minutes write timeout
                pool=600.0         # 10 minutes pool timeout
            )
            
            # Create SGLang client without connection limits - use Semaphore for concurrency control
            self.sglang_client = openai.AsyncClient(
                base_url=f"http://127.0.0.1:{self.port}/v1", 
                api_key="None",
                timeout=3600.0,  # 1 hour total timeout
                http_client=httpx.AsyncClient(
                    timeout=timeout_config
                )
            )  
            
            is_qwen_model = 'qwen' in task_model_name.lower()  
            is_deepseek_model = 'deepseek' in task_model_name.lower()  

            self.stop_tokens = []  
            if is_qwen_model:  
                self.stop_tokens = ["<|im_end|>"]  
            elif is_deepseek_model:  
                self.stop_tokens = ["Please let me know if",  "Please let me know"]  
            else:  
                self.stop_tokens = ["</s>"]  
            
            self.logger.info(f"SGLang API client initialized with stop tokens: {self.stop_tokens}")  
            self.task_sampling_params = SGLangParams(  
                max_tokens=self.max_tokens,  
                temperature=self.temperature,  
                top_p=self.top_p,  
                stop=self.stop_tokens  # Add appropriate stop tokens  
            )  
            
        except Exception as e:  
            self.logger.error(f"Error initializing SGLang API service: {e}")
            # Clean up started processes if initialization fails
            if hasattr(self, 'server_process') and self.server_process is not None:
                try:
                    self.logger.info("Cleaning up SGLang process due to initialization failure")
                    _safe_terminate_process(self.server_process)
                    self.server_process = None
                    self.port = None
                except Exception as cleanup_error:
                    self.logger.error(f"Error cleaning up SGLang process: {cleanup_error}")
            raise  # Re-raise exception to terminate program

    def count_tokens(self, text: str) -> int:
        """
        Counts tokens in the provided text, primarily using tiktoken for OpenAI models.
        Includes fallback mechanisms for other tokenizer types or approximations.

        Args:
            text: The text for which to count tokens.

        Returns:
            The number of tokens in the text.
        """
        try:
            # 1. Try using pre-initialized tiktoken encoding (most efficient for OpenAI models)
            if self.tiktoken_encoding:
                try:
                    return len(self.tiktoken_encoding.encode(text))
                except Exception as e:
                    self.logger.warning(f"Error using pre-initialized tiktoken encoding: {e}. Trying dynamic lookup.")

            try:
                # self.task_model_name should reflect the current model intended for tokenization.
                encoding = tiktoken.encoding_for_model(self.task_model_name)
                return len(encoding.encode(text))
            except Exception as e_tiktoken:
                self.logger.warning(
                    f"Dynamic tiktoken encoding for '{self.task_model_name}' failed: {e_tiktoken}. "
                    f"Attempting fallback to GPT-4o tokenizer."
                )
                # Try GPT-4o as fallback
                try:
                    encoding = tiktoken.encoding_for_model("gpt-4o")
                    return len(encoding.encode(text))
                except Exception as e_gpt4o:
                    self.logger.warning(f"GPT-4o fallback tokenizer also failed: {e_gpt4o}. Trying other methods.")

            # 3. Fallback: Try using self.fallback_tokenizer
            if hasattr(self, 'fallback_tokenizer') and self.fallback_tokenizer is not None:
                try:
                    self.logger.info("Falling back to self.fallback_tokenizer (e.g., generic Hugging Face).")
                    return len(self.fallback_tokenizer.encode(text))
                except Exception as e_fallback_custom:
                    self.logger.warning(f"Error with self.fallback_tokenizer: {e_fallback_custom}")

            if hasattr(self, 'task_model_type') and self.task_model_type in ["hf", "hf_8bit"] and \
            hasattr(self, 'task_model_name') and self.task_model_name in getattr(self, 'hf_models', {}):
                
                model_info = self.hf_models.get(self.task_model_name)
                if model_info and "tokenizer" in model_info:
                    hf_tokenizer_instance = model_info["tokenizer"]
                    if hf_tokenizer_instance: # Ensure the tokenizer instance exists
                        try:
                            self.logger.info(f"Falling back to specific Hugging Face tokenizer for: {self.task_model_name}.")
                            return len(hf_tokenizer_instance.encode(text))
                        except Exception as e_hf_specific:
                            self.logger.warning(f"Error with specific HF tokenizer '{self.task_model_name}': {e_hf_specific}")
                    else:
                        self.logger.warning(f"HF tokenizer instance for '{self.task_model_name}' is None.") # Log if tokenizer is None
                        
        except Exception as e:
            self.logger.warning(f"All token counting methods failed or were not applicable for '{self.task_model_name}'. Falling back to approximate character count (length / 4).")
            return len(text) // 4

    
    async def generate_response(  
        self,  
        persona: str,  
        content: str,   
        custom_formatter: Optional[Callable] = None,  
        max_input_tokens: int = 20000,  # Default token limit   
        max_output_tokens: Optional[int] = 2500
    ) -> str:  
        """  
        Generate a response for the given content and instruction using the persona.  
        
        Args:  
            persona: Description of the persona  
            content: Text content to process  
            instruction: Task instruction   
            custom_formatter: Optional custom formatter for the prompt  
            max_input_tokens: Maximum number of tokens allowed in input  
            
        Returns:  
            Generated response  
        """  
        # Check if a custom formatter has been provided for the persona prompt  
        if custom_formatter:  
            prompt = custom_formatter(  
                persona=persona,  
                content=content,  
            )  
        else:  
            # Use the default formatter  
            prompt = format_persona_prompt(  
                persona=persona,  
                content=content,  
            )  
        
        # Final safety check to warn if prompt exceeds token limit but don't truncate
        final_token_count = self.count_tokens(prompt)
        # print("step 1-token count: ", final_token_count)
        if final_token_count > max_input_tokens:  
            self.logger.warning(f"WARNING: Final prompt exceeds token limit. Token count: {final_token_count} > {max_input_tokens}. Prompt will NOT be truncated.")  
  
        self.logger.info(f"Prompt sent to model ({self.task_model_type}):\n{prompt}\n")

        # Generate a response using the appropriate model
        if self.task_model_type == "vllm":
            is_ds_model = 'deepseek' in self.task_model_name.lower()
            
            # vLLM is synchronous, so we run it in a separate thread
            outputs = await asyncio.to_thread(
                self.task_model.generate, [prompt], self.task_sampling_params
            )
            
            response_text = outputs[0].outputs[0].text.strip()
            
            if is_ds_model:
                response_text = self._post_process_deepseek_response(response_text)
                
            return response_text

        elif self.task_model_type == "sglang":
            # Check if using DeepSeek model for post-processing
            is_ds_model = 'deepseek' in self.task_model_name.lower()
            
            messages = [
                {"role": "system", "content": f"You are an AI assistant generating response according to given persona"},
                {"role": "user", "content": f"{prompt}"}
            ]
            
            # Use retry mechanism for SGLang API calls
            response_text = await self._call_sglang_with_retry(
                messages=messages,
                max_output_tokens=max_output_tokens or self.max_tokens
            )
            
            if is_ds_model:
                response_text = self._post_process_deepseek_response(response_text)
            
            return response_text

        elif self.task_model_type == "openai":
            request_kwargs = {
                "model": self.task_model_name,
                "messages": [
                    {"role": "system", "content": f"You are an AI assistant generating response according to given persona"},
                    {"role": "user", "content": f"{prompt}"}
                ],
                "max_completion_tokens": openai_max_completion_tokens(
                    self.task_model_name, max_output_tokens or self.max_tokens
                ),
            }
            if not is_gpt_5_family(self.task_model_name):
                request_kwargs["temperature"] = self.temperature
                request_kwargs["top_p"] = self.top_p
            response = await self.openai_client.chat.completions.create(**request_kwargs)
            record_openai_usage(self.token_usage, response, source="task_openai")
            content = response.choices[0].message.content
            if content is None:
                content = ""
            return content.strip()
            
        elif self.task_model_type in ["hf", "hf_8bit"]:  
            # Use Hugging Face model  
            return await self.generate_with_model(  
            messages = [
                {"role": "system", "content": f"You are an AI assistant generating response according to given persona"},
                {"role": "user", "content": f"{prompt}"}
            ],
                model_name=self.task_model_name,  
                model_type=self.task_model_type  
            )  
            
        elif self.task_model_type == "bedrock":
            system_prompt_text = "You are an AI assistant generating response according to given persona"

            user_message_content = f"{prompt}"

            messages_for_bedrock = []
            final_system_prompt = self.model_kwargs.get('bedrock_system_prompt', system_prompt_text)

            # User message
            messages_for_bedrock.append({"role": "user", "content": user_message_content})

            try:
                inference_config = {
                    "maxTokens": max_output_tokens or self.max_tokens,
                    "temperature": self.model_kwargs.get('claude_temperature', self.temperature),
                    "topP": self.model_kwargs.get('claude_top_p', self.top_p),
                }

                # Call Bedrock asynchronously
                response = await self._call_bedrock_async(
                    messages=messages_for_bedrock, 
                    system_prompt=final_system_prompt,
                    inference_config=inference_config
                )

                if response and 'output' in response and 'message' in response['output'] and \
                'content' in response['output']['message'] and \
                isinstance(response['output']['message']['content'], list) and \
                len(response['output']['message']['content']) > 0 and \
                'text' in response['output']['message']['content'][0]:
                    return response['output']['message']['content'][0]['text'].strip()
                else:
                    self.logger.error(f"Unexpected Bedrock response format: {response}")
                    return "Error: Could not parse Bedrock task model response."
            except Exception as e:
                self.logger.error(f"Error calling Bedrock task model {self.bedrock_task_model_id}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                return f"Error generating response from Bedrock task model: {e}"

    async def _call_bedrock_async(self, messages: List[Dict[str, Any]], system_prompt: str, inference_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronously call AWS Bedrock Converse API with semaphore-based concurrency control.
        
        Args:
            messages: List of messages for the conversation
            system_prompt: System prompt text
            inference_config: Configuration for inference (maxTokens, temperature, etc.)
            
        Returns:
            Response from Bedrock API
        """
        async with self.bedrock_semaphore:
            async with self.bedrock_session.client('bedrock-runtime', config=self.bedrock_config) as bedrock_runtime:
                # Transform messages for Bedrock format
                transformed_messages = [self._transform_bedrock_message(msg) for msg in messages]
                
                # Prepare the request
                request_body = {
                    "modelId": self.bedrock_task_model_id,
                    "messages": transformed_messages,
                    "system": [{"text": system_prompt}],
                    "inferenceConfig": inference_config
                }
                
                # Make the API call
                response = await bedrock_runtime.converse(**request_body)
                record_bedrock_usage(self.token_usage, response, source="task_bedrock")
                return response

    async def _call_sglang_with_retry(self, messages: List[Dict[str, Any]], max_output_tokens: int, max_attempts: Optional[int] = None) -> str:
        # Use configured retry attempts if not specified
        if max_attempts is None:
            max_attempts = getattr(self, 'sglang_max_attempts', 200)
        
        last_exception = None
        
        for attempt in range(1, max_attempts + 1):
            try:
                # Use semaphore for concurrency control if available
                if self.sglang_semaphore:
                    async with self.sglang_semaphore:
                        response = await self.sglang_client.chat.completions.create(
                            model=self.task_model_name,
                            messages=messages,
                            max_tokens=max_output_tokens,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            stop=self.task_sampling_params.stop
                        )
                else:
                    response = await self.sglang_client.chat.completions.create(
                        model=self.task_model_name,
                        messages=messages,
                        max_tokens=max_output_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        stop=self.task_sampling_params.stop
                    )
                record_openai_usage(self.token_usage, response, source="task_sglang")
                
                # If successful, return the response text
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"SGLang API call attempt {attempt}/{max_attempts} failed: {e}")
                
                # If this is the last attempt, don't wait
                if attempt == max_attempts:
                    break
                
                # Calculate exponential backoff with jitter
                base_delay = min(2 ** (attempt - 1), 60)  # Cap at 60 seconds
                jitter = random.uniform(0.1, 0.5)  # Add 10-50% jitter
                delay = base_delay + jitter
                
                self.logger.info(f"Retrying SGLang API call in {delay:.2f} seconds...")
                await asyncio.sleep(delay)
        
        # If we get here, all attempts failed
        error_msg = f"SGLang API call failed after {max_attempts} attempts. Last error: {last_exception}"
        self.logger.error(error_msg)
        raise Exception(error_msg)
    
    async def generate_with_refiner(self, prompt):  
        """  
        Generate a text response using persona_refiner  
        
        Args:  
            prompt: Prompt text  
            
        Returns:  
            Generated response text  
        """   
        return await self.persona_refiner._generate_text(prompt)   
    
    async def run_iterations(  
        self,  
        initial_persona: str,  
        content: Union[str, List[str]],  
        ground_truth: Union[str, List[str]],  
        persona_formatter: Optional[Callable] = None,  
        analysis_formatter: Optional[Callable] = None,  
        refinement_formatter: Optional[Callable] = None,
        response_max_tokens: int = 2500,
        pre_generated_response: Optional[Union[str, List[str]]] = None,  # Support pre-generated responses
        interview_mode: bool = False  # Interview mode flag
    ) -> Dict[str, Any]:  
        """  
        Run iterations of the DPRF process to refine the persona.  
        
        Args:  
            initial_persona: Initial description of the persona  
            content: Text content to process (single string or list for interview mode)  
            ground_truth: Ground truth response (single string or list for interview mode)  
            persona_formatter: Optional custom formatter for persona prompt  
            analysis_formatter: Optional custom formatter for analysis  
            refinement_formatter: Optional custom formatter for refinement  
            pre_generated_response: Optional pre-generated response(s)
            interview_mode: If True, handles multiple data points with batch generation and unified refinement
            
        Returns:  
            Dictionary with results of the DPRF process  
        """  
        # Log custom formatters if provided  
        if analysis_formatter:  
            print("Using custom analysis_formatter") 
        if refinement_formatter:  
            print("Using custom refinement_formatter") 

        if interview_mode:
            return await self._run_interview_mode_iterations(
                initial_persona=initial_persona,
                content_list=content if isinstance(content, list) else [content],
                ground_truth_list=ground_truth if isinstance(ground_truth, list) else [ground_truth],
                persona_formatter=persona_formatter,
                analysis_formatter=analysis_formatter,
                refinement_formatter=refinement_formatter,
                response_max_tokens=response_max_tokens,
                pre_generated_responses=pre_generated_response if isinstance(pre_generated_response, list) else ([pre_generated_response] if pre_generated_response else None)
            )
        else:
            return await self._run_standard_mode_iterations(
                initial_persona=initial_persona,
                content=content if isinstance(content, str) else content[0],
                ground_truth=ground_truth if isinstance(ground_truth, str) else ground_truth[0],
                persona_formatter=persona_formatter,
                analysis_formatter=analysis_formatter,
                refinement_formatter=refinement_formatter,
                response_max_tokens=response_max_tokens,
                pre_generated_response=pre_generated_response if isinstance(pre_generated_response, str) else (pre_generated_response[0] if pre_generated_response else None)
            )

    async def _run_interview_mode_iterations(
        self,
        initial_persona: str,
        content_list: List[str],
        ground_truth_list: List[str],
        persona_formatter: Optional[Callable] = None,
        analysis_formatter: Optional[Callable] = None,
        refinement_formatter: Optional[Callable] = None,
        response_max_tokens: int = 2500,
        pre_generated_responses: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run DPRF iterations in interview mode: batch generate -> unified refine -> batch generate again
        """
        self.logger.info(f"Starting interview mode iterations with {len(content_list)} data points")
        
        current_persona = initial_persona
        iteration_results = []
        
        # Store per-datapoint results for each iteration
        datapoint_results = []
        
        for iteration in range(1, self.max_iterations + 1):
            self.logger.info(f"Starting iteration {iteration}/{self.max_iterations}")
            
            # Step 1: Generate responses for all data points
            if iteration == 1 and pre_generated_responses is not None:
                self.logger.info("Using pre-generated responses for first iteration")
                generated_responses = pre_generated_responses
            else:
                self.logger.info(f"Generating responses for {len(content_list)} data points")
                response_tasks = []
                for content in content_list:
                    response_tasks.append(
                        self.generate_response(
                            persona=current_persona,
                            content=content,
                            custom_formatter=persona_formatter,
                            max_output_tokens=response_max_tokens
                        )
                    )
                generated_responses = await asyncio.gather(*response_tasks)
            
            # Step 2: Combine all content, responses, and ground truth for unified analysis
            combined_content_parts = []
            combined_response_parts = []
            combined_ground_truth_parts = []
            
            for i, (content, response, gt) in enumerate(zip(content_list, generated_responses, ground_truth_list)):
                combined_content_parts.append(f"Data Point {i+1}:\n{content}")
                combined_response_parts.append(f"Response {i+1}:\n{response}")
                combined_ground_truth_parts.append(f"Ground Truth {i+1}:\n{gt}")
            
            combined_content = "\n\n".join(combined_content_parts)
            combined_response = "\n\n".join(combined_response_parts)
            combined_ground_truth = "\n\n".join(combined_ground_truth_parts)
            
            # Check combined ground_truth length and warn if too long
            max_ground_truth_tokens = 2500
            gt_token_count = self.count_tokens(combined_ground_truth)
            if gt_token_count > max_ground_truth_tokens:
                self.logger.warning(
                    f"WARNING: Combined ground_truth is {gt_token_count} tokens, exceeding recommended limit of {max_ground_truth_tokens} tokens. Will NOT be truncated."
                )
            combined_ground_truth_trimmed = combined_ground_truth
            
            # Step 3: Unified refinement based on all data points
            self.logger.info("Performing unified persona refinement based on all data points")
            refined_persona, analysis = await self.persona_refiner.refine_persona(
                persona=current_persona,
                content=combined_content,
                generated_response=combined_response,
                ground_truth=combined_ground_truth_trimmed,
                analysis_formatter=analysis_formatter,
                refinement_formatter=refinement_formatter
            )
            
            # Store iteration result
            iteration_result = {
                "iteration": iteration,
                "persona": current_persona,
                "combined_content": combined_content,
                "combined_generated_response": combined_response,
                "individual_responses": generated_responses,
                "analysis": analysis,
                "refined_persona": refined_persona,
                "num_data_points": len(content_list)
            }
            
            iteration_results.append(iteration_result)
            
            # Store per-datapoint results for this iteration
            datapoint_iteration_results = []
            for i, (content, response, gt) in enumerate(zip(content_list, generated_responses, ground_truth_list)):
                datapoint_iteration_results.append({
                    "datapoint_index": i,
                    "content": content,
                    "ground_truth": gt,
                    "generated_response": response,
                    "persona": current_persona,  # Persona used for this generation
                    "iteration": iteration
                })
            datapoint_results.append(datapoint_iteration_results)
            
            # Check if the persona has changed
            if refined_persona == current_persona:
                self.logger.info(f"Persona did not change after iteration {iteration}. Stopping early.")
                break
            
            # Update current persona for next iteration
            current_persona = refined_persona
        
        # Generate final responses with the refined persona if we did refinement
        final_responses = generated_responses  # Default to last generated responses
        if len(iteration_results) > 0 and current_persona != initial_persona:
            self.logger.info("Generating final responses with refined persona")
            final_response_tasks = []
            for content in content_list:
                final_response_tasks.append(
                    self.generate_response(
                        persona=current_persona,
                        content=content,
                        custom_formatter=persona_formatter,
                        max_output_tokens=response_max_tokens
                    )
                )
            final_responses = await asyncio.gather(*final_response_tasks)
        
        # Prepare final results
        final_results = {
            "initial_persona": initial_persona,
            "final_persona": current_persona,
            "content_list": content_list,
            "ground_truth_list": ground_truth_list,
            "final_responses": final_responses,
            "iterations": iteration_results,
            "datapoint_results_by_iteration": datapoint_results,
            "num_data_points": len(content_list),
            "interview_mode": True,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Save logs if requested
        if self.save_logs:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(self.log_dir, f"dprf_interview_log_{timestamp}.json")
            save_json(final_results, log_path)
            self.logger.info(f"Saved DPRF interview log to {log_path}")
        
        return final_results

    async def _run_standard_mode_iterations(
        self,
        initial_persona: str,
        content: str,
        ground_truth: str,
        persona_formatter: Optional[Callable] = None,
        analysis_formatter: Optional[Callable] = None,
        refinement_formatter: Optional[Callable] = None,
        response_max_tokens: int = 2500,
        pre_generated_response: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run DPRF iterations in standard mode (original logic)
        """
        # Initialize variables  
        current_persona = initial_persona  
        iteration_results = []  
        
        for iteration in range(1, self.max_iterations+1):  
            self.logger.info(f"Starting iteration {iteration}/{self.max_iterations}")  
            
            # Generate response using current persona or use pre-generated response for first iteration
            if iteration == 1 and pre_generated_response is not None:
                self.logger.info("Using pre-generated response for first iteration")
                generated_response = pre_generated_response
            else:
                generated_response = await self.generate_response(  
                    persona=current_persona,  
                    content=content,  
                    custom_formatter=persona_formatter,  
                    max_output_tokens=response_max_tokens  
                )  
            # Check ground_truth length and warn if too long
            max_ground_truth_tokens = 2500
            gt_token_count = self.count_tokens(ground_truth)
            if gt_token_count > max_ground_truth_tokens:
                self.logger.warning(
                    f"WARNING: Ground truth is {gt_token_count} tokens, exceeding recommended limit of {max_ground_truth_tokens} tokens. Will NOT be truncated."
                )
            ground_truth_trimmed = ground_truth
            
            # Refine persona based on comparison with ground truth  
            refined_persona, analysis = await self.persona_refiner.refine_persona(  
                persona=current_persona,  
                content=content,  
                generated_response=generated_response,  
                ground_truth=ground_truth_trimmed,
                analysis_formatter=analysis_formatter,
                refinement_formatter=refinement_formatter
            )  
            
            # Store results for this iteration  
            iteration_result = {  
                "iteration": iteration,  
                "persona": current_persona,  
                "generated_response": generated_response,  
                "analysis": analysis,  
                "refined_persona": refined_persona  
            }  
            
            iteration_results.append(iteration_result)  
            
            # Check if the persona has changed  
            if refined_persona == current_persona:  
                self.logger.info(f"Persona did not change after iteration {iteration}. Stopping early.")  
                break  
            
            # Update current persona for next iteration  
            current_persona = refined_persona  
        
        # Prepare final results  
        final_results = {  
            "initial_persona": initial_persona,  
            "final_persona": current_persona,  
            "content": content,
            "ground_truth": ground_truth_trimmed,  
            "iterations": iteration_results,
            "interview_mode": False,
            "timestamp": datetime.datetime.now().isoformat()  
        }  
        
        # Save logs if requested  
        if self.save_logs:  
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  
            log_path = os.path.join(self.log_dir, f"dprf_log_{timestamp}.json")  
            save_json(final_results, log_path)  
            self.logger.info(f"Saved DPRF log to {log_path}")  
        
        return final_results  
    
    def peer_review_task(self, content: str) -> str:  
        """  
        Return standard instructions for a peer review task based on paper content.  
        
        Args:  
            content: The content of the paper to review  
            
        Returns:  
            Instruction string for peer review  
        """  
        return format_peer_review_instruction(content)   
    
    # Add in core/dprf_agent.py  
    def _init_transformers_model(self, model_name):  
        """Initialize Transformers model"""  
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer  
        import torch  
        
        try:  
            print(f"Loading model from {model_name}...")  
            tokenizer = AutoTokenizer.from_pretrained(model_name)  
            
            # Use torch.bfloat16 or torch.float16 to reduce memory usage  
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32  
            device = "cuda" if torch.cuda.is_available() else "cpu"  
            
            model = AutoModelForCausalLM.from_pretrained(  
                model_name,  
                torch_dtype=dtype,  
                device_map="auto",  
                trust_remote_code=True  
            )  
            
            return {"model": model, "tokenizer": tokenizer}  
        except Exception as e:  
            print(f"Error initializing Transformers model: {e}")  
            raise  
            
    # For Transformers  
    def _generate_with_transformers(self, model_info, prompt, max_tokens=512):  
        model = model_info["model"]  
        tokenizer = model_info["tokenizer"]  
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)  
        
        outputs = model.generate(  
            inputs.input_ids,  
            max_new_tokens=max_tokens,  
            temperature=0.7,  
            top_p=0.95,  
            repetition_penalty=1.1,  
            do_sample=True  
        )  
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)  
        return response  