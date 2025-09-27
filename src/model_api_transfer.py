import base64
import logging
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Optional, List, Any
from PIL import Image
import argparse
import requests
import io
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_markdown_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

@dataclass
class Prompt:
    """Prompt class for model input
    
    Attributes:
        text: The text content of the prompt
        role: The role of the message sender (e.g., 'user', 'system')
        images: Optional list of images to include with the prompt
    """
    text: str
    role: str = "user"
    images: Optional[List[Image.Image]] = None

class ModelOutput:
    """Class for model generation output
    
    Attributes:
        text: The generated text response
        usage: Dictionary containing token usage statistics
        finish_reason: Reason why the model stopped generating
    """
    def __init__(self, text: str, usage: dict = None, finish_reason: str = None):
        self.text = text
        self.usage = usage or {}
        self.finish_reason = finish_reason

class BaseModelInterface:
    """Base interface for AI model implementations
    
    This class provides common functionality for all model interfaces
    and defines the interface that specific model implementations must follow.
    
    Attributes:
        config: Configuration for the model
        _client: The underlying client for the model API
    """
    
    def __init__(self, model_name: str, api_url: Optional[str] = None, api_key: Optional[str] = None, 
                 max_tokens: int = 8192, temperature: float = 0.0, retries: int = 3, retry_interval: int = 1):
        """Initialize the model interface
        
        Args:
            config: Configuration for the model
        """
        self.model_name = model_name
        self.api_url = api_url
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.retries = retries
        self.retry_interval = retry_interval
        self._client = None
        
    def _retry_operation(self, operation):
        """Execute operation with retry logic
        
        Args:
            operation: Function to execute with retries
            
        Returns:
            The result of the operation
            
        Raises:
            Exception: If all retry attempts fail
        """
        attempts = 0
        while attempts < self.retries:
            try:
                return operation()
            except Exception as e:
                attempts += 1
                delay = self.retry_interval * (2 ** attempts)
                logger.error(f"Operation failed: {e}. Attempt {attempts}/{self.retries}")
                if attempts < self.retries:
                    time.sleep(delay)
                else:
                    raise
                    
    def _process_image(self, image: Image.Image) -> str:
        """Convert image to base64 string
        
        Args:
            image: PIL Image to convert
            
        Returns:
            Base64 encoded string of the image
        """
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def generate(self, prompt: Prompt) -> ModelOutput:
        """Generate response from model
        
        Args:
            prompt: The prompt to send to the model
            
        Returns:
            ModelOutput containing the generated response
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

class OpenAIInterface(BaseModelInterface):
    """Interface for OpenAI models"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from openai import OpenAI
        # import pdb; pdb.set_trace()
        if self.api_url:
            # if not self.api_url.endswith('v1'):
            #     self.api_url = self.api_url + '/v1'
            # import pdb; pdb.set_trace()
            self._client = OpenAI(api_key=self.api_key, base_url=self.api_url)
        else:
            self._client = OpenAI(api_key=self.api_key)
        
    def generate(self, prompt: Prompt) -> ModelOutput:
        def _generate():
            messages = [{"role": prompt.role, "content": []}]
            
            # Add text content
            if prompt.text:
                messages[0]["content"].append({"type": "text", "text": prompt.text})
            
            # Add image content if available
            if prompt.images:
                for image in prompt.images:
                    image_content = {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{self._process_image(image)}"}
                    }
                    messages[0]["content"].append(image_content)
            
            completion = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            # import pdb; pdb.set_trace()
            return ModelOutput(
                text=completion.choices[0].message.content,
                usage=completion.usage.model_dump(),
                finish_reason=completion.choices[0].finish_reason
            )
        return self._retry_operation(_generate)

class VLLMInterface(BaseModelInterface):
    """Interface for VLLM models (locally deployed with OpenAI-compatible API)"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # We'll use requests directly instead of the OpenAI client
        self._requests = requests
        
    def generate(self, prompt: Prompt) -> ModelOutput:
        def _generate():
            # Format messages with proper content structure for VLLM
            content = []
            
            # Add text content
            if prompt.text:
                content.append({"type": "text", "text": prompt.text})
            
            # Add image content if available
            if prompt.images:
                for image in prompt.images:
                    image_content = {
                        "type": "image_url", 
                        "image_url": {"url": f"data:image/png;base64,{self._process_image(image)}"}
                    }
                    content.append(image_content)

            # import pdb; pdb.set_trace()
            
            # Prepare the request payload
            payload = {
                "model": self.model_name,
                "messages": [{"role": prompt.role, "content": content}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "encoding_format": "float"
            }

            # import pdb; pdb.set_trace()
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            # Make the direct API call
            response = self._requests.post(
                f"{self.api_url}/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            response_json = response.json()
            
            # Extract the response content
            # import pdb; pdb.set_trace()
            return ModelOutput(
                text=response_json['choices'][0]['message']['content'],
                usage=response_json.get('usage', {}),
                finish_reason=response_json['choices'][0].get('finish_reason', None)
            )
        return self._retry_operation(_generate)

class QwenInterface(BaseModelInterface):
    """Interface for Qwen models"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.server_url = self.api_url

    def generate(self, prompt: Prompt) -> ModelOutput:
        def _generate():
            if prompt.text:
                data = {
                    "prompt": prompt.text
                }
            if prompt.images:
                image = prompt.images[0]

                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='JPEG')
                image_data = img_byte_arr.getvalue()
                
                # Prepare request data
                files = {
                    "image": ("image.jpg", image_data, "image/jpeg")
                }
            response = requests.post(
                f"{self.server_url}/generate",
                files=files,
                data=data
            )
            response.raise_for_status()
            return ModelOutput(text=response.json()["response"])
        return self._retry_operation(_generate)

class LocalInterface(BaseModelInterface):
    """Interface for locally deployed models"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.server_url = self.api_url

    def generate(self, prompt: Prompt) -> ModelOutput:
        def _generate():
            if prompt.text:
                data = {
                    "prompt": prompt.text
                }
            if prompt.images:
                # for image in prompt.images:
                #     image_data = self._process_image(image)
                #     files = {
                #         "image": ("image.jpg", image_data, "image/jpeg")
                #     }
                image = prompt.images[0]

                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='JPEG')
                image_data = img_byte_arr.getvalue()
                
                # Prepare request data
                files = {
                    "image": ("image.jpg", image_data, "image/jpeg")
                }
            response = requests.post(
                f"{self.server_url}/generate",
                files=files,
                data=data
            )
            response.raise_for_status()
            return ModelOutput(text=response.json()["response"])
        return self._retry_operation(_generate)



def create_model(model_name: str, api_url: Optional[str] = None, api_key: Optional[str] = None, **kwargs) -> BaseModelInterface:
    """Factory function to create model interface
    
    Args:
        model_name: The name of the model to create an interface for.
        api_url: Optional API endpoint URL.
        api_key: Optional API key.
        **kwargs: Additional keyword arguments for the model interface.
        
    Returns:
        An instance of the appropriate model interface
        
    Raises:
        ValueError: If the provider is not supported
    """
    model_kwargs = {
        "model_name": model_name,
        "api_url": api_url,
        "api_key": api_key,
        **kwargs
    }
    model_name_lower = model_name.lower()
    if "gemini" in model_name_lower or "vllm" in model_name_lower or "claude" in model_name_lower or 'gpt' in model_name_lower or 'openai' in model_name_lower or 'qwen2.5' in model_name_lower:
        return OpenAIInterface(**model_kwargs)
    elif "prismatic" in model_name_lower or "paligemma" in model_name_lower or "phi3" in model_name_lower:
        return LocalInterface(**model_kwargs)
    else:
        return LocalInterface(**model_kwargs)

def generate_response(model_name: str, prompt_text: str, image_path: str = None, api_url: str = None, api_key: str = None) -> str:
    """High-level function to generate response from model
    
    Args:
        model_name: Name of the model to use
        prompt_text: Text prompt to send to the model
        image_path: Optional path to an image file to include with the prompt
        api_url: Optional API endpoint URL
        
    Returns:
        The generated text response
    """
    
    # Create model interface
    model = create_model(
        model_name=model_name,
        api_url=api_url,
        api_key=api_key
    )
    
    # Process image if provided
    images = None
    if image_path:
        try:
            images = [Image.open(image_path)]
            if images[0].mode == 'RGBA':
                images[0] = images[0].convert('RGB')
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
    
    # Create prompt
    prompt = Prompt(text=prompt_text, images=images)
    
    # Generate response
    response = model.generate(prompt)
    return response.text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Model Interface")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--prompt", required=False, help="Input prompt")
    parser.add_argument("--prompt_file", required=False, help="Input prompt file")
    parser.add_argument("--image", help="Optional image path")
    parser.add_argument("--api_url", help="Optional API endpoint", default="http://localhost:8000/v1")
    parser.add_argument("--api_key", help="Optional API key")
    
    args = parser.parse_args()
    args.api_key = "cap3d 58d0bacc761d4678855f9582819abcc77a4bacaa54984213905d9d546670638a"
    
    if args.prompt_file:
        args.prompt = read_markdown_file(args.prompt_file)
    
    response = generate_response(
        args.model,
        args.prompt,
        args.image,
        args.api_url,
        args.api_key
    )
    
    print("\nModel Response:")
    print("=" * 50)
    print(response)
    print("=" * 50)
