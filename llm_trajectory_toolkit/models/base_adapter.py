"""
Base Model Adapter Interface

Abstract base class for supporting different LLM architectures.
"""

from abc import ABC, abstractmethod
from typing import Optional, List
import torch


class BaseModelAdapter(ABC):
    """
    Abstract interface for LLM trajectory capture.

    Subclass this to add support for new model architectures.

    Example:
        >>> class MyModelAdapter(BaseModelAdapter):
        ...     def load_model(self):
        ...         return MyModel.from_pretrained(self.model_name)
        ...
        ...     def get_layer_modules(self):
        ...         return self.model.my_custom_layers
    """

    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        Args:
            model_name: HuggingFace model identifier or path
            device: Device to load model on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def load_model(self):
        """
        Load model and tokenizer.

        Must set:
        - self.model: The loaded model
        - self.tokenizer: The loaded tokenizer

        Returns:
            Tuple of (model, tokenizer)
        """
        pass

    @abstractmethod
    def get_layer_modules(self) -> List:
        """
        Get list of transformer layer modules for hook registration.

        Returns:
            List of nn.Module objects representing transformer blocks
        """
        pass

    def get_num_layers(self) -> int:
        """
        Get total number of transformer layers.

        Returns:
            Number of layers
        """
        layers = self.get_layer_modules()
        return len(layers)

    def prepare_for_generation(self):
        """
        Optional: Prepare model for generation (set to eval mode, etc.)
        """
        if self.model is not None:
            self.model.eval()

    def cleanup(self):
        """
        Optional: Cleanup resources (clear GPU memory, etc.)
        """
        if self.device == 'cuda':
            torch.cuda.empty_cache()


class GPT2Adapter(BaseModelAdapter):
    """
    Adapter for GPT-2 models (gpt2, gpt2-medium, gpt2-large, gpt2-xl).

    Example:
        >>> adapter = GPT2Adapter("gpt2")
        >>> model, tokenizer = adapter.load_model()
    """

    def load_model(self):
        """Load GPT-2 model and tokenizer"""
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = GPT2LMHeadModel.from_pretrained(
            self.model_name,
            device_map=self.device,
            torch_dtype=torch.float16 if 'cuda' in str(self.device) else torch.float32,
            low_cpu_mem_usage=True
        )

        self.model.eval()
        return self.model, self.tokenizer

    def get_layer_modules(self) -> List:
        """Get GPT-2 transformer blocks"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        return self.model.transformer.h


class LlamaAdapter(BaseModelAdapter):
    """
    Adapter for Llama models (Llama-2-7b, Llama-2-13b, Llama-2-70b, Llama-3-*).

    Supports:
    - Automatic multi-GPU loading for 70B models
    - Layer sampling for memory efficiency

    Example:
        >>> adapter = LlamaAdapter("meta-llama/Llama-2-7b-hf")
        >>> model, tokenizer = adapter.load_model()
    """

    def __init__(self, model_name: str, device: Optional[str] = None, max_memory_per_gpu: str = "75GB"):
        """
        Args:
            model_name: HuggingFace Llama model identifier
            device: Device ('cuda', 'cpu', or None for auto)
            max_memory_per_gpu: Memory limit per GPU for large models
        """
        super().__init__(model_name, device)
        self.max_memory_per_gpu = max_memory_per_gpu

    def load_model(self):
        """Load Llama model and tokenizer"""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Use device_map='auto' for 70B models
        if '70b' in self.model_name.lower():
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map='auto',
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                max_memory={i: self.max_memory_per_gpu for i in range(torch.cuda.device_count())}
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map='cuda:0' if 'cuda' in str(self.device) else 'cpu',
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )

        self.model.eval()
        return self.model, self.tokenizer

    def get_layer_modules(self) -> List:
        """Get Llama transformer blocks"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        return self.model.model.layers


class UniversalAdapter(BaseModelAdapter):
    """
    Universal adapter that auto-detects architecture and works with any HuggingFace model.

    Supports:
    - GPT-2 (all variants)
    - Llama (2, 3, all sizes)
    - Qwen (all sizes)
    - Falcon (all sizes)
    - DeepSeek (all sizes)
    - Mistral, Mixtral
    - Any other transformer-based causal LM

    Features:
    - Automatic architecture detection
    - Automatic layer path detection
    - Intelligent memory management for large models
    - 8-bit quantization for 70B+ models

    Example:
        >>> adapter = UniversalAdapter("Qwen/Qwen-72B")
        >>> model, tokenizer = adapter.load_model()
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        max_memory_per_gpu: str = "40GB",
        use_8bit: Optional[bool] = None,
        trust_remote_code: bool = True
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            device: Device ('cuda', 'cpu', or None for auto)
            max_memory_per_gpu: Memory limit per GPU for large models
            use_8bit: Force 8-bit quantization (auto-detect if None)
            trust_remote_code: Allow remote code execution for some models
        """
        super().__init__(model_name, device)
        self.max_memory_per_gpu = max_memory_per_gpu
        self.use_8bit = use_8bit
        self.trust_remote_code = trust_remote_code
        self.layer_path = None  # Will be detected

    def _detect_model_size(self) -> str:
        """Detect model size category from name"""
        name_lower = self.model_name.lower()

        if '180b' in name_lower or '120b' in name_lower:
            return 'xxl'  # 100B+
        elif '70b' in name_lower or '72b' in name_lower or '67b' in name_lower:
            return 'xl'   # 60-80B
        elif '13b' in name_lower or '20b' in name_lower:
            return 'large'  # 10-20B
        elif '7b' in name_lower:
            return 'medium'  # 7B
        else:
            return 'small'  # < 7B

    def load_model(self):
        """Load model with automatic architecture detection and optimization"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import os

        # Set memory optimization
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code
        )

        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Detect model size and configure loading
        model_size = self._detect_model_size()

        # Auto-enable 8-bit for very large models
        if self.use_8bit is None:
            self.use_8bit = model_size in ['xl', 'xxl']

        # Build loading kwargs
        load_kwargs = {
            'trust_remote_code': self.trust_remote_code,
            'low_cpu_mem_usage': True
        }

        # Configure device and precision based on size
        if self.device == 'cuda' or (self.device is None and torch.cuda.is_available()):
            load_kwargs['torch_dtype'] = torch.float16

            if model_size in ['xl', 'xxl']:
                # Large models: use device_map auto
                load_kwargs['device_map'] = 'auto'

                if self.use_8bit:
                    load_kwargs['load_in_8bit'] = True
                    num_gpus = torch.cuda.device_count()
                    load_kwargs['max_memory'] = {i: self.max_memory_per_gpu for i in range(num_gpus)}
                    print(f"⚠ Large model detected ({model_size}), using 8-bit quantization across {num_gpus} GPU(s)")
            else:
                # Smaller models: single device
                load_kwargs['device_map'] = 'cuda:0'
        else:
            # CPU mode
            load_kwargs['torch_dtype'] = torch.float32
            load_kwargs['device_map'] = 'cpu'

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **load_kwargs
        )

        self.model.eval()

        # Detect layer path
        self._detect_layer_path()

        return self.model, self.tokenizer

    def _detect_layer_path(self):
        """Automatically detect the path to transformer layers"""
        if self.model is None:
            raise ValueError("Model not loaded")

        # Common layer paths for different architectures
        layer_paths = [
            # Llama, Mistral, Qwen, DeepSeek
            ('model', 'layers'),
            # GPT-2, GPT-Neo
            ('transformer', 'h'),
            # Falcon
            ('transformer', 'layers'),
            # BLOOM
            ('transformer', 'h'),
            # OPT
            ('model', 'decoder', 'layers'),
            # Direct access (some custom models)
            ('layers',),
            ('h',),
        ]

        for path in layer_paths:
            try:
                obj = self.model
                for attr in path:
                    obj = getattr(obj, attr)

                # Check if it's a list/ModuleList of layers
                if hasattr(obj, '__len__') and len(obj) > 0:
                    self.layer_path = path
                    print(f"✓ Detected layer path: {'.'.join(path)} ({len(obj)} layers)")
                    return
            except AttributeError:
                continue

        # If no standard path found, try to find it
        raise ValueError(
            f"Could not auto-detect layer path for {self.model_name}. "
            f"Model architecture: {self.model.__class__.__name__}. "
            f"Please implement a custom adapter."
        )

    def get_layer_modules(self) -> List:
        """Get transformer layer modules (auto-detected)"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        if self.layer_path is None:
            self._detect_layer_path()

        obj = self.model
        for attr in self.layer_path:
            obj = getattr(obj, attr)

        return obj
