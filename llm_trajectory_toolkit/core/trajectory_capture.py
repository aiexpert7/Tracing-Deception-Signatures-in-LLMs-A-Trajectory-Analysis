"""
Trajectory Capture Module

Captures layer-wise hidden states during LLM text generation using forward hooks.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CapturedTrajectory:
    """Container for captured trajectory data"""
    prompt: str
    generated_text: str
    generated_tokens: List[str]
    layer_states: np.ndarray  # (num_layers, hidden_dim)
    layer_indices: List[int]
    metadata: Dict


class TrajectoryCapture:
    """
    Captures internal layer states during model generation.

    Uses PyTorch forward hooks to extract hidden states from each transformer layer
    at the final generated token position.

    Example:
        >>> capture = TrajectoryCapture()
        >>> traj = capture.capture(model, tokenizer, "What is 2+2?", max_tokens=15)
        >>> print(traj.layer_states.shape)  # (num_layers, hidden_dim)
    """

    def __init__(self, sample_rate: int = 1):
        """
        Args:
            sample_rate: Capture every Nth layer (1 = all layers, 4 = every 4th)
        """
        self.sample_rate = sample_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def capture(
        self,
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 15,
        layer_modules: Optional[List] = None
    ) -> CapturedTrajectory:
        """
        Capture trajectory for a single prompt.

        Args:
            model: HuggingFace model (GPT-2, Llama, etc.)
            tokenizer: Corresponding tokenizer
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            layer_modules: List of layer modules to hook (auto-detected if None)

        Returns:
            CapturedTrajectory with layer states and metadata
        """

        # Auto-detect layers if not provided
        if layer_modules is None:
            layer_modules = self._get_layer_modules(model)

        # Storage for captured states
        layer_states = {}
        hooks = []

        # Register hooks
        for idx, layer in enumerate(layer_modules):
            if idx % self.sample_rate == 0:
                hook_fn = self._create_hook(layer_states, idx)
                hook = layer.register_forward_hook(hook_fn)
                hooks.append((hook, idx))

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate with hooks active
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Deterministic
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
                return_dict_in_generate=True
            )

        # Remove hooks
        for hook, _ in hooks:
            hook.remove()

        # Extract generated tokens
        generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
        generated_tokens = [tokenizer.decode([tok_id]) for tok_id in generated_ids]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Convert to trajectory array
        layer_indices = sorted(layer_states.keys())
        trajectory = np.array([layer_states[idx] for idx in layer_indices])

        # Clean GPU memory
        torch.cuda.empty_cache()

        return CapturedTrajectory(
            prompt=prompt,
            generated_text=generated_text,
            generated_tokens=generated_tokens,
            layer_states=trajectory,
            layer_indices=layer_indices,
            metadata={
                'num_layers': len(layer_indices),
                'hidden_dim': trajectory.shape[1] if len(trajectory) > 0 else 0,
                'sample_rate': self.sample_rate,
                'max_new_tokens': max_new_tokens
            }
        )

    def batch_capture(
        self,
        model,
        tokenizer,
        prompts: List[str],
        max_new_tokens: int = 15,
        layer_modules: Optional[List] = None
    ) -> List[CapturedTrajectory]:
        """
        Capture trajectories for multiple prompts.

        Args:
            model: HuggingFace model
            tokenizer: Corresponding tokenizer
            prompts: List of input prompts
            max_new_tokens: Maximum tokens per generation
            layer_modules: List of layer modules (auto-detected if None)

        Returns:
            List of CapturedTrajectory objects
        """
        trajectories = []

        for prompt in prompts:
            traj = self.capture(
                model, tokenizer, prompt,
                max_new_tokens=max_new_tokens,
                layer_modules=layer_modules
            )
            trajectories.append(traj)

        return trajectories

    def _create_hook(self, storage: Dict, layer_idx: int):
        """Create a forward hook that captures final token state"""
        states_list = []

        def hook_fn(module, input, output):
            # Handle different output formats
            if hasattr(output, 'last_hidden_state'):
                hidden = output.last_hidden_state
            elif isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            # Extract last token state, move to CPU immediately
            last_state = hidden[:, -1, :].detach().cpu().float().numpy()[0]
            states_list.append(last_state)

        # Store reference for later retrieval
        hook_fn.states_list = states_list

        # After generation completes, store the final state
        def store_final_state():
            if states_list:
                storage[layer_idx] = states_list[-1]

        hook_fn.finalize = store_final_state

        # Store final state in dictionary immediately
        def finalized_hook_fn(module, input, output):
            hook_fn(module, input, output)
            if states_list:
                storage[layer_idx] = states_list[-1]

        return finalized_hook_fn

    def _get_layer_modules(self, model):
        """
        Auto-detect transformer layer modules from model.

        Supports:
        - GPT-2: model.transformer.h
        - Llama: model.model.layers
        """
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            # Llama-style
            return model.model.layers
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            # GPT-2 style
            return model.transformer.h
        else:
            raise ValueError(
                "Could not auto-detect layer modules. "
                "Please provide layer_modules explicitly or implement adapter."
            )
