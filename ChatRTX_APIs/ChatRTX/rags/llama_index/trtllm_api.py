# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import gc
import time
import uuid
import torch
from ChatRTX.inference.trtllm.trtllm import TrtLlm
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    ChatResponseGen,
    CompletionResponseGen,
    LLMMetadata
)
from llama_index.core.base.llms.generic_utils import (
    completion_response_to_chat_response,
    stream_completion_response_to_chat_response,
)
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.custom import CustomLLM
from typing import Any, Callable, Dict, Optional, Sequence
from typing import Any, Callable, Dict, Optional

class TrtLlmAPI(CustomLLM):
    """A custom LLM class for handling models optimized with TensorRT.

    Attributes:
        messages_to_prompt (Callable): Function to convert messages to a prompt.
        completion_to_prompt (Callable): Function to convert model completions to prompts.
        generate_kwargs (Dict[str, Any]): Keyword arguments used for text generation.
        model_kwargs (Dict[str, Any]): Keyword arguments for model initialization.
    """
    messages_to_prompt: Callable = Field(
        description="The function to convert messages to a prompt.", exclude=True
    )
    completion_to_prompt: Callable = Field(
        description="The function to convert a completion to a prompt.", exclude=True
    )
    generate_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Kwargs used for generation."
    )
    model_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Kwargs used for model initialization."
    )

    _model: Any = PrivateAttr()
    _model_path: Any = PrivateAttr()
    _verbose = PrivateAttr()
    _context_window = PrivateAttr()
    _max_new_tokens = PrivateAttr()

    def __init__(
            self,
            model_path: Optional[str] = None,
            tokenizer_dir: Optional[str] = None,
            vocab_file: Optional[str] = None,
            temperature: float = 0.1,
            max_new_tokens: int = DEFAULT_NUM_OUTPUTS,
            context_window: int = DEFAULT_CONTEXT_WINDOW,
            #messages_to_prompt: Optional[Callable] = None,
            completion_to_prompt: Optional[Callable] = None,
            prompt_template=None,
            callback_manager: Optional[CallbackManager] = None,
            generate_kwargs: Optional[Dict[str, Any]] = None,
            model_kwargs: Optional[Dict[str, Any]] = None,
            use_py_session=True,
            add_special_tokens=False,
            trtLlm_debug_mode=False
    ) -> None:
        """Initialize the LlamaIndexTrtLlm class with specified parameters.

        Args:
            model_path (str, optional): Path to the TensorRT model engine.
            tokenizer_dir (str, optional): Directory containing the tokenizer files.
            vocab_file (str, optional): Path to the vocabulary file.
            temperature (float): Sampling temperature for generation.
            max_new_tokens (int): Maximum number of tokens to generate.
            context_window (int): Number of tokens in the model's context window.
            messages_to_prompt (Callable, optional): Function to convert messages to prompts.
            completion_to_prompt (Callable, optional): Function to convert completions to prompts.
            prompt_template: Template for formatting prompts (unused placeholder).
            callback_manager (CallbackManager, optional): Manager for handling callbacks.
            generate_kwargs (dict, optional): Additional keyword arguments for generation.
            model_kwargs (dict, optional): Additional keyword arguments for model setup.
            use_py_session (bool): Flag to use Python session for execution.
            add_special_tokens (bool): Flag to add special tokens in prompts.
            trtLlm_debug_mode (bool): Enable debug mode for TensorRT operations.
            verbose (bool): Enable verbose output.
        """
        self._model = TrtLlm(
            model_path=model_path,
            tokenizer_dir=tokenizer_dir,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            context_window=context_window,
            vocab_file=vocab_file,  # Previously was set as None mistakenly.
            use_py_session=use_py_session,
            add_special_tokens=add_special_tokens,
            trtLlm_debug_mode=trtLlm_debug_mode
        )

        self._model_path = model_path
        self._context_window = context_window
        self._max_new_tokens = max_new_tokens

        model_kwargs = model_kwargs or {}
        model_kwargs.update({"n_ctx": context_window, "verbose": False})
        generate_kwargs = generate_kwargs or {}
        generate_kwargs.update({"temperature": temperature, "max_tokens": max_new_tokens})

        super().__init__(
            model_path=model_path,
            temperature=temperature,
            context_window=context_window,
            max_new_tokens=max_new_tokens,
            messages_to_prompt=None,
            completion_to_prompt=completion_to_prompt,
            callback_manager=callback_manager,
            generate_kwargs=generate_kwargs,
            model_kwargs=model_kwargs,
            verbose=False,
        )

    def generate_completion_dict(self, text_str):
        """
        Generate a dictionary for text completion details.

        Args:
            text_str (str): The generated text string from the model.

        Returns:
            dict: A dictionary containing completion details including the text,
                  a unique completion ID, and metadata about the generation.
        """
        completion_id: str = f"cmpl-{str(uuid.uuid4())}"
        created: int = int(time.time())
        model_name: str = self._model_name if hasattr(self, '_model_name') else 'Unknown'

        return {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "text": text_str,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": 'stop'
                }
            ],
            "usage": {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None
            }
        }

    @classmethod
    def class_name(cls) -> str:
        """Return the class name as a string."""
        return cls.__name__

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """
        Process a sequence of chat messages and return a chat response.

        Args:
            messages (Sequence[ChatMessage]): A sequence of messages to process.
            kwargs (dict): Additional keyword arguments for processing.

        Returns:
            ChatResponse: The response generated from the processed chat messages.
        """

        prompt = self.messages_to_prompt(messages)
        completion_response = self.complete(prompt, formatted=True, **kwargs)
        return completion_response_to_chat_response(completion_response)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:

        """
        Process a sequence of chat messages and return a streaming chat response.

        This method streams the chat response as it is generated, allowing for real-time
        interaction and potentially large-scale processing without waiting for the full response.

        Args:
            messages (Sequence[ChatMessage]): A sequence of chat messages to be processed.
            kwargs (dict): Additional keyword arguments that may influence the response.

        Returns:
            ChatResponseGen: A generator that yields chat response parts as they are generated.
        """

        prompt = self.messages_to_prompt(messages)
        completion_response = self.stream_complete(prompt, formatted=True, **kwargs)
        return stream_completion_response_to_chat_response(completion_response)

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """
        Generate a completion response from a given prompt.

        Args:
            prompt (str): The prompt to process.
            kwargs (dict): Additional keyword arguments for completion generation.

        Returns:
            CompletionResponse: Structured response containing the text and metadata.
        """
        is_formatted = kwargs.pop("formatted", False)
        output_txt = self._model.complete(prompt, **kwargs)
        return CompletionResponse(text=output_txt, raw=self.generate_completion_dict(output_txt))

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:

        """
        Stream completions for a given prompt in a generator fashion.

        This function is designed to generate and yield completion responses incrementally,
        which is useful for handling long or continuous interactions without needing to wait
        for a full completion.

        Args:
            prompt (str): The prompt to generate completions for.
            formatted (bool): Indicates whether the prompt is pre-formatted.
            kwargs (dict): Additional keyword arguments for dynamic completion generation.

        Returns:
            CompletionResponseGen: A generator that yields completion responses as generated.
        """

        self.generate_kwargs.update({"stream": True})

        if not formatted:
            prompt = self.completion_to_prompt(prompt)

        response_iter = self._model.stream_complete(prompt=prompt, **kwargs)

        def gen() -> CompletionResponseGen:
            text = ""
            for response in response_iter:
                delta = response
                text += delta
                yield CompletionResponse(delta=delta, text=text, raw=self.generate_completion_dict(response))

        return gen()

    def unload_llm(self):
        """
        Unload the model from memory and perform necessary cleanup.
        """
        if self._model is not None:
            del self._model
            self._model = None  # Ensure the reference is cleaned up after deletion.

        torch.cuda.empty_cache()
        gc.collect()

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            context_window=self._context_window,
            num_output=self._max_new_tokens,
            model_name=self._model_path,
        )
