# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
from typing import Any, Callable, Dict, Optional, Sequence
from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.callbacks import CallbackManager
from llama_index.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.llms.base import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    LLMMetadata,
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.llms.custom import CustomLLM
from llama_index.llms.generic_utils import completion_response_to_chat_response
from llama_index.llms.generic_utils import (
    messages_to_prompt as generic_messages_to_prompt,
)
from transformers import LlamaTokenizer
import gc
import json
import torch
import numpy as np
from tensorrt_llm.runtime import ModelConfig, SamplingConfig
import tensorrt_llm
from pathlib import Path
import uuid
import time

EOS_TOKEN = 2
PAD_TOKEN = 2

class TrtLlmAPI(CustomLLM):
    model_path: Optional[str] = Field(
        description="The path to the trt engine."
    )
    temperature: float = Field(description="The temperature to use for sampling.")
    max_new_tokens: int = Field(description="The maximum number of tokens to generate.")
    context_window: int = Field(
        description="The maximum number of context tokens for the model."
    )
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
    verbose: bool = Field(description="Whether to print verbose output.")

    _model: Any = PrivateAttr()
    _model_config: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _max_new_tokens = PrivateAttr()
    _sampling_config = PrivateAttr()
    _verbose = PrivateAttr()

    def __init__(
            self,
            model_path: Optional[str] = None,
            engine_name: Optional[str] = None,
            tokenizer_dir: Optional[str] = None,
            temperature: float = 0.1,
            max_new_tokens: int = DEFAULT_NUM_OUTPUTS,
            context_window: int = DEFAULT_CONTEXT_WINDOW,
            messages_to_prompt: Optional[Callable] = None,
            completion_to_prompt: Optional[Callable] = None,
            callback_manager: Optional[CallbackManager] = None,
            generate_kwargs: Optional[Dict[str, Any]] = None,
            model_kwargs: Optional[Dict[str, Any]] = None,
            verbose: bool = False
    ) -> None:
        """        Initialize the object with the provided parameters.

        Args:
            model_path (Optional[str]): Path to the model.
            engine_name (Optional[str]): Name of the engine.
            tokenizer_dir (Optional[str]): Directory of the tokenizer.
            temperature (float): Temperature for token generation.
            max_new_tokens (int): Maximum number of new tokens to generate.
            context_window (int): Context window size.
            messages_to_prompt (Optional[Callable]): Function for prompting messages.
            completion_to_prompt (Optional[Callable]): Function for prompting completions.
            callback_manager (Optional[CallbackManager]): Manager for callbacks.
            generate_kwargs (Optional[Dict[str, Any]]): Additional keyword arguments for generation.
            model_kwargs (Optional[Dict[str, Any]]): Additional keyword arguments for the model.
            verbose (bool): Verbosity flag.

        Raises:
            ValueError: If the provided model path does not exist.

        Note:
            The function initializes the object with the provided parameters and sets up the model configuration, tokenizer, and sampling configuration.
        """


        model_kwargs = model_kwargs or {}
        model_kwargs.update({"n_ctx": context_window, "verbose": verbose})
        self._max_new_tokens = max_new_tokens
        self._verbose = verbose
        # check if model is cached
        if model_path is not None:
            if not os.path.exists(model_path):
                raise ValueError(
                    "Provided model path does not exist. "
                    "Please check the path or provide a model_url to download."
                )
            else:
                engine_dir = model_path
                engine_dir_path = Path(engine_dir)
                config_path = engine_dir_path / 'config.json'

                # config function
                with open(config_path, 'r') as f:
                    config = json.load(f)
                use_gpt_attention_plugin = config['plugin_config']['gpt_attention_plugin']
                remove_input_padding = config['plugin_config']['remove_input_padding']
                tp_size = config['builder_config']['tensor_parallel']
                pp_size = config['builder_config']['pipeline_parallel']
                world_size = tp_size * pp_size
                assert world_size == tensorrt_llm.mpi_world_size(), \
                    f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'
                num_heads = config['builder_config']['num_heads'] // tp_size
                hidden_size = config['builder_config']['hidden_size'] // tp_size
                vocab_size = config['builder_config']['vocab_size']
                num_layers = config['builder_config']['num_layers']
                num_kv_heads = config['builder_config'].get('num_kv_heads', num_heads)
                paged_kv_cache = config['plugin_config']['paged_kv_cache']
                if config['builder_config'].get('multi_query_mode', False):
                    tensorrt_llm.logger.warning(
                        "`multi_query_mode` config is deprecated. Please rebuild the engine."
                    )
                    num_kv_heads = 1
                num_kv_heads = (num_kv_heads + tp_size - 1) // tp_size

                self._model_config = ModelConfig(num_heads=num_heads,
                                                 num_kv_heads=num_kv_heads,
                                                 hidden_size=hidden_size,
                                                 vocab_size=vocab_size,
                                                 num_layers=num_layers,
                                                 gpt_attention_plugin=use_gpt_attention_plugin,
                                                 paged_kv_cache=paged_kv_cache,
                                                 remove_input_padding=remove_input_padding)

                assert pp_size == 1, 'Python runtime does not support pipeline parallelism'
                world_size = tp_size * pp_size

                runtime_rank = tensorrt_llm.mpi_rank()
                runtime_mapping = tensorrt_llm.Mapping(world_size,
                                                       runtime_rank,
                                                       tp_size=tp_size,
                                                       pp_size=pp_size)
                torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)
                self._tokenizer = LlamaTokenizer.from_pretrained(tokenizer_dir, legacy=False)
                self._sampling_config = SamplingConfig(end_id=EOS_TOKEN,
                                                       pad_id=PAD_TOKEN,
                                                       num_beams=1,
                                                       temperature=temperature)

                serialize_path = engine_dir_path / engine_name
                with open(serialize_path, 'rb') as f:
                    engine_buffer = f.read()
                decoder = tensorrt_llm.runtime.GenerationSession(self._model_config,
                                                                 engine_buffer,
                                                                 runtime_mapping,
                                                                 debug_mode=False)
                self._model = decoder
        messages_to_prompt = messages_to_prompt or generic_messages_to_prompt
        completion_to_prompt = completion_to_prompt or (lambda x: x)

        generate_kwargs = generate_kwargs or {}
        generate_kwargs.update(
            {"temperature": temperature, "max_tokens": max_new_tokens}
        )

        super().__init__(
            model_path=model_path,
            temperature=temperature,
            context_window=context_window,
            max_new_tokens=max_new_tokens,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            callback_manager=callback_manager,
            generate_kwargs=generate_kwargs,
            model_kwargs=model_kwargs,
            verbose=verbose,
        )

    @classmethod
    def class_name(cls) -> str:
        """        Get the name of the class.

        This function returns the name of the class as a string.

        Returns:
            str: The name of the class.
        """
        return "TrtLlmAPI"

    @property
    def metadata(self) -> LLMMetadata:
        """        Return LLM metadata.

        Returns:
            LLMMetadata: An instance of LLMMetadata containing context_window, num_output, and model_name.
        """
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_new_tokens,
            model_name=self.model_path,
        )

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """        Generate a chat response based on the input messages.

        Args:
            messages (Sequence[ChatMessage]): The input messages for the chat.
            **kwargs (Any): Additional keyword arguments for customization.

        Returns:
            ChatResponse: The response generated based on the input messages.
                This function takes a sequence of ChatMessage objects as input and generates a chat response based on these messages. It uses the messages to prompt the chat, completes the prompt, and then converts the completion response to a ChatResponse.
        """

        prompt = self.messages_to_prompt(messages)
        completion_response = self.complete(prompt, formatted=True, **kwargs)
        return completion_response_to_chat_response(completion_response)

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """        Generate completion for the given prompt.

        Args:
            prompt (str): The input prompt for which completion needs to be generated.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            CompletionResponse: The response object containing the completion text and raw completion dictionary.
        """

        self.generate_kwargs.update({"stream": False})

        is_formatted = kwargs.pop("formatted", False)
        if not is_formatted:
            prompt = self.completion_to_prompt(prompt)

        input_text = prompt
        input_ids, input_lengths = self.parse_input(input_text, self._tokenizer,
                                                    EOS_TOKEN,
                                                    self._model_config)

        max_input_length = torch.max(input_lengths).item()
        self._model.setup(input_lengths.size(0), max_input_length, self._max_new_tokens, 1) # beam size is set to 1
        if self._verbose:
            start_time = time.time()

        output_ids = self._model.decode(input_ids, input_lengths, self._sampling_config)
        torch.cuda.synchronize()

        elapsed_time = None
        if self._verbose:
            end_time = time.time()
            elapsed_time = end_time - start_time


        output_txt, output_token_ids = self.get_output(output_ids,
                                       input_lengths,
                                       self._max_new_tokens,
                                       self._tokenizer)

        if self._verbose:
            print(f"Input context length  : {input_ids.shape[1]}")
            print(f"Inference time        : {elapsed_time:.2f} seconds")
            print(f"Output context length : {len(output_token_ids)} ")
            print(f"Inference token/sec   : {(len(output_token_ids) / elapsed_time):2f}")

        # call garbage collected after inference
        torch.cuda.empty_cache()
        gc.collect()

        return CompletionResponse(text=output_txt, raw=self.generate_completion_dict(output_txt))

    def parse_input(self, input_text: str, tokenizer, end_id: int,
                    remove_input_padding: bool):
        """        Parse the input text using the provided tokenizer and return the input ids and input lengths.

        Args:
            input_text (str): The input text to be tokenized.
            tokenizer: The tokenizer object to encode the input text.
            end_id (int): The end id for padding the input tokens.
            remove_input_padding (bool): A flag indicating whether to remove input padding.

        Returns:
            tuple: A tuple containing the input ids and input lengths.
        """

        input_tokens = []

        input_tokens.append(
            tokenizer.encode(input_text, add_special_tokens=False))

        input_lengths = torch.tensor([len(x) for x in input_tokens],
                                     dtype=torch.int32,
                                     device='cuda')
        if remove_input_padding:
            input_ids = np.concatenate(input_tokens)
            input_ids = torch.tensor(input_ids, dtype=torch.int32,
                                     device='cuda').unsqueeze(0)
        else:
            input_ids = torch.nested.to_padded_tensor(
                torch.nested.nested_tensor(input_tokens, dtype=torch.int32),
                end_id).cuda()

        return input_ids, input_lengths

    def remove_extra_eos_ids(self, outputs):
        """        Remove extra end-of-sequence (EOS) IDs from the outputs.

        This function reverses the 'outputs' list, removes any leading EOS IDs (value 2), and then reverses the list back.
        Finally, it appends an EOS ID to the end of the 'outputs' list.

        Args:
            outputs (list): The list of output IDs.

        Returns:
            list: The modified 'outputs' list with extra EOS IDs removed and an additional EOS ID appended.
        """

        outputs.reverse()
        while outputs and outputs[0] == 2:
            outputs.pop(0)
        outputs.reverse()
        outputs.append(2)
        return outputs

    def get_output(self, output_ids, input_lengths, max_output_len, tokenizer):
        """        Generate the output text based on the given output_ids, input_lengths, max_output_len, and tokenizer.

        Args:
            output_ids (Tensor): The output ids generated by the model.
            input_lengths (Tensor): The lengths of the input sequences.
            max_output_len (int): The maximum length of the output text.
            tokenizer (Tokenizer): The tokenizer used to decode the output ids.

        Returns:
            output_text (str): The decoded output text.
            outputs (list): The list of output ids after removing extra eos ids.
            This function iterates through the input lengths and generates the output text based on the given parameters.
            It decodes the output ids using the provided tokenizer and returns the decoded output text along with the list of output ids.
        """

        num_beams = output_ids.size(1)
        output_text = ""
        outputs = None
        for b in range(input_lengths.size(0)):
            for beam in range(num_beams):
                output_begin = input_lengths[b]
                output_end = input_lengths[b] + max_output_len
                outputs = output_ids[b][beam][output_begin:output_end].tolist()
                outputs = self.remove_extra_eos_ids(outputs)
                output_text = tokenizer.decode(outputs)

        return output_text, outputs

    def generate_completion_dict(self, text_str):
        """        Generate a dictionary for text completion details.

        Args:
            text_str (str): The input text for which completion details are to be generated.

        Returns:
            dict: A dictionary containing completion details, including completion ID, creation time, model name, and text choices.
        """
        completion_id: str = f"cmpl-{str(uuid.uuid4())}"
        created: int = int(time.time())
        model_name: str = self._model if self._model is not None else self.model_path
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

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """        Complete the stream based on the given prompt.

        Args:
            prompt (str): The prompt for which the stream needs to be completed.
            **kwargs: Additional keyword arguments.

        Returns:
            CompletionResponse: The response containing the completed stream.
        """

        pass
