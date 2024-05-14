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

from typing import Any, Callable, Dict, Optional, Sequence
from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.callbacks import CallbackManager
from llama_index.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.llms.base import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    ChatResponseGen,
    CompletionResponseGen,
    LLMMetadata,
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.llms.custom import CustomLLM
from llama_index.llms.generic_utils import stream_completion_response_to_chat_response
from llama_index.llms.generic_utils import completion_response_to_chat_response
from llama_index.llms.generic_utils import (
    messages_to_prompt as generic_messages_to_prompt,
)
from utils import (DEFAULT_HF_MODEL_DIRS, DEFAULT_PROMPT_TEMPLATES,
                   load_tokenizer, read_model_name, throttle_generator)
import gc
import torch
import tensorrt_llm
import uuid
import time
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner, ModelRunnerCpp
from tensorrt_llm.logger import logger
from llm_prompt_templates import LLMPromptTemplate

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
    _model_name = PrivateAttr()
    _model_version = PrivateAttr()
    _model_config: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _pad_id:Any = PrivateAttr()
    _end_id: Any = PrivateAttr()
    _max_new_tokens = PrivateAttr()
    _max_input_tokens = PrivateAttr()
    _sampling_config = PrivateAttr()
    _debug_mode = PrivateAttr()
    _add_special_tokens = PrivateAttr()
    _verbose = PrivateAttr()

    def __init__(
            self,
            model_path: Optional[str] = None,
            engine_name: Optional[str] = None,
            tokenizer_dir: Optional[str] = None,
            vocab_file: Optional[str] = None,
            temperature: float = 0.1,
            max_new_tokens: int = DEFAULT_NUM_OUTPUTS,
            context_window: int = DEFAULT_CONTEXT_WINDOW,
            messages_to_prompt: Optional[Callable] = None,
            completion_to_prompt: Optional[Callable] = None,
            prompt_template = None,
            callback_manager: Optional[CallbackManager] = None,
            generate_kwargs: Optional[Dict[str, Any]] = None,
            model_kwargs: Optional[Dict[str, Any]] = None,
            use_py_session = True,
            add_special_tokens = False,
            trtLlm_debug_mode = False,
            verbose: bool = False
    ) -> None:
        runtime_rank = tensorrt_llm.mpi_rank()
        self._model_name, self._model_version = read_model_name(model_path)
        if tokenizer_dir is None:
            logger.warning(
                "tokenizer_dir is not specified. Try to infer from model_name, but this may be incorrect."
            )

            if self._model_name == "GemmaForCausalLM":
                tokenizer_dir = 'gpt2'
            else:
                tokenizer_dir = DEFAULT_HF_MODEL_DIRS[self._model_name]

        self._max_input_tokens=context_window
        self._add_special_tokens=add_special_tokens
        self._verbose = verbose
        model_kwargs = model_kwargs or {}
        model_kwargs.update({"n_ctx": context_window, "verbose": verbose})
        #logger.set_level('verbose')
        self._tokenizer, self._pad_id, self._end_id = load_tokenizer(
            tokenizer_dir=tokenizer_dir,
            vocab_file=vocab_file,
            model_name=self._model_name,
            model_version=self._model_version,
            #tokenizer_type=args.tokenizer_type,
        )

        runner_cls = ModelRunner if use_py_session else ModelRunnerCpp
        if verbose:
            print(f"[ChatRTX] Trt-llm mode debug mode: {trtLlm_debug_mode}")

        runner_kwargs = dict(engine_dir=model_path,
                             rank=runtime_rank,
                             debug_mode=trtLlm_debug_mode,
                             lora_ckpt_source='hf')

        if not use_py_session:
            runner_kwargs.update(free_gpu_memory_fraction = 0.5)

        self._model = runner_cls.from_dir(**runner_kwargs)
        generate_kwargs = generate_kwargs or {}
        generate_kwargs.update(
           {"temperature": temperature, "max_tokens": max_new_tokens}
        )

        self._max_new_tokens = max_new_tokens

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
        """Get class name."""
        return "TrtLlmAPI"

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_new_tokens,
            model_name=self.model_path,
        )

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = self.messages_to_prompt(messages)
        completion_response = self.complete(prompt, formatted=True, **kwargs)
        return completion_response_to_chat_response(completion_response)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        prompt = self.messages_to_prompt(messages)
        completion_response = self.stream_complete(prompt, formatted=True, **kwargs)
        return stream_completion_response_to_chat_response(completion_response)

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        self.generate_kwargs.update({"stream": False})
        is_formatted = kwargs.pop("formatted", False)
        if not is_formatted:
            prompt_template = LLMPromptTemplate()
            prompt = prompt_template.model_default_template(model=self._model_name,query=prompt)

        if self._verbose:
            print(f"[ChatRTX] Context send to LLM \n: {prompt}")

        input_text = [prompt]
        batch_input_ids = self.parse_input(
                                tokenizer=self._tokenizer,
                                input_text=input_text,
                                prompt_template=None,
                                input_file=None,
                                add_special_tokens=self._add_special_tokens,
                                max_input_length=self._max_input_tokens,
                                pad_id=self._pad_id,
                                num_prepend_vtokens=None,
                                model_name= self._model_name,
                                model_version=self._model_version)
        input_lengths = [x.size(0) for x in batch_input_ids]

        if self._verbose:
            print(f"[ChatRTX] Number of token : {input_lengths[0]}")

        with torch.no_grad():
            outputs = self._model.generate(
                batch_input_ids,
                max_new_tokens=self._max_new_tokens,
                max_attention_window_size=4096,
                #sink_token_length=None,
                end_id=self._end_id,
                pad_id=self._pad_id,
                temperature=1.0,
                top_k=1,
                top_p=0,
                num_beams=1,
                length_penalty=1.0,
                early_stopping=False,
                repetition_penalty=1.0,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                stop_words_list=None,
                bad_words_list=None,
                lora_uids=None,
                prompt_table_path=None,
                prompt_tasks=None,
                streaming=False,
                output_sequence_lengths=True,
                return_dict=True)
            torch.cuda.synchronize()

        output_ids = outputs['output_ids']
        sequence_lengths = outputs['sequence_lengths']
        output_txt, output_token_ids = self.print_output(self._tokenizer,
                                                        output_ids,
                                                        input_lengths,
                                                        sequence_lengths)
        # call garbage collected after inference
        torch.cuda.empty_cache()
        gc.collect()
        return CompletionResponse(text=output_txt, raw=self.generate_completion_dict(output_txt))

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        is_formatted = kwargs.pop("formatted", False)
        if not is_formatted:
            prompt_template = LLMPromptTemplate()
            prompt = prompt_template.model_default_template(model=self._model_name,query=prompt)
        if self._verbose:
            print(prompt)
        input_text = [prompt]

        batch_input_ids = self.parse_input(
                                tokenizer=self._tokenizer,
                                input_text=input_text,
                                prompt_template=None,
                                input_file=None,
                                add_special_tokens=self._add_special_tokens,
                                max_input_length=self._max_input_tokens,
                                pad_id=self._pad_id,
                                num_prepend_vtokens=None,
                                model_name= self._model_name,
                                model_version=self._model_version)
        input_lengths = [x.size(0) for x in batch_input_ids]

        with torch.no_grad():
            outputs = self._model.generate(
                batch_input_ids,
                max_new_tokens=self._max_new_tokens,
                max_attention_window_size=4096,
                sink_token_length=None,
                end_id=self._end_id,
                pad_id=self._pad_id,
                temperature=1.0,
                top_k=1,
                top_p=0,
                num_beams=1,
                length_penalty=1.0,
                early_stopping=True,
                repetition_penalty=1.0,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                stop_words_list=None,
                bad_words_list=None,
                lora_uids=None,
                prompt_table_path=None,
                prompt_tasks=None,
                streaming=True,
                output_sequence_lengths=True,
                return_dict=True)
            torch.cuda.synchronize()
        previous_text = ""  # To keep track of the previously yielded text

        def gen() -> CompletionResponseGen:
            nonlocal previous_text  # Declare previous_text as nonlocal
            for curr_outputs in throttle_generator(outputs,
                                                   5):
                output_ids = curr_outputs['output_ids']
                sequence_lengths = curr_outputs['sequence_lengths']
                output_txt, output_token_ids = self.print_output(self._tokenizer,
                                                                 output_ids,
                                                                 input_lengths,
                                                                 sequence_lengths)
                torch.cuda.synchronize()
                if output_txt.endswith("</s>"):
                    output_txt = output_txt[:-4]
                pre_token_len = len(previous_text)
                new_text = output_txt[pre_token_len:]  # Get only the new text

                yield CompletionResponse(delta=new_text, text=output_txt,
                                         raw=self.generate_completion_dict(output_txt))
                previous_text = output_txt  # Update the previously yielded text after yielding
        return gen()

    def parse_input(self,
                    tokenizer,
                    input_text=None,
                    prompt_template=None,
                    input_file=None,
                    add_special_tokens=False,
                    max_input_length=4096,
                    pad_id=None,
                    num_prepend_vtokens=[],
                    model_name=None,
                    model_version=None):
        if pad_id is None:
            pad_id = tokenizer.pad_token_id
        if model_name == 'GemmaForCausalLM':
            add_special_tokens=True
        batch_input_ids = []
        if input_file is None:
            for curr_text in input_text:
                if prompt_template is not None:
                    curr_text = prompt_template.format(input_text=curr_text)
                input_ids = tokenizer.encode(curr_text,
                                             add_special_tokens=add_special_tokens,
                                             truncation=True,
                                             max_length=max_input_length)
                batch_input_ids.append(input_ids)

        if num_prepend_vtokens:
            assert len(num_prepend_vtokens) == len(batch_input_ids)
            base_vocab_size = tokenizer.vocab_size - len(
                tokenizer.special_tokens_map.get('additional_special_tokens', []))
            for i, length in enumerate(num_prepend_vtokens):
                batch_input_ids[i] = list(
                    range(base_vocab_size,
                          base_vocab_size + length)) + batch_input_ids[i]

        if model_name == 'ChatGLMForCausalLM' and model_version == 'glm':
            for ids in batch_input_ids:
                ids.append(tokenizer.sop_token_id)

        batch_input_ids = [
            torch.tensor(x, dtype=torch.int32) for x in batch_input_ids
        ]

        return batch_input_ids

    def remove_extra_eos_ids(self, outputs):
        outputs.reverse()
        while outputs and outputs[0] == 2:
            outputs.pop(0)
        outputs.reverse()
        outputs.append(2)
        return outputs

    def print_output(self,
                     tokenizer,
                     output_ids,
                     input_lengths,
                     sequence_lengths,
                     output_csv=None,
                     output_npy=None,
                     context_logits=None,
                     generation_logits=None,
                     output_logits_npy=None):
        output_text = ""
        batch_size, num_beams, _ = output_ids.size()
        if output_csv is None and output_npy is None:
            for batch_idx in range(batch_size):
                inputs = output_ids[batch_idx][0][:input_lengths[batch_idx]].tolist(
                )
                for beam in range(num_beams):
                    output_begin = input_lengths[batch_idx]
                    output_end = sequence_lengths[batch_idx][beam]
                    outputs = output_ids[batch_idx][beam][
                              output_begin:output_end].tolist()
                    output_text = tokenizer.decode(outputs)

        output_ids = output_ids.reshape((-1, output_ids.size(2)))
        return output_text, output_ids

    def get_output(self, output_ids, input_lengths, max_output_len, tokenizer):
        num_beams = 1
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
        """
        Generate a dictionary for text completion details.
        Returns:
        dict: A dictionary containing completion details.
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

    def unload_model(self):
        if self._model is not None:
            del self._model
        # Step 3: Additional cleanup if needed
        torch.cuda.empty_cache()
        gc.collect()

    def get_model_name(self):
        return  self._model_name

