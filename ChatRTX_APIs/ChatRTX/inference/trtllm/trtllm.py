# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import torch
import tensorrt_llm
from typing import Any, Optional
from tensorrt_llm.runtime import ModelRunner, ModelRunnerCpp
from tensorrt_llm.logger import logger
from ChatRTX.inference.trtllm.utils import (DEFAULT_HF_MODEL_DIRS, load_tokenizer, read_model_name, throttle_generator)
from ChatRTX.logger import ChatRTXLogger

class TrtLlm():
    """
    A class to manage and interact with a TensorRT LLM model, providing functions to load,
    parse input, and generate text completions.
    """
    DEFAULT_TEMPERATURE = 0.1
    DEFAULT_MAX_NEW_TOKENS = 100
    DEFAULT_CONTEXT_WINDOW = 2048

    def __init__(
            self,
            model_path: Optional[str] = None,
            tokenizer_dir: Optional[str] = None,
            vocab_file: Optional[str] = None,
            temperature: float = 0.1,
            max_new_tokens: int = 100,  # Your default value for max_new_tokens
            context_window: int = 2048,  # Your default value for context_window
            use_py_session=True,
            add_special_tokens=False,
            trtLlm_debug_mode=False
    ) -> None:
        self._model_name, self._model_version = read_model_name(model_path)
        self._max_input_tokens=context_window
        self._add_special_tokens=add_special_tokens
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._logger = ChatRTXLogger.get_logger()
        try:
            if tokenizer_dir is None:
                logger.warning(
                    "tokenizer_dir is not specified. Try to infer from model_name, but this may be incorrect."
                )

                if self._model_name == "GemmaForCausalLM":
                    tokenizer_dir = 'gpt2'
                else:
                    tokenizer_dir = DEFAULT_HF_MODEL_DIRS[self._model_name]

            self._tokenizer, self._pad_id, self._end_id = load_tokenizer(
                tokenizer_dir=tokenizer_dir,
                vocab_file=vocab_file,
                model_name=self._model_name,
                model_version=self._model_version,
                #tokenizer_type=args.tokenizer_type,
            )

            runner_cls = ModelRunner if use_py_session else ModelRunnerCpp

            self._logger.debug(f"Trt-llm mode debug mode: {trtLlm_debug_mode}")

            runtime_rank = tensorrt_llm.mpi_rank()
            runner_kwargs = dict(engine_dir=model_path,
                                 rank=runtime_rank,
                                 debug_mode=trtLlm_debug_mode,
                                 lora_ckpt_source='hf')
            if not use_py_session:
                runner_kwargs.update(free_gpu_memory_fraction = 0.5)
            self._model = runner_cls.from_dir(**runner_kwargs)
        except Exception as e:
            self._logger.error(f"Fail to create TRT-LLM object for model: {model_path}. \n Error: {str(e)}")
            raise Exception(f"Fail to create TRT-LLM object for model: {model_path}. \n Error: {str(e)}")

    def get_model_name(self):
        if self._model is not None:
            return self._model_name
        else:
            return None

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "TrtLlm"

    def print_output(self, tokenizer, output_ids, input_lengths, sequence_lengths):
        """
        Processes the model output to convert output_ids to human-readable text.

        Args:
            tokenizer: The tokenizer used to decode the output ids.
            output_ids: Tensor containing output token ids from the model.
            input_lengths: List of input lengths to determine the start of the output in output_ids.
            sequence_lengths: List of actual lengths of each output sequence to determine end of output.

        Returns:
            output_text (str): Decoded text of the first beam.
            output_ids (tensor): Reshaped output ids.
        """
        output_text = ""
        batch_size, num_beams, _ = output_ids.size()
        for batch_idx in range(batch_size):
            for beam in range(num_beams):
                output_begin = input_lengths[batch_idx]
                output_end = sequence_lengths[batch_idx][beam]
                outputs = output_ids[batch_idx][beam][output_begin:output_end].tolist()
                output_text = tokenizer.decode(outputs)

        output_ids = output_ids.reshape((-1, output_ids.size(2)))
        return output_text, output_ids

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

    def complete(self, prompt: str):
        """
        if not is_formatted:
            prompt_template = LLMPromptTemplate()
            prompt = prompt_template.model_default_template(model=self._model_name,query=prompt)
        """
        try:
            self._logger.debug(f"Prompt send to LLM \n: {prompt}")
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

            self._logger.debug(f"Number of token : {input_lengths[0]}")

            with torch.no_grad():
                outputs = self._model.generate(
                    batch_input_ids,
                    max_new_tokens=self._max_new_tokens,
                    max_attention_window_size=4096,
                    #sink_token_length=None,
                    end_id=self._end_id,
                    pad_id=self._pad_id,
                    temperature=self._temperature,
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
            torch.cuda.empty_cache()
            gc.collect()
            return output_txt
        except Exception as e:
            self._logger.error(f"Fail to generate response for promt {prompt}. \n Error: {str(e)}")
            raise Exception(f"Fail to generate response for promt {prompt}. \n Error: {str(e)}")

    def stream_complete(self, prompt: str, **kwargs: Any):
        self._logger.debug(f"Prompt send to LLM \n: {prompt}")
        input_text = [prompt]
        try:
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
            self._logger.debug(f"Number of token : {input_lengths[0]}")

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

            def gen():
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
                    yield new_text
                    previous_text = output_txt  # Update the previously yielded text after yielding
            return gen()
        except Exception as e:
            self._logger.error(f"Fail to generate stream response for promt {prompt}. \n Error: {str(e)}")
            raise Exception(f"Fail to generate stream  response for promt {prompt}. \n Error: {str(e)}")

    def unload_llm(self):
        try:
            if self is not None:
                del self._model
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            self._logger.error(f"Fail to unload the model. \n Error: {str(e)}")