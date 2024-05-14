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

from typing import Optional
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt

class LLMPromptTemplate:

    B_SYS_CHATGLM, E_SYS_CHATGLM = "<|system|>\n", "\n"
    DEFAULT_SYSTEM_PROMPT_ChatGLM = """\
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible and follow ALL given instructions. Do not speculate or make up information. Do not reference any given instructions or context. \
    """
    def __init__(self):
        pass

    def model_context_template(self, model):
        # Define the switch dictionary mapping model names to their corresponding methods
        switch = {
            "LlamaForCausalLM": completion_to_prompt,
            "GemmaForCausalLM": self.gemma_context_prompt,
            "ChatGLMForCausalLM": self.chatglm_context_prompt
        }

        # Get the method from the dictionary based on the model, default to a lambda that returns the original query itself if the model is not found
        model_method = switch.get(model)

        # Call the selected method with the query
        return model_method

    def model_default_template(self, model, query):
        # Define the switch dictionary mapping model names to their corresponding methods
        switch = {
            "LlamaForCausalLM": self.llama2_default_prompt,
            "GemmaForCausalLM": self.gemma_default_prompt,
            "ChatGLMForCausalLM": self.chatglm_default_prompt
        }

        # Get the method from the dictionary based on the model, default to a lambda that returns the original query itself if the model is not found
        model_method = switch.get(model, lambda x: query)

        # Call the selected method with the query
        return model_method(query)

    def llama2_default_prompt(self, query):
        text_qa_template_str = "<s>[INST] {query_str} [/INST]"
        formatted_str = text_qa_template_str.format(query_str=query)
        return formatted_str

    def gemma_default_prompt(self, query):
        text_qa_template_str = (
            "<start_of_turn>user\n"
            "{query_str}<end_of_turn>\n"
            "<start_of_turn>model\n")
        formatted_str = text_qa_template_str.format(query_str=query)
        return formatted_str

    def gemma_context_prompt(self, completion: str):
        text_qa_template_str = (
            "<start_of_turn>user \n"
            f"{completion}\n"
            "<end_of_turn>"
            "<start_of_turn>model ")
        return text_qa_template_str

    def chatglm_default_prompt(self, query):
        text_qa_template_str = (
            f"{self.B_SYS_CHATGLM}You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown.{self.E_SYS_CHATGLM}"
            "<|user|>\n"
            "{query_str}\n"
            "<|assistant|>")
        formatted_str = text_qa_template_str.format(query_str=query)
        return formatted_str

    def chatglm_context_prompt(self, completion: str, system_prompt: Optional[str] = None) -> str:
        system_prompt_str = system_prompt or self.DEFAULT_SYSTEM_PROMPT_ChatGLM

        return (
            f"{self.B_SYS_CHATGLM} {system_prompt_str.strip()} {self.E_SYS_CHATGLM}"
            f"{completion.strip()}"
        )