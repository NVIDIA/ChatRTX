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

import logging
import logging.handlers
import os

class ChatRTXLogger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ChatRTXLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self, log_level=logging.DEBUG, log_file=None, log_format=None, max_bytes=10485760, backup_count=5):
        if hasattr(self, '_initialized') and self._initialized:
            return

        self._initialized = True

        if log_format is None:
            log_format = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'

        self.logger = logging.getLogger("[ChatRTX]")
        self.logger.setLevel(log_level)

        formatter = logging.Formatter(log_format)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler
        if log_file:

            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            # Check if the file exists, and create it if it does not
            if not os.path.exists(log_file):
                open(log_file, 'w').close()

            file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    @staticmethod
    def get_logger():
        return ChatRTXLogger().logger

    @staticmethod
    def set_log_level(level):
        logger = ChatRTXLogger.get_logger()
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)

    @staticmethod
    def set_verbose_mode(enable):
        level = logging.DEBUG if enable else logging.INFO
        ChatRTXLogger.set_log_level(level)