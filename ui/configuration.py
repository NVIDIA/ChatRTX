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

import json

class Configuration:

    default_config_path = "./config/config.json"
    modified_config_path = "./config/preferences.json"
    _deafult_config = None

    def __init__(self) -> None:
        default_config = self._read_json_file(self.default_config_path)
        self._deafult_config = default_config
        self.modified_config = self._read_json_file(self.modified_config_path)
        self.merged_config = self._merge_objects(default_config, self.modified_config)
        self._display_strings = self.get_config("strings")
        self._display_strings_rev = {}
        for key, val in self._display_strings.items():
            if isinstance(val, str):
                self._display_strings_rev[val] = key
        pass

    def get_config_from_file(self, key: str, file: str):
        if file is None or len(file) == 0:
            return None
        
        config = self._read_json_file(file)
        if key is None or len(key) == 0:
            return None
        keys = key.split('/')
        for k in keys:
            if k in config:
                config = config[k]
            else:
                config = None
                break
        return config

    def get_config(self, key: str):
        if key is None or len(key) == 0:
            return self.merged_config
        keys = key.split('/')
        root = self.merged_config
        for k in keys:
            if k in root:
                root = root[k]
            else:
                root = None
                break
        return root

    def set_config(self, key: str, value: any):
        keys = key.split('/')
        current_level = self.modified_config

        for k in keys[:-1]:
            current_level = current_level.setdefault(k, {})

        current_level[keys[-1]] = value
        self.merged_config = self._merge_objects(self.merged_config, self.modified_config)
        self._write_json_to_file(self.modified_config, self.modified_config_path)

    def _write_json_to_file(self, json_object, file_path):
        with open(file_path, 'w') as file:
            json.dump(json_object, file, indent=2)

    def _read_json_file(self, file_path):
        retVal = {}
        try:
            with open(file_path, 'r', encoding='utf8') as file:
                data = json.load(file)
                retVal = data if data else {}  # Return empty JSON object if file is empty
        except json.JSONDecodeError as e:
            print(f"The file at {file_path} is not a valid JSON file. Error: {e}")
        except FileNotFoundError:
            print(f"The file at {file_path} does not exist.")
        except Exception as e:
            print(f"error reading {file_path}. Error: {e}")

        return retVal
        
    def _merge_objects(self, obj1, obj2):
        if not isinstance(obj1, dict) or not isinstance(obj2, dict):
            return obj2  # If either is not a dictionary, return the second object

        merged = obj1.copy()

        for key, value in obj2.items():
            if key in merged and isinstance(merged[key], dict):
                # Recursive merge if the value is another dictionary
                merged[key] = self._merge_objects(merged[key], value)
            else:
                merged[key] = value

        return merged
    
    def get_display_strings(self, key):
        if isinstance(key, list):
            return [self.get_display_strings(_key) for _key in key]                
        
        if isinstance(key, str):
            if key in self._display_strings:
                return self._display_strings[key]
            else:
                return key
        
        return key
    
    def get_display_string_keys(self, value):
        if isinstance(value, list):
            return [self.get_display_string_keys(_value) for _value in value]

        if isinstance(value, str):
            if value in self._display_strings_rev:
                return self._display_strings_rev[value]
            else:
                return value
        
        return value

    def read_default_config(self, key):
        if key is None or len(key) == 0:
            return self._deafult_config
        keys = key.split('/')
        root = self._deafult_config
        for k in keys:
            if k in root:
                root = root[k]
            else:
                root = None
                break
        return root


    def write_default_config(self, key:str, value: any):
        keys = key.split('/')
        current_level = self._deafult_config
        for k in keys[:-1]:
            current_level = current_level.setdefault(k, {})
        current_level[keys[-1]] = value
        self._write_json_to_file(self._deafult_config, self.default_config_path)
