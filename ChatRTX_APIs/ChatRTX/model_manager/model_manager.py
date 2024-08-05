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
import logging
import os.path
import shutil
from ChatRTX.model_manager.model_manager_util import download_model_by_name, build_engine_by_name, verify_clip_checksum
from ChatRTX.model_manager.verify_model_install import update_config
from ChatRTX.logger import ChatRTXLogger
from ChatRTX.model_manager.config import Config

class ModelManager:
    """
    Handles the management of model configurations, including downloading and installing models.
    """

    # Configuration keys and model directory constants
    CONFIG_KEY = "models"
    SUPPORTED_KEY = "supported"
    MODEL_DIR = "model"

    def __init__(self, models_dir, config_path="../config/config.json", sample_data = "../sample_data"):
        """
        Initializes the ModelManager class.

        Args:
            models_dir (str): The directory where models are stored.
            config_path (str): The path to the configuration file.
        """
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        base_config = os.path.join(current_file_dir, "../config/config.json")
        app_config_dir = os.path.join(models_dir, "config")
        app_config_file = os.path.join(app_config_dir, "config.json")

        # Check if app_config_dir exists, if not, create it
        if not os.path.exists(app_config_dir):
            os.makedirs(app_config_dir)
            print(f"Created directory: {app_config_dir}")

        # Check if config.json exists in app_config_dir, if not, copy it from base_config
        if not os.path.exists(app_config_file):
            shutil.copy(base_config, app_config_file)
            print(f"Copied {base_config} to {app_config_file}")

        # check for the sample data
        base_sample_data_dir = os.path.join(current_file_dir, sample_data)
        app_sample_data_dir  = os.path.join(models_dir, "sample_data")

        # Check if app_sample_data_dir exists, if not, create it
        try:
            # Copy the contents of the sample data directory recursively
            if os.path.exists(base_sample_data_dir):
                for item in os.listdir(base_sample_data_dir):
                    s = os.path.join(base_sample_data_dir, item)
                    d = os.path.join(app_sample_data_dir, item)
                    if os.path.isdir(s):
                        shutil.copytree(s, d, dirs_exist_ok=True)
                    else:
                        shutil.copy2(s, d)
                print(f"Copied contents of {base_sample_data_dir} to {app_sample_data_dir}")
        except Exception as e:
            print(f"Eception {e}")

        self._model_directory = os.path.join(models_dir, "models")
        if not os.path.exists(self._model_directory):
            os.makedirs(self._model_directory)

        self.config_path = app_config_file
        update_config(self._model_directory, self.config_path)
        self.config = Config(self.config_path)

        # Expand the paths in the config files
        keys = ['sample_questions/default/dataset_path', 'sample_questions/chinese/dataset_path', 'sample_questions/images/dataset_path',
                'dataset/path', 'dataset/path_chinese', 'dataset/path_clip', 'dataset/selected_path']

        for key in keys:
            expanded_path = self.expand_programdata_path(self.config.get_config(key))
            self.config.write_default_config(key, expanded_path)
        ChatRTXLogger(log_level=logging.DEBUG, log_file='chatRTX.log')
        self._logger = ChatRTXLogger.get_logger()

    def expand_programdata_path(self, path):
        """
        Expands the %programdata% environment variable in the given path.

        Parameters:
        path (str): The path containing the %programdata% variable.

        Returns:
        str: The path with the %programfiles% variable expanded.
        """
        expanded_path = os.path.expandvars(path)
        return expanded_path

    def get_model_info(self):
        """
        Retrieves detailed information about all supported models.

        Returns:
            list: A list of dictionaries containing detailed information about each model.
        """
        try:
            model_info_list = self.config.get_config('models/supported')
            if not model_info_list:
                self._logger.error("No supported models found in the configuration.")
                return []

            detailed_info_list = []
            for model_info in model_info_list:
                detailed_info = {
                    "name": model_info.get("name", "Unknown"),
                    "id": model_info.get("id", "Unknown"),
                    "ngc_model_name": model_info.get("ngc_model_name", "Unknown"),
                    "is_downloaded_required": model_info.get("is_downloaded_required", False),
                    "downloaded": model_info.get("downloaded", False),
                    "is_installation_required": model_info.get("is_installation_required", False),
                    "setup_finished": model_info.get("setup_finished", False),
                    "min_gpu_memory": model_info.get("min_gpu_memory", "Unknown"),
                    "should_show_in_UI": model_info.get("should_show_in_UI", False),
                    "isFineTuningSupported": model_info.get("isFineTuningSupported", False),
                    "prerequisite": model_info.get("prerequisite", {}),
                    "metadata": model_info.get("metadata", {}),
                    "model_info": model_info.get("model_info", "No information available"),
                    "model_license": model_info.get("model_license", "No license information available"),
                    "model_size": model_info.get("model_size", "Unknown")
                }
                detailed_info_list.append(detailed_info)

            return detailed_info_list
        except KeyError as e:
            self._logger.error(f"Failed to retrieve model information. Key error: {str(e)}")
            return []
        except Exception as e:
            self._logger.error(f"An unexpected error occurred while retrieving model information. Error: {str(e)}")
            return []

    def _load_config(self, file_name):
        """
        Loads the configuration from the specified file.

        Args:
            file_name (str): The name of the configuration file.

        Returns:
            dict: A dictionary containing the supported models information.

        Raises:
            FileNotFoundError: If the configuration file is not found.
            ValueError: If there is an error decoding the JSON.
            Exception: If an unexpected error occurs.
        """
        try:
            with open(file_name, 'r', encoding='utf8') as file:
                return json.load(file)[ModelManager.CONFIG_KEY][ModelManager.SUPPORTED_KEY]
        except FileNotFoundError:
            raise FileNotFoundError(f"The configuration file {file_name} was not found.")
        except json.JSONDecodeError:
            raise ValueError(f"There was an error decoding the JSON from the file {file_name}.")
        except Exception as e:
            raise Exception(f"An unexpected error occurred: {e}")

    def get_model_list(self):
        """
        Returns a list of available models with their details.

        Returns:
            list: A list of dictionaries containing model details.
        """
        model_details_list = []
        try:
            models = self.config.get_config('models/supported')
            if not models:
                self._logger.error("No supported models found in the configuration.")
                return []

            for model in models:
                model_details = {
                    "model_name": model.get("name", "Unknown"),
                    "model_id": model.get("id", "Unknown"),
                    "downloaded": model.get("downloaded", False),
                    "setup_finished": model.get("setup_finished", False),
                    "min_gpu_memory_required": model.get("min_gpu_memory", "Unknown"),
                    "model_info": model.get("model_info", "No information available")
                }
                model_details_list.append(model_details)

        except KeyError as e:
            self._logger.error(f"Failed to get the model list. Missing key: {str(e)}")
        except Exception as e:
            self._logger.error(f"An unexpected error occurred: {str(e)}")

        return model_details_list

    def verify_clip_checksum(self, model_id):
        try:
            model_info_list = self.config.get_config('models/supported')
            if not model_info_list:
                self._logger.error("No supported models found in the configuration.")
                return False

            model_info = next((m for m in model_info_list if m["id"] == model_id), None)
            if model_info is None:
                self._logger.error(f"Model {model_id} not found.")
                return False

            status = verify_clip_checksum(model_info, self._model_directory)
            return status
        except Exception as e:
            self._logger.error(f"Error while verifying checksum for {model_id}. Error: {str(e)}")
            return False

    def download_model(self, model_id):
        """
        Downloads the specified model.

        Args:
            model_id (str): The ID of the model to download.

        Returns:
            bool: True if the model was downloaded successfully, False otherwise.
        """
        try:
            model_info_list = self.config.get_config('models/supported')
            if not model_info_list:
                self._logger.error("No supported models found in the configuration.")
                return False

            model_info = next((m for m in model_info_list if m["id"] == model_id), None)
            if model_info is None:
                self._logger.error(f"Model {model_id} not found.")
                return False

            status = download_model_by_name(model_info, self._model_directory)
            if status:
                model_info_list_updated = self.config.get_config('models/supported')
                for i in range(len(model_info_list_updated)):
                    if (model_info_list_updated[i]['id'] == model_id):
                        model_info_list_updated[i]['downloaded'] = True
                self.config.write_default_config('models/supported', model_info_list_updated)
            return status
        except KeyError as e:
            self._logger.error(f"Failed to download the model {model_id}. Key error: {str(e)}")
            return False
        except Exception as e:
            self._logger.error(f"An unexpected error occurred while downloading the model {model_id}. Error: {str(e)}")
            return False

    def install_model(self, model_id):
        """
        Installs the specified model.

        Args:
            model_id (str): The ID of the model to install.

        Returns:
            bool: True if the model was installed successfully, False otherwise.
        """
        try:
            model_info_list = self.config.get_config('models/supported')
            if not model_info_list:
                self._logger.error("No supported models found in the configuration.")
                return False

            model_info = next((m for m in model_info_list if m["id"] == model_id), None)
            if model_info is None:
                self._logger.error(f"Model {model_id} not found.")
                return False

            status = build_engine_by_name(model_info=model_info, download_path=self._model_directory)
            if status:
                model_info_list = self.config.get_config('models/supported')
                for i in range(len(model_info_list)):
                    if (model_info_list[i]['id'] == model_id):
                        model_info_list[i]['setup_finished'] = True

                self.config.write_default_config('models/supported', model_info_list)
                self.config.write_default_config('models/selected', model_id)
                self._logger.info(f"\n\n\n\n\n{self.config.get_config('models/supported')}")
            return status
        except KeyError as e:
            self._logger.error(f"Failed to install the model {model_id}. Key error: {str(e)}")
            return False
        except Exception as e:
            self._logger.error(f"An unexpected error occurred while installing the model {model_id}. Error: {str(e)}")
            return False

    def is_model_downloaded(self, model_id):
        """
        Checks if the specified model is downloaded.

        Args:
            model_id (str): The ID of the model to check.

        Returns:
            bool: True if the model is downloaded, False otherwise.
        """
        try:
            model_info_list = self.config.get_config('models/supported')
            if not model_info_list:
                self._logger.error("No supported models found in the configuration.")
                return False

            model_info = next((m for m in model_info_list if m["id"] == model_id), None)
            if model_info is None:
                self._logger.error(f"Model {model_id} not found.")
                return False

            return model_info.get('downloaded', False)
        except Exception as e:
            self._logger.error(f"Failed to check download status for model_id {model_id}. Error: {str(e)}")
            return False

    def is_model_installed(self, model_id):
        """
        Checks if the specified model is installed.

        Args:
            model_id (str): The ID of the model to check.

        Returns:
            bool: True if the model is installed, False otherwise.
        """
        try:
            model_info_list = self.config.get_config('models/supported')
            if not model_info_list:
                self._logger.error("No supported models found in the configuration.")
                return False

            model_info = next((m for m in model_info_list if m["id"] == model_id), None)
            if model_info is None:
                self._logger.error(f"Model {model_id} not found.")
                return False

            return model_info.get('setup_finished', False)
        except Exception as e:
            self._logger.error(f"Failed to check install status for model_id {model_id}. Error: {str(e)}")
            return False

    def delete_model(self, model_id):
        """
        Deletes the specified model from the local storage.

        Args:
            model_id (str): The ID of the model to delete.

        Returns:
            bool: True if the model was deleted successfully, False otherwise.
        """
        try:
            model_info_list = self.config.get_config('models/supported')
            if not model_info_list:
                self._logger.error("No supported models found in the configuration.")
                return False

            model_info = next((m for m in model_info_list if m["id"] == model_id), None)
            if model_info is None:
                self._logger.error(f"Model {model_id} not found.")
                return False

            # Construct the model directory path
            model_dir = os.path.join(self._model_directory, model_id)

            # Remove the model directory and its contents
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)
            else:
                self._logger.error(f"Model directory {model_dir} does not exist.")
                return False

            # Update the configuration file after deletion
            for model in model_info_list:
                if model['id'] == model_id:
                    model['downloaded'] = False
                    model['setup_finished'] = False
            self.config.write_default_config('models/supported', model_info_list)

            return True
        except FileNotFoundError:
            self._logger.error(f"Model directory for {model_id} not found.")
            return False
        except Exception as e:
            self._logger.error(f"Failed to delete the model with model_id {model_id}. Error: {str(e)}")
            return False

    def update_active_model(self, model_id):
        self.config.write_default_config('models/selected', model_id)
        return True

    def update_dataset(self, dataset: str):
        self.config.write_default_config('dataset/selected', dataset)
        return True

    def update_data_directory_path(self, dataset_dir: str):
        self.config.write_default_config('dataset/selected_path', dataset_dir)
        return True
