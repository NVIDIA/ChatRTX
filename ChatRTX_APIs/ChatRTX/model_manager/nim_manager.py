# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import time
import subprocess
import os
import requests
import logging
import csv
import json
from ChatRTX.logger import ChatRTXLogger
from ChatRTX.model_manager.nvrtx_getkey import NVRTX_GetKey


class NIMManager:
    _instance = None  # Added for Singleton

    CSV_FILENAME = "nim-commands.csv"

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(NIMManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, distro_name_str="NVIDIA-Workbench"):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True

        '''
        Initializes the NIMManager class.
        '''
        ChatRTXLogger(log_level=logging.INFO, log_file='ChatRTX.log')
        self._logger = ChatRTXLogger.get_logger()
        self._ngc_api_key = None
        self._nim_server_proc_dict = {}
        self._workbench_cmd_prefix = "wsl -d " + distro_name_str + " -- "
        self._workbench_root_cmd_prefix = "wsl -u root -d " + distro_name_str + " -- "
        self._get_key = NVRTX_GetKey()
        if self._login_to_ngc_registry() == False:
            self._logger.error("failed to login to ngc registry")
            raise Exception("failed to login to ngc registry")
        else:
            self._update_nim_env_file()
        self._nim_commands_dict = self.read_csv()

        # close all the active NIM and restart based on the NIM user selects in the applciation
        local_nim = self.list_local_nims()
        for nim_profile in local_nim:
            self.force_close_nim(nim_profile)

    def _update_nim_env_file(self):
        # check if .nv_nim_env exists
        cmd_touch_profile = self._workbench_cmd_prefix + "touch $HOME/.nv_nim_env"
        logging.debug("executing " + cmd_touch_profile)
        result = subprocess.run(cmd_touch_profile, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode != 0:
            logging.debug("could not touch $HOME/.nv_nim_env")
            return False

        cmd_find_key = self._workbench_cmd_prefix + "grep \"NGC_API_KEY=" + self._ngc_api_key + "\" $HOME/.nv_nim_env"
        logging.debug("executing " + cmd_find_key)
        result = subprocess.run(cmd_find_key, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode == 0:
            logging.debug("NGC_API_KEY already exists in $HOME/.nv_nim_env, returning success")
            return True
        else:
            cmd_append_key = self._workbench_cmd_prefix + "awk -i inplace '{ print } ENDFILE { print \"NGC_API_KEY=" + self._ngc_api_key + "\" }' $HOME/.nv_nim_env"
            logging.debug(cmd_append_key)
            result = subprocess.run(cmd_append_key, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if result.returncode == 0:
                logging.debug("updated NGC_API_KEY in $HOME/.nv_nim_env")
                return True
            else:
                logging.error("could not update NGC_API_KEY in $HOME/.nv_nim_env")
                return False

    def _login_to_ngc_registry(self):
        key = os.environ.get('NGC_API_KEY_TEMP')
        if key:
            self._logger.warn("Using the API key from the environment variable instead of calling API")
        else:
            self._logger.warn("NGC_API_KEY_TEMP env var not found, generating key")
            deviceInfoNvml = self._get_key.get_device_info_nvml()
            key = self._get_key.get_ngc_key(deviceInfoNvml)

        command = self._workbench_cmd_prefix + "podman login --username '$oauthtoken' --password-stdin nvcr.io"
        login_process = subprocess.Popen(command,
                                         stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE)
        if login_process:
            out, err = login_process.communicate(input=bytes(key, "utf-8"))
            retcode = login_process.wait()
            if retcode:
                self._logger.error("login attempt failed with returncode " + str(retcode))
                return False
            else:
                self._ngc_api_key = key
                self._logger.info("login successful")
                return True
        else:
            self._logger.error("failed to execute login command " + command)

    def is_installed(self, nim_id):
        self._logger.debug("entering is_installed")
        nim_cmds = self._nim_commands_dict[nim_id]
        command = nim_cmds['chk_install']
        if command:
            self._logger.debug(command)
            result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if result.returncode == 0:
                return True
            else:
                return False
        else:
            return True

    def read_csv(self):
        start_cmdline_prefix = self._workbench_cmd_prefix + 'podman run -it --rm --device=nvidia.com/gpu=all '
        stop_cmdline_prefix = self._workbench_cmd_prefix + 'podman stop '

        csv_abs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), NIMManager.CSV_FILENAME)

        with open(csv_abs_path, "r", encoding="utf-8-sig") as file:
            csv_reader = csv.DictReader(file)
            nim_list = [row for row in csv_reader]

        nim_commands_list = []
        nim_commands_dict = {}

        for nim in nim_list:
            chk_install_cmdline = ""
            install_cmdline = ""
            prereq_1_cmdline = ""
            prereq_2_cmdline = ""
            start_cmdline = start_cmdline_prefix
            stop_cmdline = stop_cmdline_prefix

            start_cmdline += '--name='
            start_cmdline += nim['name']
            start_cmdline += ' '

            stop_cmdline += nim['name']

            start_cmdline += '--shm-size='
            start_cmdline += nim['shm_size']
            start_cmdline += ' '

            if int(nim['requires_ngc_api_key']) == 1:
                start_cmdline += '--env-file $HOME/.nv_nim_env '

            if nim['cache_dir']:
                src_dir = '$HOME/nimonwsl2/' + nim['name'] + '/' + nim['tag'] + '/.cache'
                start_cmdline += '-v '
                start_cmdline += src_dir
                start_cmdline += ':'
                start_cmdline += nim['cache_dir']
                start_cmdline += ' '

                chk_install_cmdline = self._workbench_cmd_prefix
                chk_install_cmdline += 'test -d '
                chk_install_cmdline += src_dir

                prereq_1_cmdline = self._workbench_cmd_prefix
                prereq_1_cmdline += 'mkdir -p '
                prereq_1_cmdline += src_dir

                prereq_2_cmdline = self._workbench_cmd_prefix
                prereq_2_cmdline += 'chmod 777 -R '
                prereq_2_cmdline += src_dir

            if nim['env_0']:
                start_cmdline += '-e '
                start_cmdline += nim['env_0']
                start_cmdline += ' '

            if nim['env_1']:
                start_cmdline += '-e '
                start_cmdline += nim['env_1']
                start_cmdline += ' '

            if nim['env_2']:
                start_cmdline += '-e '
                start_cmdline += nim['env_2']
                start_cmdline += ' '

            if nim['env_3']:
                start_cmdline += '-e '
                start_cmdline += nim['env_3']
                start_cmdline += ' '

            if nim['env_4']:
                start_cmdline += '-e '
                start_cmdline += nim['env_4']
                start_cmdline += ' '

            if nim['env_5']:
                start_cmdline += '-e '
                start_cmdline += nim['env_5']
                start_cmdline += ' '

            if nim['portmap_0']:
                start_cmdline += '-p '
                start_cmdline += nim['portmap_0']
                start_cmdline += ' '

            if nim['portmap_1']:
                start_cmdline += '-p '
                start_cmdline += nim['portmap_1']
                start_cmdline += ' '

            if nim['portmap_2']:
                start_cmdline += '-p '
                start_cmdline += nim['portmap_2']
                start_cmdline += ' '

            start_cmdline += nim['registry_path']
            start_cmdline += ":"
            start_cmdline += nim['tag']

            nim_commands = {}
            nim_commands['name'] = nim['name']
            nim_commands['gpu'] = nim['gpu']
            nim_commands['id'] = nim['registry_path'] + ':' + nim['tag']
            nim_commands['chk_install'] = chk_install_cmdline
            nim_commands['install'] = install_cmdline
            nim_commands['prereq_1'] = prereq_1_cmdline
            nim_commands['prereq_2'] = prereq_2_cmdline
            nim_commands['start'] = start_cmdline
            nim_commands['stop'] = stop_cmdline
            nim_commands['health_ready_port'] = nim['health_ready_port']
            nim_commands['http_ports_out'] = nim['http_ports_out']
            nim_commands['grpc_ports_out'] = nim['grpc_ports_out']

            nim_commands_list.append(nim_commands)
            nim_commands_dict[nim_commands['id']] = nim_commands

        # print(nim_commands_list)
        # print(nim_commands_dict)
        # print('\n')
        return nim_commands_dict

    def is_authenticated(self):
        command = self._workbench_cmd_prefix
        command += "podman login --get-login nvcr.io"
        self._logger.info(command)
        result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode == 0:
            return True
        else:
            return False

    def download_nim(self, nim_id):
        '''
        Downloads the specified NIM image.

        Args:
            nim_id (str): The NIM registry URL.

        Returns:
            bool: True if the NIM was downloaded successfully, False otherwise.
        '''
        if self.is_authenticated() == False:
            self._logger.error(
                "User is not authenticated, please authenticate using podman login nvcr.io. Use $oauthtoken as Username and NGC_API_KEY as password.")
            return False

        self._logger.debug("entering download_nim")
        command = self._workbench_cmd_prefix
        command += "podman pull "
        command += nim_id
        self._logger.info(command)
        result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode == 0:
            self._logger.info(nim_id + " image downloaded successfully")
            return True
        else:
            self._logger.error("failed to download image " + nim_id)
            return False

    def list_local_nims(self):
        '''
        Lists the NIM images available locally.

        Returns:
            list: A list of NIM registry URLs.
        '''
        self._logger.info("entering list_local_nims")
        local_nim_ids = []
        command = self._workbench_cmd_prefix
        command += "podman images"
        self._logger.info(command)
        result = subprocess.run(command, stdout=subprocess.PIPE, universal_newlines=True)
        result_multiline = result.stdout.splitlines()
        if len(result_multiline) == 1:
            return local_nim_ids
        else:
            for image_info_line in result_multiline[1:]:
                image_info_list = image_info_line.split()
                image_repo = image_info_list[0]
                image_tag = image_info_list[1]
                nim_id = image_repo + ':' + image_tag
                local_nim_ids.append(nim_id)

        if len(local_nim_ids) == 0:
            self._logger.warning(
                "No local NIM image found, use download_nim to download NIM image from builds.nvidia.com")
        else:
            self._logger.info("List of local NIMs")
            for nim_id in local_nim_ids:
                self._logger.info(nim_id)

        return local_nim_ids

    def install_nim(self, nim_id):
        self._logger.debug("entering install_nim")

        if self.is_installed(nim_id):
            self._logger.debug(nim_id + "already installed")
            return True

        nim_cmdlines = self._nim_commands_dict.get(nim_id)
        if nim_cmdlines is None:
            return False

        prereq_1_cmdline = nim_cmdlines['prereq_1']
        prereq_2_cmdline = nim_cmdlines['prereq_2']
        try:
            if prereq_1_cmdline:
                self._logger.debug("executing " + prereq_1_cmdline)
                mkdir_process = subprocess.run(prereq_1_cmdline)
                if prereq_2_cmdline:
                    self._logger.debug("executing " + prereq_2_cmdline)
                    chmod_process = subprocess.run(prereq_2_cmdline)
        except subprocess.CalledProcessError as e:
            self._logger.error(f"An error occurred while executing the command: {e}")
            return False



    def _get_default_user_home_dir(self):
        cmd_get_home_dir = self._workbench_cmd_prefix + "echo $HOME"
        result = subprocess.run(cmd_get_home_dir, capture_output=True, text=True)
        if result.returncode:
            logging.error("could not get home directory")
            return ""
        home_dir = result.stdout.strip()
        logging.debug("$HOME for default user is " + home_dir)
        return home_dir

    def _uninstall_nim(self, nim_id):
        '''
        Do not call this from client code
        This function is for testing purpose only
        '''
        host_cache_dir = nim_id.rsplit('/', 1)[-1]
        host_cache_dir = host_cache_dir.replace(":", "/")

        home_dir = self._get_default_user_home_dir()
        if home_dir == False:
            return False

        host_cache_dir = home_dir + "/nimonwsl2/" + host_cache_dir + "/.cache"
        cmd_rm_cache_dir = self._workbench_root_cmd_prefix + "rm -rf " + host_cache_dir
        logging.debug("executing " + cmd_rm_cache_dir)
        result = subprocess.run(cmd_rm_cache_dir)
        if result.returncode:
            logging.error("could not delete cache directory " + host_cache_dir)
            return False

        cmd_rmi_nim = self._workbench_cmd_prefix + "podman rmi --ignore " + nim_id
        result = subprocess.run(cmd_rmi_nim)
        if result.returncode:
            logging.error("could not remove container image " + nim_id)
            return False

        logging.info("successfully uninstalled nim " + nim_id)
        return True

    def is_nim_running(self, nim_id):
        podman_ps_command = self._workbench_cmd_prefix + "podman ps --format json"
        podman_ps_json = subprocess.run(podman_ps_command, capture_output=True)
        podman_ps_json = podman_ps_json.stdout
        podman_ps_json = json.loads(podman_ps_json)
        for nim_json in podman_ps_json:
            if nim_json['Image'] == nim_id:
                return True
        return False

    def start_nim(self, nim_id, http_ports, grpc_ports, force_nim_start=False):
        '''
        Starts the container for specified NIM.

        Args:
            nim_id (str): The NIM registry URL.

        Returns:
            bool: True if the NIM was started successfully, False otherwise.
        '''
        self._logger.info("entering start_nim")
        if force_nim_start:
            if self.force_close_nim(nim_id) == False:
                self._logger.error("Error in force clssign the NIMs ")
                return False

        if self.is_nim_running(nim_id):
            nim_cmds = self._nim_commands_dict[nim_id]
            if nim_cmds['http_ports_out']:
                http_ports.append(nim_cmds['http_ports_out'])
            if nim_cmds['grpc_ports_out']:
                grpc_ports.append(nim_cmds['grpc_ports_out'])
            logging.warn("NIM is already running, nothing to do, returning with success")
            return True

        if self.is_installed(nim_id) == False:
            if self.install_nim(nim_id) == False:
                return False

        nim_cmds = self._nim_commands_dict[nim_id]
        command = nim_cmds['start']

        # TODO: identify a free port and map NIM port to that

        self._logger.info("executing " + command)
        run_process = subprocess.Popen(command,
                                       stdin=subprocess.PIPE,
                                       stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL)
        if run_process:
            self._nim_server_proc_dict[nim_id] = run_process
        else:
            return False

        while True:
            running = run_process.poll() is None
            if running == False:
                logging.error("nim exited unexpectedly with returncode " + str(run_process.returncode))
                return False
            try:
                nim_cmds = self._nim_commands_dict[nim_id]
                health_ready_port = nim_cmds['health_ready_port']
                health_ready_url = "http://localhost:" + health_ready_port + "/v1/health/ready"
                r = requests.get(health_ready_url)
                self._logger.debug(r)
                if r.status_code == 200:
                    self._logger.info("NIM is up and running")
                    if nim_cmds['http_ports_out']:
                        http_ports.append(nim_cmds['http_ports_out'])
                    if nim_cmds['grpc_ports_out']:
                        grpc_ports.append(nim_cmds['grpc_ports_out'])
                    break
            except:
                pass
            time.sleep(1)


        # Clear the sram cache
        cache_clear_cmd = self._workbench_root_cmd_prefix + 'sh -c "echo 1 > /proc/sys/vm/drop_caches"'
        self._logger.info("Clearing cache with command: " + cache_clear_cmd)
        cache_result = subprocess.run(cache_clear_cmd,
                                       stdin=subprocess.PIPE,
                                       stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL)
        if cache_result.returncode != 0:
            self._logger.error("Failed to clear cache: " + cache_result.stderr.decode().strip())
        else:
            self._logger.info("Cache cleared successfully.")

        return True

    def stop_nim(self, nim_id):
        '''
        Stops the container for specified NIM.

        Args:
            nim_id (str): The NIM registry URL.

        Returns:
            bool: True if the NIM was stopped successfully, False otherwise.
        '''
        self._logger.debug("entering stop_nim")
        if self.is_nim_running(nim_id) == False:
            self._logger.warning("No NIM(" + nim_id + ") is running, nothing stop, returning success")
            return True

        if self._nim_server_proc_dict.get(nim_id):
            try:
                nim_commands = self._nim_commands_dict[nim_id]
                command = nim_commands['stop']
                self._logger.info("executing " + command)
                result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if result.returncode == 0:
                    self._logger.info(nim_id + " NIM stopped successfully")
                    self._nim_server_proc_dict[nim_id] = None
                    return True
                else:
                    self._logger.error("failed to stop NIM " + nim_id)
                    return False
            except TimeoutExpired:
                # TODO: update following print
                self._logger.error("NIM container did not stop in 120 seconds")
        else:
            self._logger.warning("No NIM process is running that was launched by current application")

    def delete_nim(self, nim_id):
        '''
        Deletes the specified NIM image from the local storage.

        Args:
            nim_id (str): The NIM registry URL.

        Returns:
            bool: True if the NIM was deleted successfully, False otherwise.
        '''
        # TODO: check if container with nim_id image is already running
        self._logger.debug("entering delete_nim")
        try:
            command = self._workbench_cmd_prefix
            command += "podman rmi "
            command += nim_id
            self._logger.debug("executing " + command)
            result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if result.returncode == 0:
                self._logger.info(nim_id + " NIM stopped successfully")
                return True
            else:
                self._logger.error("failed to stop NIM " + nim_id)
                return False
        except TimeoutExpired:
            # TODO: update following print
            self._logger.error("NIM container did not stop in 120 seconds")

    def container_exists(self, container_name):
        """
        Checks if a container with the given name exists in any state using 'podman ps -a'.

        Args:
            container_name (str): The name of the container to check.

        Returns:
            bool: True if the container exists, False otherwise.
        """
        command = ["podman", "ps", "-a", "--format", "json"]
        try:
            result = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,  # Capture stdout to be able to parse it
                stderr=subprocess.DEVNULL,
                check=True,
                text=True
            )
            containers = json.loads(result.stdout)
            for container in containers:
                # Depending on your podman version, the container name may be in 'Names' or 'Name'
                if 'Names' in container and container['Names'] == container_name:
                    return True
                if 'Name' in container and container['Name'] == container_name:
                    return True
            return False
        except Exception as e:
            self._logger.error("Exception occurred while checking container existence: " + str(e))
            return False

    def force_close_nim(self, nim_id):
        """
        Forcefully stops and removes the container for the specified NIM using 'podman rm -f'.

        Args:
            nim_id (str): The NIM registry URL (identifier) from which the container name is derived.

        Returns:
            bool: True if the container was forcefully removed successfully, False otherwise.
        """
        self._logger.debug("entering force_close_nim")
        nim_commands = self._nim_commands_dict.get(nim_id)
        if not nim_commands:
            self._logger.error(f"NIM with id {nim_id} not found in commands dictionary")
            return False
        container_name = nim_commands['name']
        command = self._workbench_cmd_prefix + "podman rm -f " + container_name
        self._logger.info("executing " + command)
        try:
            result = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True
            )
            self._logger.info(f"Container {container_name} forcefully removed: {result.stdout.strip()}")
            if self._nim_server_proc_dict.get(nim_id):
                self._nim_server_proc_dict[nim_id] = None
            return True
        except subprocess.CalledProcessError as e:
            self._logger.error(f"Failed to force remove container {container_name}: {e.stderr.strip()}")
            return False

    # nim_manager = NIMManager.get_instance(distro_name_str="NVIDIA-Workbench")
    @classmethod
    def get_instance(cls, *args, **kwargs):
        """
        Returns the singleton instance of NIMManager.
        If it doesn't exist, it creates one.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            NIMManager: The singleton instance.
        """
        if cls._instance is None:
            cls._instance = cls(*args, **kwargs)
        return cls._instance