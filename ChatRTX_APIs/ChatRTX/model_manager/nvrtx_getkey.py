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

import requests
import logging
import subprocess
import os
import pynvml as nvml
import re
from ChatRTX.logger import ChatRTXLogger

class NVRTX_GetKey:

    def __init__(self):
         ChatRTXLogger(log_level=logging.INFO, log_file='ChatRTX.log')
         self._logger = ChatRTXLogger.get_logger()

    def get_device_info_nvml(self):
        try:
            nvml.nvmlInit()
            deviceCount = nvml.nvmlDeviceGetCount()
            deviceInfo = []
            for i in range(deviceCount):
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                uuid = nvml.nvmlDeviceGetUUID(handle)
                name = nvml.nvmlDeviceGetName(handle)
                brand = nvml.nvmlDeviceGetBrand(handle)
                architecture = nvml.nvmlDeviceGetArchitecture(handle)
                pciDeviceId = nvml.nvmlDeviceGetPciInfo(handle).pciDeviceId
                pciDeviceId_64bit = format(pciDeviceId, 'X')
                # pdi = 'Unknown' # nvml.nvmlDeviceGetPDI(handle) # TODO - PDI is not available in NVML API yet
                fakePdi = int(hash(uuid) % 2**64) # hash the uuid to a 16 bit integer for testing
                pdi = fakePdi # TODO - just for testing until PDI is available in NVML API
                pdi_64bit = format(pdi, 'X')
                # Map brand and architecture to human-readable strings
                # brand_str = get_brand_name(brand)
                # architecture_str = get_architecture_name(architecture)

                deviceInfo.append({
                    'uuid': uuid,
                    'pdi': f"0x{pdi_64bit}",
                    'name': name,
                    'brand': brand,
                    'architecture': architecture,
                    'pci_device_id': f"0x{pciDeviceId_64bit}",
                })
            nvml.nvmlShutdown()
            #print("NVML device info:")
            #print(deviceInfo)
            return deviceInfo
        except nvml.NVMLError as e:
            self._logger.error(f"NVML Error: {e}")
            nvml.nvmlShutdown()
            return []

    def get_device_info_smi(self):
        try:
            output = subprocess.check_output(['nvidia-smi', '-q'], text=True)
            deviceInfo = []
            uuid_pattern = re.compile(r"GPU UUID\s*:\s*(.+)")
            pdi_pattern = re.compile(r"PDI\s*:\s*(.+)")
            name_pattern = re.compile(r"Product Name\s*:\s*(.+)")
            brand_pattern = re.compile(r"Product Brand\s*:\s*(.+)")
            arch_pattern = re.compile(r"Product Architecture\s*:\s*(.+)")
            pci_pattern = re.compile(r"Device Id\s*:\s*(.+)")

            matches = {
                "uuid": uuid_pattern.search(output),
                "pdi": pdi_pattern.search(output),
                "name": name_pattern.search(output),
                "brand": brand_pattern.search(output),
                "architecture": arch_pattern.search(output),
                "pci_device_id": pci_pattern.search(output),
            }

            # TODO - just for testing until PDI is available in NVML API
            if not matches["pdi"]:
                fakePdi = int(hash(matches["uuid"].group(1).strip()) % 2**64) # hash the uuid to a 16 bit integer for testing
                pdi_64bit = format(fakePdi, 'X')
                matches["pdi"] = re.match(r"(.+)", f"0x{pdi_64bit}")

            deviceInfo.append({k: (m.group(1).strip() if m else "Unknown") for k, m in matches.items()})
            #print("SMI device info:")
            #print(deviceInfo)
            return deviceInfo
        except subprocess.CalledProcessError as e:
            self._logger.error(f"nvidia-smi error: {e}")
            return []

    def validate_device_info(self, deviceInfo):
        matchInName = ['RTX', 'OTHER DEVICE NAME TEST']
        for device in deviceInfo:
            if any(keyword in device.get('name', '') for keyword in matchInName):
                #print(f"Valid device name {device['name']} found")
                return True
        matchInBrandString = ['GeForce','5']
        for device in deviceInfo:
            if any(keyword in str(device.get('brand', '')) for keyword in matchInBrandString):
                #print(f"Valid device brand {device['brand']} found")
                return True
        #print("No valid device found in the list")
        return False

    def get_ngc_key(self, deviceInfo):
        # if NGC_API_KEY or NGC_CLI_API_KEY is set warn user to use that instead
        apiKey = os.environ.get('NGC_API_KEY')
        if apiKey:
            #print("Using the API key from the environment variable instead of calling API")
            return apiKey

        ngcKeyServiceUrl = 'https://nts.ngc.nvidia.com/v1/token'
        payload = {
            "client_id": "nim-mgmt-api",
            "pdi": deviceInfo[0].get('pdi', 'Unknown') if deviceInfo else 'Unknown',
            "access_policy_name": "nim-dev",
            "device": deviceInfo[0] if deviceInfo else {},
            "expires_in" : 36000
        }
        #print(f"debug Payload: {payload}")
        try:
            response = requests.post(ngcKeyServiceUrl, headers={'Accept': 'application/json'}, json=payload)
            #print(f"Response: {response}")
            response.raise_for_status()
            keyData = response.json()
            #print(f"Key data: {keyData}")
            return keyData.get('access_token')
        except requests.RequestException as e:
            self._logger.error(f"Error fetching API key: {e}")
            return None