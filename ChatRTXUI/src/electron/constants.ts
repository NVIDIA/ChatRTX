/* SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

export const HISTORY_UPDATE_EVENT = 'history-update'
export const WAITING_FOR_RESPONSE_UPDATE_EVENT = 'waiting-for-response-update'

export const ON_RECORDING = 'on-recording'
export const ON_PROCESSING_SPEECH = 'on-processing-speech'
export const ON_INIT_SPEECH = 'on-init-speech'
export const ON_RECORDING_TIME = 'on-recording-time'
export const ON_SPEECH_ERROR = 'on-speech-error'
export const ON_SPEECH_RESULT = 'on-speech-result'

// Event from electron when app is ready to use
export const APP_READY = 'APP_READY'

// Event from electron to be sent to render in case visible data changes for model
export const SUPPORTED_MODEL_UPDATE = 'SUPPORTED_MODEL_UPDATE'
// Event from electron to be sent to render in case visible data changes for dataset
export const DATASET_INFO_UPDATED = 'DATASET_INFO_UPDATED'
// Event from electron to be sent to render in case of visible changes for fine tune
export const FINE_TUNE_UPDATED = 'FINE_TUNE_UPDATED'
export const ON_FINE_TUNING_FOLDER_PATH_UPDATE =
    'ON_FINE_TUNING_FOLDER_PATH_UPDATE'

// PYTHON EVENTS -- received to main process from python backend

export const ON_PYTHON_ENGINE_INIT = 'ON_PYTHON_ENGINE_INIT'
export const ON_PYTHON_ENGINE_INIT_ERROR = 'ON_PYTHON_ENGINE_INIT_ERROR'

export const ON_RESET_CHAT = 'ON_RESET_CHAT'

// Model management event
export const ACTIVE_MODEL_UPDATE = 'ACTIVE_MODEL_UPDATE'
export const ACTIVE_MODEL_UPDATE_ERROR = 'ACTIVE_MODEL_UPDATE_ERROR'
export const MODEL_DOWNLOAD_ERROR = 'MODEL_DOWNLOAD_ERROR'
export const MODEL_INSTALL_ERROR = 'MODEL_INSTALL_ERROR'
export const MODEL_DELETE_ERROR = 'MODEL_DELETE_ERROR'
export const MODEL_DOWNLOADED = 'MODEL_DOWNLOADED'
export const MODEL_INSTALLED = 'MODEL_INSTALLED'
export const MODEL_DELETED = 'MODEL_DELETED'

// Dataset management event
export const ON_DATASET_UPDATE = 'ON_DATASET_UPDATE'
export const ON_DATASET_UPDATE_ERROR = 'ON_DATASET_UPDATE_ERROR'
export const ON_INDEX_REGENERATED = 'ON_DATA_REGENERATED'
export const ON_INDEX_REGENERATE_ERROR = 'ON_DATA_REGENERATE_ERROR'

// Fine tuning data updated
export const ON_BASE_MODEL_DOWNLOADED = 'ON_BASE_MODEL_DOWNLOADED'
export const ON_BASE_MODEL_DOWNLOAD_ERROR = 'ON_BASE_MODEL_DOWNLOAD_ERROR'
export const ON_PROFILE_CREATED = 'ON_PROFILE_CREATED'
export const ON_PROFILE_CREATE_ERROR = 'ON_PROFILE_CREATE_ERROR'
export const ON_PROFILE_SELECTED = 'ON_PROFILE_SELECTED'
export const ON_PROFILE_SELECT_ERROR = 'ON_PROFILE_SELECT_ERROR'
export const ON_PROFILE_DELETED = 'ON_PROFILE_DELETED'
export const ON_PROFILE_DELETED_ERROR = 'ON_PROFILE_DELETED_ERROR'
export const ON_FINE_TUNING_ENABLED = 'ON_FINE_TUNING_ENABLED'
export const ON_FINE_TUNING_ENABLE_ERROR = 'ON_FINE_TUNING_ENABLE_ERROR'
export const ON_FINE_TUNING_DISABLED = 'ON_FINE_TUNING_DISABLED'
export const ON_FINE_TUNING_DISABLE_ERROR = 'ON_FINE_TUNING_DISABLE_ERROR'
