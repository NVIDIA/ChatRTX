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

import { contextBridge, ipcRenderer } from 'electron'
import {
    DatasetInfo,
    DatasetSources,
    FineTuningAdvancedParams,
    FineTuningProfileConfig,
    HistoryItem,
    ModelDetails,
    ModelFineTuningDetails,
    ModelId,
} from './types'
import {
    ACTIVE_MODEL_UPDATE,
    HISTORY_UPDATE_EVENT,
    WAITING_FOR_RESPONSE_UPDATE_EVENT,
    MODEL_DOWNLOAD_ERROR,
    MODEL_INSTALL_ERROR,
    MODEL_DOWNLOADED,
    MODEL_INSTALLED,
    DATASET_INFO_UPDATED,
    ACTIVE_MODEL_UPDATE_ERROR,
    MODEL_DELETED,
    MODEL_DELETE_ERROR,
    SUPPORTED_MODEL_UPDATE,
    FINE_TUNE_UPDATED,
    ON_PROFILE_CREATED,
    APP_READY,
    ON_DATASET_UPDATE_ERROR,
    ON_DATASET_UPDATE,
} from './constants'

const makeListener = (eventName: string, callback: (data: any) => void) => {
    const onChange = (_event: any, data: any) => callback(data)
    ipcRenderer.on(eventName, onChange)
    return () => ipcRenderer.removeListener(eventName, onChange)
}

export const ClientAPI = {
    /** MAIN APP */
    maximize: () => ipcRenderer.invoke('maximize'),
    unmaximize: () => ipcRenderer.invoke('unmaximize'),
    isMaximized: () => ipcRenderer.sendSync('isMaximized'),
    minimize: () => ipcRenderer.invoke('minimize'),
    close: () => ipcRenderer.invoke('close'),
    openLink: (url: string) => ipcRenderer.invoke('openLink', url),

    /** CHAT BRIDGE */
    sendPrompt: (prompt: string, isStreaming: boolean) =>
        ipcRenderer.invoke('sendPrompt', prompt, isStreaming),
    resetChat: () => ipcRenderer.invoke('resetChat'),
    undoPrompt: () => ipcRenderer.invoke('undoPrompt'),
    retryPrompt: (id: string) => ipcRenderer.invoke('retryPrompt', id),
    isWaiting: (): boolean => ipcRenderer.sendSync('isWaiting'),
    onWaiting: (callback: (data: boolean) => void): (() => void) =>
        makeListener(WAITING_FOR_RESPONSE_UPDATE_EVENT, callback),
    onAppReady: (callback: () => void): (() => void) =>
        makeListener(APP_READY, callback),
    isAppReady: (): boolean => ipcRenderer.sendSync('isAppReady'),

    /** ASR */
    getTextFromAudio: (data: string): Promise<string> =>
        ipcRenderer.invoke('getTextFromAudio', data),
    initAsrModel: () => ipcRenderer.invoke('initAsrModel'),

    /** MODEL MANAGEMENT API */
    getSupportedModels: (): ModelDetails[] =>
        ipcRenderer.sendSync('getSupportedModels'),
    getModelDetails: (modelId: ModelId): ModelDetails =>
        ipcRenderer.sendSync('getModelDetails', modelId),
    getActiveModelDetails: (): ModelDetails =>
        ipcRenderer.sendSync('getActiveModelDetails'),
    getActiveModel: (): ModelId => ipcRenderer.sendSync('getActiveModel'),
    setActiveModel: (modelId: ModelId) =>
        ipcRenderer.invoke('setActiveModel', modelId),
    deleteModel: (modelId: ModelId) =>
        ipcRenderer.invoke('deleteModel', modelId),
    downloadModel: (modelId: ModelId) =>
        ipcRenderer.invoke('downloadModel', modelId),
    installModel: (modelId: ModelId) =>
        ipcRenderer.invoke('installModel', modelId),
    onSupportedModelsUpdated: (callback: () => void): (() => void) =>
        makeListener(SUPPORTED_MODEL_UPDATE, callback),
    onActiveModelUpdate: (callback: (modelId: ModelId) => void): (() => void) =>
        makeListener(ACTIVE_MODEL_UPDATE, callback),
    onActiveModelUpdateError: (
        callback: (modelId: ModelId) => void
    ): (() => void) => makeListener(ACTIVE_MODEL_UPDATE_ERROR, callback),
    onModelDownloadError: (
        callback: (modelId: ModelId) => void
    ): (() => void) => makeListener(MODEL_DOWNLOAD_ERROR, callback),
    onModelDownloaded: (callback: (modelId: ModelId) => void): (() => void) =>
        makeListener(MODEL_DOWNLOADED, callback),
    onModelInstallError: (callback: (modelId: ModelId) => void): (() => void) =>
        makeListener(MODEL_INSTALL_ERROR, callback),
    onModelInstalled: (callback: (modelId: ModelId) => void): (() => void) =>
        makeListener(MODEL_INSTALLED, callback),
    onModelDeleteError: (callback: (modelId: ModelId) => void): (() => void) =>
        makeListener(MODEL_DELETE_ERROR, callback),
    onModelDeleted: (callback: (modelId: ModelId) => void): (() => void) =>
        makeListener(MODEL_DELETED, callback),

    /**FINE TUNNING */
    // parentId refers to the modelId of parent under which fine tuning is shown
    isFineTuningEnabled: (parentId: ModelId): boolean =>
        ipcRenderer.sendSync('isFineTuningEnabled', parentId),
    getModelFineTuningDetails: (parentId: ModelId): ModelFineTuningDetails =>
        ipcRenderer.sendSync('getModelFineTuningDetails', parentId),
    getDefaultAdvancedParams: (parentId: ModelId): FineTuningAdvancedParams =>
        ipcRenderer.sendSync('getDefaultAdvancedParams', parentId),
    getFineTunningProfileConfig: (
        parentId: ModelId,
        profileId: string
    ): FineTuningProfileConfig =>
        ipcRenderer.sendSync(
            'getFineTunningProfileConfig',
            parentId,
            profileId
        ),
    downloadBaseModel: (parentId: ModelId) =>
        ipcRenderer.invoke('downloadBaseModel', parentId),
    selectFineTuningModel: (parentId: ModelId, profileId: string) =>
        ipcRenderer.invoke('selectFineTuningModel', parentId, profileId),
    createFineTuningProfile: (
        parentId: ModelId,
        fineTuneConfig: FineTuningProfileConfig
    ) =>
        ipcRenderer.invoke('createFineTuningProfile', parentId, fineTuneConfig),
    deleteFineTuningProfile: (parentId: ModelId, profileId: string) =>
        ipcRenderer.invoke('deleteFineTuningProfile', parentId, profileId),
    toggleFineTuningEnablement: (parentId: ModelId, shouldEnable: boolean) =>
        ipcRenderer.invoke(
            'toggleFineTuningEnablement',
            parentId,
            shouldEnable
        ),
    selectFolder: () => ipcRenderer.invoke('selectFolder'),
    onModelFineTuningDetailsUpdate: (callback: () => void): (() => void) =>
        makeListener(FINE_TUNE_UPDATED, callback),
    onFineTuningProfileCreated: (
        callback: (profileId: string) => void
    ): (() => void) => makeListener(ON_PROFILE_CREATED, callback),

    /** HISTORY */
    getHistory: (): HistoryItem[] => ipcRenderer.sendSync('getHistory'),
    onHistoryUpdate: (callback: (data: HistoryItem[]) => void): (() => void) =>
        makeListener(HISTORY_UPDATE_EVENT, callback),

    /** DATASETS */
    getDatasetInfo: (modelId: ModelId): DatasetInfo =>
        ipcRenderer.sendSync('getDatasetInfo', modelId),
    selectDatasetFolder: () => ipcRenderer.invoke('selectDatasetFolder'),
    changeDatasetSource: (val: DatasetSources) =>
        ipcRenderer.invoke('changeDatasetSource', val),
    regenerateIndex: () => ipcRenderer.invoke('regenerateIndex'),
    onDatasetInfoUpdate: (callback: () => void): (() => void) =>
        makeListener(DATASET_INFO_UPDATED, callback),
    onDatasetUpdate: (callback: () => void): (() => void) =>
        makeListener(ON_DATASET_UPDATE, callback),
    onDatasetUpdateError: (callback: () => void): (() => void) =>
        makeListener(ON_DATASET_UPDATE_ERROR, callback),
    openPath: (path: string) => ipcRenderer.invoke('openPath', path),

    /** SAMPLE QUESTIONS */
    getSampleQuestions: (modelId: ModelId, selected_path: string): string[] =>
        ipcRenderer.sendSync('getSampleQuestions', modelId, selected_path),
}

contextBridge.exposeInMainWorld('clientAPI', ClientAPI)
