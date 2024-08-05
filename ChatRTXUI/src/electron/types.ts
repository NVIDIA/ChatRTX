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

export const consentTypes = ['none', 'passive', 'active', 'modal'] as const

export enum DatasetSources {
    DIRECTORY = 'directory',
    NO_DATASET = 'nodataset',
}

export enum ModelId {
    Mistral = 'mistral_7b_AWQ_int4_chat',
    Llaama = 'llama2_13b_AWQ_INT4_chat',
    ChatGlm = 'chatglm3_6b_AWQ_int4',
    Gemma = 'gemma_7b_int4',
    Clip = 'clip_model',
}

export interface ModelDetails {
    name: string
    id: ModelId
    downloaded: boolean
    setup_finished: boolean
    should_show_in_UI: boolean
    model_info: string
    model_license: string
    model_size: string
    isFineTuningSupported?: boolean
    modelDevelopers?: string
    isTextBased?: boolean
    isImageBased?: boolean
    isChineseSupported?: boolean
    isEnglishSupported?: boolean
}

export interface FineTuningAdvancedParams {
    systemPrompt: string
    loraRank: number
    saveSteps: number
    epochs: number
    contextLenght: number
}

export interface FineTuningProfileConfig {
    name: string
    profileId?: string
    folderPath: string
    testFolderPath?: string
    advancedParams: FineTuningAdvancedParams
}

export interface ModelFineTuningDetails {
    parentModelId: ModelId
    baseModelId: string
    baseModelName: string
    selectedProfileId: string
    isFineTuningEnabled: boolean
    isBaseModelDownloaded: boolean
    defaultAdvancedParams: FineTuningAdvancedParams
    FineTuningProfileConfigs: FineTuningProfileConfig[]
}

export interface FineTuningInfo {
    [k: string | ModelId]: ModelFineTuningDetails
}

export interface ModelInfo {
    supported: ModelDetails[]
    selected: ModelId
}

export enum DataFormat {
    Text = 'text',
    Images = 'images',
}

export interface DatasetInfo {
    sources: DatasetSources[]
    selected: DatasetSources
    path: string
    path_chinese: string
    path_clip: string
    selected_path: string
    supportedDataFormat?: DataFormat
}

export enum sampleQuestionInfoKey {
    Default = 'default',
    Chinese = 'chinese',
    Images = 'images',
}

export interface sampleQuestionVal {
    modelIdList: ModelId[]
    dataset_path: string
    queries: string[]
}

export type SampleQuestionInfo = {
    [key in sampleQuestionInfoKey]: sampleQuestionVal
}

export interface ChatState {
    datasetType: (typeof DatasetTypes)[number]
    datasetValue: string
}

export const DatasetTypes = ['Folder Path'] as const

export interface HistoryItem {
    id: string
    prompt: string
    response: string
    isResponseGenerated?: boolean
    isTextResponse?: boolean
}

export type SpeechCallback = (value: string, done: boolean) => void
