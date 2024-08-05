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

import { BrowserWindow, ipcMain } from 'electron'
import {
    ACTIVE_MODEL_UPDATE,
    ACTIVE_MODEL_UPDATE_ERROR,
    MODEL_DELETED,
    MODEL_DELETE_ERROR,
    MODEL_DOWNLOADED,
    MODEL_DOWNLOAD_ERROR,
    MODEL_INSTALLED,
    MODEL_INSTALL_ERROR,
    ON_PYTHON_ENGINE_INIT,
    ON_PYTHON_ENGINE_INIT_ERROR,
    SUPPORTED_MODEL_UPDATE,
} from './constants'
import Equals from './equals'
import { ModelDetails, ModelId, ModelInfo } from './types'

const Events = [
    SUPPORTED_MODEL_UPDATE,
    ACTIVE_MODEL_UPDATE,
    ACTIVE_MODEL_UPDATE_ERROR,
    MODEL_DOWNLOAD_ERROR,
    MODEL_INSTALL_ERROR,
    MODEL_DELETE_ERROR,
    MODEL_DOWNLOADED,
    MODEL_INSTALLED,
    MODEL_DELETED,
    ON_PYTHON_ENGINE_INIT,
    ON_PYTHON_ENGINE_INIT_ERROR,
] as const

type EventNames = (typeof Events)[number]

export default class ModelManager {
    private _win!: BrowserWindow
    private _chatBot: any

    private _model: ModelId = null

    get model() {
        return this._model
    }

    set model(id: ModelId) {
        if (id === this._model) {
            return
        }

        const model = this.supportedModels.find((m) => m.id === id)
        if (model) {
            this._model = id
            this.emit(ACTIVE_MODEL_UPDATE, id)
        }
    }

    private _supportedModels: ModelDetails[] = []

    get supportedModels() {
        return this._supportedModels
    }

    set supportedModels(data: ModelDetails[]) {
        if (Equals(data, this._supportedModels)) {
            return
        }

        const installedModel = []
        const downloadedAndNotInstalledModel = []
        const availableForDownloadModel = []

        data.forEach((value) => {
            if (value.downloaded && value.setup_finished) {
                installedModel.push(value)
            } else if (value.downloaded && !value.setup_finished) {
                downloadedAndNotInstalledModel.push(value)
            } else {
                availableForDownloadModel.push(value)
            }
        })

        const newData = [
            ...installedModel,
            ...downloadedAndNotInstalledModel,
            ...availableForDownloadModel,
        ]

        if (Equals(newData, this._supportedModels)) {
            return
        }

        this._supportedModels = newData
        console.log('Emiited here', SUPPORTED_MODEL_UPDATE)
        this.emit(SUPPORTED_MODEL_UPDATE)
    }

    constructor(win, chatBot) {
        this._win = win
        this._chatBot = chatBot
        this._initIpcEvents()
    }

    private _initIpcEvents() {
        ipcMain.on('getActiveModel', (event) => {
            event.returnValue = this.model
        })
        ipcMain.on('getModelDetails', (event, modelId) => {
            event.returnValue = this.supportedModels.find(
                (m) => m.id === modelId
            )
        })
        ipcMain.on('getActiveModelDetails', (event) => {
            event.returnValue = this.supportedModels.find(
                (m) => m.id === this.model
            )
        })
        ipcMain.on('getSupportedModels', (event) => {
            event.returnValue = this.supportedModels
        })

        ipcMain.handle('setActiveModel', (_event, id: ModelId) =>
            this.setActiveModel(id)
        )
        ipcMain.handle('deleteModel', (_event, id: ModelId) =>
            this.deleteModel(id)
        )
        ipcMain.handle('downloadModel', (_event, id: ModelId) =>
            this.downloadModel(id)
        )
        ipcMain.handle('installModel', (_event, id: ModelId) =>
            this.installModel(id)
        )
        this.handlePythonEvents()
    }

    private handlePythonEvents() {
        this._chatBot.listenPythonEvents(Events).subscribe((value) => {
            const eventName = value?.channel as EventNames
            this.handleEvents(eventName, value?.data)
        })
    }

    private handleEvents(eventName: EventNames, data) {
        switch (eventName) {
            case ON_PYTHON_ENGINE_INIT:
            case ON_PYTHON_ENGINE_INIT_ERROR:
                this.getModelInfo()
                break
            case ACTIVE_MODEL_UPDATE:
                this.model = data as ModelId
                break
            case MODEL_DOWNLOADED:
            case MODEL_INSTALLED:
            case MODEL_DELETED:
                this.getModelInfo()
                this.emit(eventName, data as ModelId)
                break
            case ACTIVE_MODEL_UPDATE_ERROR:
            case MODEL_DOWNLOAD_ERROR:
            case MODEL_INSTALL_ERROR:
            case MODEL_DELETE_ERROR:
                this.emit(eventName, data as ModelId)
        }
    }

    private getModelInfo() {
        this._chatBot
            .getModelInfo()
            .then((value) => {
                const modelInfo: ModelInfo = JSON.parse(value)
                if (!modelInfo) {
                    console.log('getModelInfo call returned empty', modelInfo)
                } else {
                    console.info(
                        'getModelInfo call returned with value ',
                        modelInfo
                    )
                }

                this.supportedModels = modelInfo.supported.filter(
                    (model: ModelDetails) => model.should_show_in_UI
                )
                this.model = modelInfo.selected
            })
            .catch((error) => {
                console.log('Error for getModelInfo call ', error)
                this.supportedModels = []
                this.model = null
            })
    }

    clear = () => {
        this.supportedModels = []
        this.model = null
    }

    emit = (eventName: EventNames, data?: any) => {
        if (!!this._win && Events.indexOf(eventName as any) > -1) {
            console.log('Emitted event via emit ', eventName)
            this._win.webContents.send(eventName, data)
        }
    }

    private setActiveModel(id) {
        if (id == this.model) {
            return
        }
        this._chatBot
            .setActiveModel(id)
            .then((value) => {
                console.log('setModel call returned ', value)
            })
            .catch((error) => {
                console.log('Error for setModel call ', error)
                this.emit(ACTIVE_MODEL_UPDATE_ERROR, id)
            })
    }

    private installModel = (id: ModelId) => {
        this._chatBot
            .installModel(id)
            .then((value) => {
                console.log('installModel call returned ', value)
            })
            .catch((error) => {
                console.log('Error for installModel call ', error)
                //this.emit(MODEL_INSTALL_ERROR, id)
            })
    }

    private downloadModel = (id: ModelId) => {
        this._chatBot
            .downloadModel(id)
            .then((value) => {
                console.log('downloadModel call returned ', value)
            })
            .catch((error) => {
                console.log('Error for downloadModel call ', error)
                this.emit(MODEL_DOWNLOAD_ERROR, id)
            })
    }

    private deleteModel = (id: ModelId) => {
        this._chatBot
            .deleteModel(id)
            .then((value) => {
                console.log('deleteModel call returned ', value)
            })
            .catch((error) => {
                console.log('Error for deleteModel call ', error)
                this.emit(MODEL_DELETE_ERROR, id)
            })
    }
}
