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

import { DataFormat, DatasetInfo, DatasetSources, ModelId } from './types'
import { BrowserWindow, dialog, ipcMain, app, shell } from 'electron'
import {
    ACTIVE_MODEL_UPDATE,
    DATASET_INFO_UPDATED,
    MODEL_INSTALLED,
    MODEL_INSTALL_ERROR,
    ON_DATASET_UPDATE,
    ON_DATASET_UPDATE_ERROR,
    ON_INDEX_REGENERATED,
    ON_INDEX_REGENERATE_ERROR,
    ON_PYTHON_ENGINE_INIT,
    ON_PYTHON_ENGINE_INIT_ERROR,
} from './constants'
import Equals from './equals'
import { TRT_LLM_RAG_DIR_PACK } from '../bridge_commands/config-packed'
import { TRT_LLM_RAG_DIR } from '../bridge_commands/config'
import { join } from 'path'
import { existsSync } from 'fs'

const Events = [
    DATASET_INFO_UPDATED,
    MODEL_INSTALLED,
    MODEL_INSTALL_ERROR,
    ACTIVE_MODEL_UPDATE,
    ON_DATASET_UPDATE,
    ON_DATASET_UPDATE_ERROR,
    ON_INDEX_REGENERATED,
    ON_INDEX_REGENERATE_ERROR,
    ON_PYTHON_ENGINE_INIT,
    ON_PYTHON_ENGINE_INIT_ERROR,
] as const

type EventNames = (typeof Events)[number]

const programDataPath = join(
    process.env.PROGRAMDATA,
    'NVIDIA Corporation',
    'chatrtx'
)

export default class DatasetManager {
    private _win!: BrowserWindow
    private _chatbot!: any
    private _datasetInfo: DatasetInfo

    get datasetInfo(): DatasetInfo {
        return this._datasetInfo
    }

    set datasetInfo(value: DatasetInfo) {
        if (Equals(value, this._datasetInfo)) {
            return
        }
        this._datasetInfo = value
        this.emit(DATASET_INFO_UPDATED)
    }

    constructor(win, chatbot) {
        this._win = win
        this._chatbot = chatbot
        this._initIpcEvents()
    }

    private _initIpcEvents() {
        ipcMain.on('getDatasetInfo', (event, modelId) => {
            event.returnValue = this.modifiedDatasetInfo(modelId)
        })

        ipcMain.handle('selectDatasetFolder', (_event) =>
            this.selectDatasetFolder()
        )
        ipcMain.handle('changeDatasetSource', (_event, value) =>
            this.changeDatasetSource(value)
        )
        ipcMain.handle('regenerateIndex', (_event) => this.regenerateIndex())

        ipcMain.handle('openPath', (_event, path: string) =>
            this.openPath(path)
        )

        this.handlePythonEvents()
    }

    private openPath(path: string) {
        const projectPath = app.isPackaged
            ? TRT_LLM_RAG_DIR_PACK
            : TRT_LLM_RAG_DIR
        if (
            path.startsWith(this.datasetInfo.selected_path) ||
            path.startsWith(projectPath) ||
            (path.startsWith(programDataPath) && existsSync(path))
        ) {
            shell.openPath(path)
        }
    }

    private handlePythonEvents() {
        this._chatbot.listenPythonEvents(Events).subscribe((value) => {
            const eventName = value?.channel as EventNames
            this.handleEvents(eventName, value?.data)
        })
    }

    private handleEvents(eventName: EventNames, data) {
        switch (eventName) {
            case ON_PYTHON_ENGINE_INIT:
            case ON_PYTHON_ENGINE_INIT_ERROR:
                this.getDatasetInfo()
                break
            case ACTIVE_MODEL_UPDATE:
            case MODEL_INSTALLED:
            case MODEL_INSTALL_ERROR:
                this.getDatasetInfo()
                break
            case ON_DATASET_UPDATE:
                this.getDatasetInfo()
            case ON_INDEX_REGENERATED:
                this.emit(ON_DATASET_UPDATE)
                break
            case ON_INDEX_REGENERATE_ERROR:
            case ON_DATASET_UPDATE_ERROR:
                this.emit(ON_DATASET_UPDATE_ERROR)
                break
        }
    }

    private modifiedDatasetInfo(modelId: ModelId) {
        const datasetInfo = { ...this.datasetInfo }
        if (modelId === ModelId.Clip) {
            datasetInfo.sources = [DatasetSources.DIRECTORY]
            datasetInfo.supportedDataFormat = DataFormat.Images
        } else {
            datasetInfo.sources = [
                DatasetSources.DIRECTORY,
                DatasetSources.NO_DATASET,
            ]
            datasetInfo.supportedDataFormat = DataFormat.Text
        }
        return datasetInfo
    }

    private getDatasetInfo() {
        this._chatbot
            .getDatasetInfo()
            .then((value) => {
                const datasetinfo = JSON.parse(value)
                this.datasetInfo = datasetinfo
                console.log('getDatasetInfo call returned ', value)
            })
            .catch((error) => {
                console.log('Error for getDatasetInfo call ', error)
            })
    }

    private changeDatasetSource = (newSource: DatasetSources) => {
        if (newSource !== this.datasetInfo.selected) {
            this._chatbot
                .setDatasetSource(newSource)
                .then((value) => {
                    console.log('changeDatasetSource call returned ', value)
                })
                .catch((error) => {
                    console.log('Error for changeDatasetSource call ', error)
                    this.emit(ON_DATASET_UPDATE_ERROR)
                })
        }
    }

    private selectDatasetFolder = () =>
        dialog
            .showOpenDialog({
                properties: ['openDirectory'],
                defaultPath: this.datasetInfo.selected_path,
            })
            .then((res) => {
                if (res.canceled) {
                    this.emit(ON_DATASET_UPDATE)
                    return
                }
                const newFolderPath = res.filePaths[0]
                if (newFolderPath === this.datasetInfo.selected_path) {
                    this.emit(ON_DATASET_UPDATE)
                    return
                }
                this._chatbot
                    .setDatasetPath(newFolderPath)
                    .then((value) => {
                        console.log('selectDatasetFolder call returned ', value)
                    })
                    .catch((error) => {
                        console.log(
                            'Error for selectDatasetFolder call ',
                            error
                        )
                        this.emit(ON_DATASET_UPDATE_ERROR)
                    })
            })

    private regenerateIndex = () => {
        this._chatbot
            .generateIndex()
            .then((value) => {
                console.log('regenerateIndex call returned ', value)
            })
            .catch((error) => {
                console.log('Error for regenerateIndex call ', error)
                this.emit(ON_DATASET_UPDATE_ERROR)
            })
    }

    private emit = (eventName: EventNames, data?: any) =>
        !!this._win &&
        Events.indexOf(eventName as any) > -1 &&
        this._win.webContents.send(eventName, data)
}
