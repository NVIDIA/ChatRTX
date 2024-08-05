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

import './dataset-settings.scss'
import React, { useEffect, useState } from 'react'
import type { ClientAPI } from '../../../electron/preload'
import {
    Backdrop,
    CircularProgress,
    MenuItem,
    Select,
    Stack,
    TextField,
    Typography,
} from '@mui/material'
import {
    DataFormat,
    DatasetInfo,
    DatasetSources,
} from '../../../electron/types'
import { useTranslation } from 'react-i18next'
import { themeSettings } from '../../theme/theme'
import CustomIconButton from '../custom-icon-button/custom-icon-button'
import CustomLoadingBackdrop from '../custom-loading-backdrop/custom-loading-backdrop'

const clientAPI = (window as any).clientAPI as typeof ClientAPI

export default function DatasetSettings() {
    const [datasetInfo, setDatasetInfo] = useState<DatasetInfo>(
        clientAPI.getDatasetInfo(clientAPI.getActiveModel())
    )
    const { t } = useTranslation()
    const [modelName, setModelName] = useState<string>(
        clientAPI.getActiveModelDetails()?.name
    )
    const [datasetChangeInProgress, setDatasetChangeInProgress] =
        useState(false)

    useEffect(() => {
        const datasetchangeListener = clientAPI.onDatasetInfoUpdate(() => {
            setDatasetInfo(clientAPI.getDatasetInfo(clientAPI.getActiveModel()))
        })
        const datasetUpdateListener = clientAPI.onDatasetUpdate(() => {
            setDatasetChangeInProgress(false)
        })
        const datasetUpdateErrorListener = clientAPI.onDatasetUpdateError(
            () => {
                setDatasetChangeInProgress(false)
            }
        )
        const activeModelChangeListener = clientAPI.onActiveModelUpdate(() => {
            setDatasetInfo(clientAPI.getDatasetInfo(clientAPI.getActiveModel()))
            setModelName(clientAPI.getActiveModelDetails().name)
        })

        return () => {
            datasetchangeListener()
            datasetUpdateListener()
            datasetUpdateErrorListener()
            activeModelChangeListener()
        }
    }, [])

    const onChangeDatasetSource = (datasetSource: DatasetSources) => {
        setDatasetChangeInProgress(true)
        clientAPI.changeDatasetSource(datasetSource)
    }

    const onFolderSelect = () => {
        setDatasetChangeInProgress(true)
        clientAPI.selectDatasetFolder()
    }

    const onHandleRegerate = () => {
        setDatasetChangeInProgress(true)
        clientAPI.regenerateIndex()
    }

    return (
        <div>
            <Stack
                className="dataset-settings-conatiner"
                flexDirection={'column'}
                alignItems={'stretch'}
                justifyContent={'flex-start'}
                gap={themeSettings.space.md}
            >
                <Select
                    value={datasetInfo.selected}
                    onChange={(event) =>
                        onChangeDatasetSource(
                            event.target.value as DatasetSources
                        )
                    }
                >
                    {datasetInfo.sources.map((value) => (
                        <MenuItem key={value} value={value}>
                            <Typography variant="body2">
                                {value === DatasetSources.DIRECTORY
                                    ? t('chatWithData')
                                    : t('chatWithAI')}
                            </Typography>
                        </MenuItem>
                    ))}
                </Select>
                {datasetInfo.selected === DatasetSources.DIRECTORY && (
                    <Typography variant="body2">
                        {t('datasetDirectoryInfo', { modelName: modelName })}
                    </Typography>
                )}
                {datasetInfo.selected === DatasetSources.NO_DATASET && (
                    <Typography variant="body2">
                        {t('datasetAIDefautInfo')}
                    </Typography>
                )}
                {datasetInfo.selected === DatasetSources.DIRECTORY &&
                    datasetInfo.supportedDataFormat === DataFormat.Text && (
                        <Stack direction={'column'}>
                            <Typography variant="body2">
                                {t('txtSupportCaption')}
                            </Typography>
                            <Typography variant="body2">
                                {t('pdfSupportCaption')}
                            </Typography>
                            <Typography variant="body2">
                                {t('docxSupportCaption')}
                            </Typography>
                        </Stack>
                    )}
                {datasetInfo.selected === DatasetSources.DIRECTORY &&
                    datasetInfo.supportedDataFormat === DataFormat.Images && (
                        <Stack direction={'column'}>
                            <Typography variant="body2">
                                {t('jpgSupportCaption')}
                            </Typography>
                            <Typography variant="body2">
                                {t('pngSupportCaption')}
                            </Typography>
                            <Typography variant="body2">
                                {t('tiffSupportCaption')}
                            </Typography>
                            <Typography variant="body2">
                                {t('rawSupportCaption')}
                            </Typography>
                        </Stack>
                    )}
                {datasetInfo.selected === DatasetSources.DIRECTORY && (
                    <div>
                        <Stack
                            direction={'row'}
                            alignItems={'center'}
                            justifyContent={'center'}
                            spacing={themeSettings.space.md}
                        >
                            <CustomIconButton
                                type="folder"
                                tooltip={t('browseFolderPath')}
                                onClick={() => onFolderSelect()}
                            ></CustomIconButton>
                            <Stack
                                className="folder-text-filed"
                                direction={'column'}
                                alignItems={'stretch'}
                                justifyContent={'flex-start'}
                            >
                                <TextField
                                    required
                                    label={t('dataFolder')}
                                    variant="filled"
                                    value={datasetInfo.selected_path}
                                    onClick={() => onFolderSelect()}
                                />
                            </Stack>
                            <CustomIconButton
                                type="common-refresh"
                                tooltip={t('refreshDataset')}
                                onClick={() => onHandleRegerate()}
                            ></CustomIconButton>
                        </Stack>
                    </div>
                )}
            </Stack>
            <CustomLoadingBackdrop
                open={datasetChangeInProgress}
                loadingText={t('loadingDataset')}
            ></CustomLoadingBackdrop>
        </div>
    )
}
