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

import React, { useEffect, useState } from 'react'
import type { ClientAPI } from '../../../electron/preload'
import { useTranslation } from 'react-i18next'
import CustomDrawer from '../custom-drawer/custom-drawer'
import ModelCard from '../model-card/model-card'
import { ModelDetails, ModelId } from '../../../electron/types'
import {
    Backdrop,
    CircularProgress,
    Snackbar,
    Stack,
    Typography,
} from '@mui/material'
import { themeSettings } from '../../theme/theme'
import CustomLoadingBackdrop from '../custom-loading-backdrop/custom-loading-backdrop'

const clientAPI = (window as any).clientAPI as typeof ClientAPI

export default function ModelDrawer({
    openDrawer,
    onDrawerClosed,
}: {
    openDrawer: boolean
    onDrawerClosed: () => void
}) {
    const [supportedModels, setSupportedModels] = useState<ModelDetails[]>(
        clientAPI.getSupportedModels()
    )
    const [activeModel, setActiveModel] = useState<ModelId>(
        clientAPI.getActiveModel()
    )
    const [activeModelName, setActiveModeName] = useState<string>(
        clientAPI.getActiveModelDetails().name
    )

    const [modelSelectionInProgress, setModelSelectionInProgress] =
        useState<ModelId>(null)

    const [modelDownloadInProgress, setModelDownloadInProgress] =
        useState<ModelId>(null)

    const [modelInstallUnderProgress, setModelInstallUnderProgress] =
        useState<string>(null)
    const [toastMessage, setToastMessage] = useState<string>(null)

    const { t } = useTranslation()

    useEffect(() => {
        const supportedModelListener = clientAPI.onSupportedModelsUpdated(
            () => {
                setSupportedModels(clientAPI.getSupportedModels())
            }
        )

        const activeModelUpdateListener = clientAPI.onActiveModelUpdate(
            (id) => {
                setModelSelectionInProgress(null)
                setActiveModel(id)
                setActiveModeName(clientAPI.getActiveModelDetails().name)
                clientAPI.resetChat()
            }
        )

        const activeModelUpdateErrorListener =
            clientAPI.onActiveModelUpdateError((modelId) => {
                setModelSelectionInProgress(null)
            })

        const datasetchangeListener = clientAPI.onDatasetInfoUpdate(() => {
            setSupportedModels(clientAPI.getSupportedModels())
            clientAPI.resetChat()
        })

        const modelDownloadListener = clientAPI.onModelDownloaded((modelId) => {
            setModelDownloadInProgress(null)
            setToastMessage(
                t('modelDownloadSuccess', {
                    modelName: clientAPI.getModelDetails(modelId).name,
                })
            )
        })

        const modelDownloadErrorListener = clientAPI.onModelDownloadError(
            (modelId) => {
                setModelDownloadInProgress(null)
                setToastMessage(
                    t('modelDownloadError', {
                        modelName: clientAPI.getModelDetails(modelId).name,
                    })
                )
            }
        )

        const modelInstallListener = clientAPI.onModelInstalled((modelId) => {
            setModelInstallUnderProgress(null)
            setToastMessage(
                t('modelInstallSuccess', {
                    modelName: clientAPI.getModelDetails(modelId).name,
                })
            )
            clientAPI.resetChat()
        })

        const modelInstallErrorListener = clientAPI.onModelInstallError(
            (modelId) => {
                setModelInstallUnderProgress(null)
                setToastMessage(
                    t('modelInstallError', {
                        modelName: clientAPI.getModelDetails(modelId).name,
                    })
                )
            }
        )

        const modelDeleteListener = clientAPI.onModelDeleted((modelId) => {
            setToastMessage(
                t('modelUninstallSuccess', {
                    modelName: clientAPI.getModelDetails(modelId).name,
                })
            )
        })

        const modelDeleteErrorListener = clientAPI.onModelDeleteError(
            (modelId) => {
                setToastMessage(
                    t('modelUninstallError', {
                        modelName: clientAPI.getModelDetails(modelId).name,
                    })
                )
            }
        )

        return () => {
            supportedModelListener()
            datasetchangeListener()
            modelDownloadListener()
            modelDownloadErrorListener()
            activeModelUpdateListener()
            activeModelUpdateErrorListener()
            modelInstallListener()
            modelInstallErrorListener()
            modelDeleteListener()
            modelDeleteErrorListener()
        }
    }, [])

    const onModelSelectionChange = (modelId: ModelId) => {
        setModelSelectionInProgress(modelId)
        clientAPI.setActiveModel(modelId)
        setActiveModeName(clientAPI.getActiveModelDetails().name)
    }

    const onModelDownloadClick = (modelId: ModelId) => {
        clientAPI.downloadModel(modelId)
        setModelDownloadInProgress(modelId)
    }

    const onModelInstallClick = (modelId: ModelId, modelName: string) => {
        clientAPI.installModel(modelId)
        setModelInstallUnderProgress(modelName)
    }

    const onModelDeleteClick = (modelId: ModelId) => {
        clientAPI.deleteModel(modelId)
    }

    const handleToastClose = (_: any, reason: string) => {
        if (reason === 'clickaway') {
            return
        }

        setToastMessage(null)
    }

    return (
        <CustomDrawer
            drawerHeader={t('selectAIModel')}
            openDrawer={openDrawer}
            onDrawerClosed={onDrawerClosed}
        >
            {supportedModels.map((modelDetails: ModelDetails) => (
                <div key={modelDetails.id}>
                    <ModelCard
                        modelDetails={modelDetails}
                        isActive={activeModel === modelDetails.id}
                        isModelSelectionInProgress={
                            modelSelectionInProgress === modelDetails.id
                        }
                        isModelDownloadInProgress={
                            modelDownloadInProgress === modelDetails.id
                        }
                        disableModelAction={!!modelDownloadInProgress}
                        onModelSelectionChange={onModelSelectionChange}
                        onModelDownloadClick={onModelDownloadClick}
                        onModelInstallClick={() =>
                            onModelInstallClick(
                                modelDetails.id,
                                modelDetails.name
                            )
                        }
                        onModelDeleteClick={onModelDeleteClick}
                        activeModelName={activeModelName}
                    ></ModelCard>
                </div>
            ))}
            <Backdrop
                sx={{ backgroundColor: 'rgba(17, 17, 17)' }}
                open={!!modelInstallUnderProgress}
            >
                <Stack
                    direction={'row'}
                    alignItems={'center'}
                    justifyContent={'center'}
                >
                    <Stack
                        direction={'column'}
                        alignItems={'center'}
                        justifyContent={'center'}
                    >
                        <CircularProgress size={48}></CircularProgress>
                        <Typography
                            sx={{
                                textAlign: 'center',
                                marginTop: '24px',
                                color: 'rgba(255, 255, 255, 0.9)',
                            }}
                            variant="body2"
                        >
                            {t('installModelProgress', {
                                modelName: modelInstallUnderProgress,
                            })}
                        </Typography>
                        <Typography variant="body2">
                            {t('waitFewMin')}
                        </Typography>
                    </Stack>
                </Stack>
            </Backdrop>
            <CustomLoadingBackdrop
                open={!!modelSelectionInProgress}
                loadingText={t('loadingModel')}
            ></CustomLoadingBackdrop>
            <Snackbar
                ContentProps={{
                    sx: {
                        backgroundColor: themeSettings.colors.n600,
                    },
                }}
                open={!!toastMessage}
                autoHideDuration={5000}
                onClose={handleToastClose}
                message={
                    <Typography variant="body2">{toastMessage}</Typography>
                }
            ></Snackbar>
        </CustomDrawer>
    )
}
