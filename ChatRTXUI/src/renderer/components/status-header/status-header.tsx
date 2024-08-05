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

import './status-header.scss'
import React, { useEffect, useState } from 'react'
import type { ClientAPI } from '../../../electron/preload'
import Stack from '@mui/material/Stack'
import {
    DatasetInfo,
    DatasetSources,
    ModelDetails,
    ModelFineTuningDetails,
} from '../../../electron/types'
import { useTranslation } from 'react-i18next'
import HeaderCard from '../header-card/header-card'
import CustomLoadingBackdrop from '../custom-loading-backdrop/custom-loading-backdrop'
import { Backdrop, CircularProgress, Typography } from '@mui/material'

const clientAPI = (window as any).clientAPI as typeof ClientAPI

export default function StatusHeader({
    isModelDrawerOpen,
    isDatasetDrawerOpen,
    onModelHeaderCardClicked,
    onDatasetHeaderCardClicked,
}: {
    isModelDrawerOpen: boolean
    isDatasetDrawerOpen: boolean
    onModelHeaderCardClicked: () => void
    onDatasetHeaderCardClicked: () => void
}) {
    const [datasetInfo, setDatasetInfo] = useState<DatasetInfo>(
        clientAPI.getDatasetInfo(clientAPI.getActiveModel())
    )
    const [activeModelDetails, setActiveModelDetails] = useState<ModelDetails>(
        clientAPI.getActiveModelDetails()
    )
    const [fineTuningDetails, setFineTuningDetails] =
        useState<ModelFineTuningDetails>()
    const [selectedProfileName, setSelectedProfileName] = useState<string>('')
    const { t } = useTranslation()
    const [datasetChangeInProgress, setDatasetChangeInProgress] =
        useState(false)

    useEffect(() => {
        const activeModelUpdateListener = clientAPI.onActiveModelUpdate((_) => {
            setActiveModelDetails(clientAPI.getActiveModelDetails())
            setFineTuningDetails(
                clientAPI.getModelFineTuningDetails(activeModelDetails.id)
            )
            if (fineTuningDetails?.selectedProfileId) {
                setSelectedProfileName(
                    fineTuningDetails.FineTuningProfileConfigs.find(
                        (profile) =>
                            profile.profileId ===
                            fineTuningDetails.selectedProfileId
                    ).name
                )
            } else {
                setSelectedProfileName('')
            }
        })
        const datasetInfoUpdateListener = clientAPI.onDatasetInfoUpdate(() =>
            setDatasetInfo(clientAPI.getDatasetInfo(clientAPI.getActiveModel()))
        )

        const datasetUpdateListener = clientAPI.onDatasetUpdate(() => {
            setDatasetChangeInProgress(false)
        })
        const datasetUpdateErrorListener = clientAPI.onDatasetUpdateError(
            () => {
                setDatasetChangeInProgress(false)
            }
        )

        const init = () => {
            const profileId = clientAPI.getModelFineTuningDetails(
                clientAPI.getActiveModelDetails()?.id
            )?.selectedProfileId
            if (profileId) {
                setSelectedProfileName(profileId)
            } else {
                setSelectedProfileName('')
            }
        }

        const onModelFineTuningDetailsUpdate =
            clientAPI.onModelFineTuningDetailsUpdate(() => {
                init()
            })

        return () => {
            activeModelUpdateListener()
            datasetInfoUpdateListener()
            init()
            onModelFineTuningDetailsUpdate()
            datasetUpdateErrorListener()
            datasetUpdateListener()
        }
    }, [])

    return (
        <Stack
            className="model-status-header-container"
            direction="row"
            justifyContent="center"
            alignItems={'stretch'}
        >
            <HeaderCard
                isCardActive={isModelDrawerOpen}
                label={t('modelHeader')}
                subLabel={
                    activeModelDetails?.name +
                    (selectedProfileName
                        ? ' (' + selectedProfileName + ')'
                        : '')
                }
                onHeaderCardClick={onModelHeaderCardClicked}
                isDrawerOpen={isModelDrawerOpen || isDatasetDrawerOpen}
            ></HeaderCard>
            <HeaderCard
                isCardActive={isDatasetDrawerOpen}
                label={t('datasetHeader')}
                subLabel={
                    datasetInfo.selected === DatasetSources.DIRECTORY
                        ? datasetInfo.selected_path
                        : t('aiDefaultDatasetSubHeader')
                }
                onHeaderCardClick={onDatasetHeaderCardClicked}
                isDrawerOpen={isModelDrawerOpen || isDatasetDrawerOpen}
                onResyncClick={
                    datasetInfo.selected === DatasetSources.DIRECTORY
                        ? () => {
                              setDatasetChangeInProgress(true)
                              clientAPI.regenerateIndex()
                          }
                        : null
                }
            ></HeaderCard>
            <CustomLoadingBackdrop
                open={datasetChangeInProgress}
                loadingText={t('loadingDataset')}
            ></CustomLoadingBackdrop>
        </Stack>
    )
}
