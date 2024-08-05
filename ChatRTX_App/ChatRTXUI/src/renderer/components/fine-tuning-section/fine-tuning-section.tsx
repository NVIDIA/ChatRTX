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

import './fine-tuning-section.scss'
import React, { useEffect, useState } from 'react'
import type { ClientAPI } from '../../../electron/preload'
import {
    FineTuningProfileConfig,
    ModelFineTuningDetails,
    ModelId,
} from '../../../electron/types'
import {
    Backdrop,
    Button,
    CircularProgress,
    Stack,
    Switch,
    Typography,
} from '@mui/material'
import { useTranslation } from 'react-i18next'
import FineTuningCard from '../fine-tuning-card/fine-tuning-card'
import { themeSettings } from '../../theme/theme'

const clientAPI = (window as any).clientAPI as typeof ClientAPI

export default function FineTuningSection({ modelId }: { modelId: ModelId }) {
    const [isFineTuningActive, setIsFineTuningActive] = useState(
        clientAPI.isFineTuningEnabled(modelId)
    )
    const [isFineTuningToggleInProcess, setIsFineTuningToggleInProcess] =
        useState(false)
    const [modelFineTuningDetails, setModelFineTuningDetails] =
        useState<ModelFineTuningDetails>(
            clientAPI.getModelFineTuningDetails(modelId)
        )
    const [isBaseModelDownloadInit, setIsBaseModelDownloadInit] =
        useState<boolean>(
            clientAPI.getModelFineTuningDetails(modelId).isBaseModelDownloaded
        )
    const [isBaseModelDownloading, setIsBaseModeDownloading] =
        useState<boolean>(false)
    const [showFineTuningCard, setShowFineTuningCard] = useState(
        clientAPI.getModelFineTuningDetails(modelId).FineTuningProfileConfigs
            .length === 0
    )

    const [currentActivatingProfile, setCurrentActivatingProfile] = useState('')

    const { t } = useTranslation()

    const [
        isCreatingFineTuningUnderProcess,
        setIsCreatingFineTuningUnderProcess,
    ] = useState(false)

    useEffect(() => {
        const modelFineTuningDetailsUpdateListener =
            clientAPI.onModelFineTuningDetailsUpdate(() => {
                console.log(
                    'Fine tuning details updating ',
                    clientAPI.getModelFineTuningDetails(modelId)
                )
                setModelFineTuningDetails(
                    clientAPI.getModelFineTuningDetails(modelId)
                )
                setCurrentActivatingProfile('')
                setIsCreatingFineTuningUnderProcess(false)
                setIsFineTuningActive(clientAPI.isFineTuningEnabled(modelId))
                setIsFineTuningToggleInProcess(false)
            })

        const onCreateFineTuningListener = clientAPI.onFineTuningProfileCreated(
            (parentId: string) => {
                if (parentId === modelId) {
                    setShowFineTuningCard(false)
                }
            }
        )
        return () => {
            modelFineTuningDetailsUpdateListener()
            onCreateFineTuningListener()
        }
    }, [])

    const activateFineTuning = (event: React.ChangeEvent<HTMLInputElement>) => {
        setIsFineTuningToggleInProcess(true)
        clientAPI.toggleFineTuningEnablement(modelId, event.target.checked)
        if (
            event.target.checked &&
            modelFineTuningDetails.FineTuningProfileConfigs.length === 0
        ) {
            setShowFineTuningCard(true)
        }
        if (!isBaseModelDownloadInit) {
            // TODO: Start downloading base model only once
            console.log('Downloading base model ', isBaseModelDownloadInit)
            setIsBaseModelDownloadInit(true)
            setIsBaseModeDownloading(true)
        }
        // TODO: Decide whether to select by default profile or do it when explictly selected
        // if (modelFineTuningDetails.FineTuningProfileConfigs.length === 1) {
        //     onProfileSelected(modelFineTuningDetails.FineTuningProfileConfigs[0].profileId)
        // } else if (modelFineTuningDetails.FineTuningProfileConfigs.length && modelFineTuningDetails.selectedProfileId) {
        //     if (modelFineTuningDetails.FineTuningProfileConfigs.find(value => value.profileId === modelFineTuningDetails.selectedProfileId)) {
        //         onProfileSelected(modelFineTuningDetails.selectedProfileId)
        //     }
        // }
    }

    const onCancelFineTune = () => {
        setShowFineTuningCard(false)
        if (modelFineTuningDetails.FineTuningProfileConfigs.length === 0) {
            setIsFineTuningActive(false)
        }
    }

    const addFineTuningCard = () => {
        setShowFineTuningCard(true)
    }

    const createFineTunedModel = (
        profileConfiguration: FineTuningProfileConfig
    ) => {
        clientAPI.createFineTuningProfile(modelId, profileConfiguration)
        setIsCreatingFineTuningUnderProcess(true)
    }

    const onProfileSelected = (value: string) => {
        if (
            value === currentActivatingProfile ||
            value === modelFineTuningDetails.selectedProfileId
        ) {
            return
        }
        setCurrentActivatingProfile(value)
        clientAPI.selectFineTuningModel(modelId, value)
    }

    const onProfileDeleted = (profileId: string) => {
        clientAPI.deleteFineTuningProfile(modelId, profileId)
        setModelFineTuningDetails((fineTuningDetails) => {
            const profiles = fineTuningDetails.FineTuningProfileConfigs.filter(
                (profile) => profile.profileId !== profileId
            )
            fineTuningDetails.FineTuningProfileConfigs = profiles
            return { ...fineTuningDetails }
        })
    }

    return (
        <Stack
            direction={'column'}
            alignItems={'stretch'}
            justifyContent={'flex-start'}
        >
            <Stack
                direction={'row'}
                alignItems={'center'}
                justifyContent={'space-between'}
                className="fine-tuning-header"
            >
                <Typography variant="body1bold">{t('fineTuning')}</Typography>
                {!isFineTuningToggleInProcess && (
                    <Switch
                        checked={isFineTuningActive}
                        onChange={activateFineTuning}
                    />
                )}
                {isFineTuningToggleInProcess && (
                    <div className="circular-progress">
                        <CircularProgress size={18}></CircularProgress>
                    </div>
                )}
            </Stack>
            <Typography>{t('fineTuningDescription')}</Typography>
            {isFineTuningActive && (
                <Stack
                    direction={'column'}
                    alignContent={'stretch'}
                    justifyContent={'flex-start'}
                >
                    <div className="download-info">
                        {isBaseModelDownloading && (
                            <Typography variant="body1bold">
                                {t('downloadingBase') + ' %'}
                            </Typography>
                        )}
                        {modelFineTuningDetails.isBaseModelDownloaded && (
                            <Typography variant="body1">
                                {t('downloadedBase', {
                                    modelName:
                                        modelFineTuningDetails.baseModelName,
                                })}
                            </Typography>
                        )}
                    </div>
                    {modelFineTuningDetails.FineTuningProfileConfigs.map(
                        (profile) => (
                            <div key={profile.profileId}>
                                <FineTuningCard
                                    isActive={
                                        profile.profileId ===
                                        modelFineTuningDetails.selectedProfileId
                                    }
                                    showRadio={true}
                                    isActivating={
                                        profile.profileId ===
                                        currentActivatingProfile
                                    }
                                    modelId={modelId}
                                    profile={profile}
                                    onSelect={onProfileSelected}
                                    onProfileDelete={onProfileDeleted}
                                ></FineTuningCard>
                            </div>
                        )
                    )}
                    {showFineTuningCard && (
                        <div className="new-style-container">
                            <FineTuningCard
                                modelId={modelFineTuningDetails.parentModelId}
                                isNewStyle={true}
                                onCancel={onCancelFineTune}
                                onCreateNewProfile={(
                                    profileConfig: FineTuningProfileConfig
                                ) => createFineTunedModel(profileConfig)}
                            ></FineTuningCard>
                        </div>
                    )}
                    {modelFineTuningDetails.FineTuningProfileConfigs.length !==
                        0 && (
                        <Stack
                            className="add-style-button"
                            direction={'row'}
                            alignItems={'center'}
                            justifyContent={'flex-end'}
                        >
                            <Button
                                disabled={showFineTuningCard}
                                variant="text"
                                color="transparent"
                                onClick={addFineTuningCard}
                            >
                                {t('ADDSTYLE')}
                            </Button>
                        </Stack>
                    )}
                </Stack>
            )}
            <Backdrop
                sx={{ backgroundColor: themeSettings.colors.n900 }}
                open={isCreatingFineTuningUnderProcess}
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
                        <CircularProgress size={40}></CircularProgress>
                        <Typography
                            sx={{
                                textAlign: 'center',
                                margin: '8px',
                            }}
                            variant="body1bold"
                        >
                            Fine Tuning Model...
                        </Typography>
                    </Stack>
                </Stack>
            </Backdrop>
        </Stack>
    )
}
