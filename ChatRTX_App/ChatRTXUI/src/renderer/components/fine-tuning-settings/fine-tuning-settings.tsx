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

import './fine-tuning-settings.scss'
import React, { useState } from 'react'
import type { ClientAPI } from '../../../electron/preload'
import { FineTuningProfileConfig, ModelId } from '../../../electron/types'
import {
    Button,
    Link,
    Modal,
    Stack,
    TextField,
    Typography,
} from '@mui/material'
import { useTranslation } from 'react-i18next'
import { themeSettings } from '../../theme/theme'
import IconElement from '../icon-element/icon-element'
import FineTuningParam from '../fine-tuning-param/fine-tuning-param'

const clientAPI = (window as any).clientAPI as typeof ClientAPI

export default function FineTuningCardSettings({
    modelId,
    profile,
    isNewStyle,
    onCancel,
    onCreateNewProfile,
}: {
    modelId?: ModelId
    profile?: FineTuningProfileConfig
    isNewStyle?: boolean
    onCancel?: () => void
    onCreateNewProfile?: (profileConfig: FineTuningProfileConfig) => void
}) {
    const [isFineTuneButtonDisbaled, setIsFineTunedModelDisabled] =
        useState(true)
    const [profileConfiguration, setProfileConfiguration] =
        useState<FineTuningProfileConfig>(
            profile
                ? profile
                : {
                      name: '',
                      folderPath: '',
                      advancedParams:
                          clientAPI.getDefaultAdvancedParams(modelId),
                  }
        )
    const [parentModelName] = useState(clientAPI.getModelDetails(modelId)?.name)
    const [fineTuningModelName] = useState(
        clientAPI.getModelFineTuningDetails(modelId)?.baseModelName
    )
    const { t } = useTranslation()

    const onClickFolder = () => {
        clientAPI.selectFolder().then((value) => {
            if (value) {
                setProfileConfiguration((config) => {
                    setIsFineTunedModelDisabled(!config.name || !value)
                    return {
                        ...config,
                        folderPath: value,
                    }
                })
            }
        })
    }

    const setName = (value: string) => {
        setProfileConfiguration((config) => {
            setIsFineTunedModelDisabled(!config.folderPath || !value)
            return {
                ...config,
                name: value,
            }
        })
    }

    const [openCustomSettings, setOpenCustomSettings] = React.useState(false)
    const handleOpen = () => {
        setOpenCustomSettings(true)
    }
    const handleClose = () => {
        setOpenCustomSettings(false)
    }

    return (
        <div>
            <Stack
                className={
                    isNewStyle
                        ? 'fine-tuning-card-container new-style-container'
                        : 'fine-tuning-card-container'
                }
                direction={'column'}
                alignItems={'stretch'}
                justifyContent={'flex-start'}
                spacing={themeSettings.space.ms}
                flexGrow={1}
            >
                {isNewStyle && (
                    <Typography variant="body1">{t('addStyle')}</Typography>
                )}
                <TextField
                    required
                    label={t('styleNameLabel')}
                    type="search"
                    variant="filled"
                    value={profileConfiguration.name}
                    onChange={(e) => setName(e.target.value)}
                    disabled={!isNewStyle}
                />
                <Stack
                    direction={'row'}
                    alignItems={'flex-start'}
                    justifyContent={'start'}
                >
                    <Button
                        color="transparent"
                        className="grey-btn square-btn slim-btn light-grey-hover-btn"
                        onClick={isNewStyle ? onClickFolder : () => {}}
                    >
                        <IconElement color="n050" type="folder" size="ml" />
                    </Button>
                    <Stack
                        className="folder-text-filed"
                        direction={'column'}
                        alignItems={'stretch'}
                        justifyContent={'flex-start'}
                    >
                        <TextField
                            required
                            label={t('exampleFolderLabel')}
                            variant="filled"
                            value={profileConfiguration.folderPath}
                            onClick={
                                profileConfiguration.folderPath
                                    ? () => {}
                                    : onClickFolder
                            }
                            disabled={!isNewStyle}
                        />
                        <Typography variant="caption">
                            {t('msgSupportCaption')}
                        </Typography>
                        <Link onClick={handleOpen}>
                            <Typography variant="caption">
                                {t('customSettings')}
                            </Typography>
                        </Link>
                    </Stack>
                </Stack>
                {isNewStyle && (
                    <Stack
                        direction={'row'}
                        alignItems={'flex-end'}
                        justifyContent={'flex-end'}
                        spacing={themeSettings.space.ms}
                    >
                        <Button
                            variant="text"
                            color="transparent"
                            onClick={onCancel}
                        >
                            {t('CANCEL')}
                        </Button>
                        <Button
                            disabled={isFineTuneButtonDisbaled}
                            onClick={() =>
                                onCreateNewProfile(profileConfiguration)
                            }
                            variant="contained"
                        >
                            {t('FINETUNE')}
                        </Button>
                    </Stack>
                )}
            </Stack>
            <Modal className="custom-settings-modal" open={openCustomSettings}>
                <Stack
                    className="custom-settings-dialog"
                    direction={'column'}
                    alignItems={'stretch'}
                    justifyContent={'flex-start'}
                >
                    <Typography variant="h6bold">
                        {t('fineTuningCustomSettings')}
                    </Typography>
                    <Typography variant="body1">
                        {t('aiModelName', { modelName: parentModelName })}
                    </Typography>
                    <Typography variant="body1">
                        {t('downloadedBase', {
                            modelName: fineTuningModelName,
                        })}
                    </Typography>
                    <Stack
                        direction={'column'}
                        alignItems={'stretch'}
                        justifyContent={'flex-start'}
                        spacing={themeSettings.space.lg}
                        className="custom-settings-content"
                    >
                        <Stack
                            direction={'column'}
                            alignItems={'stretch'}
                            justifyContent={'flex-start'}
                            spacing={themeSettings.space.md}
                        >
                            <TextField
                                required
                                label={t('styleNameLabel')}
                                type="search"
                                variant="filled"
                                defaultValue={profileConfiguration.name}
                                onChange={(e) => setName(e.target.value)}
                                disabled={!isNewStyle}
                            />
                            <Stack
                                direction={'row'}
                                alignItems={'center'}
                                justifyContent={'flex-start'}
                                spacing={themeSettings.space.md}
                            >
                                <Stack
                                    direction={'row'}
                                    alignItems={'flex-start'}
                                    justifyContent={'flex-start'}
                                    flexGrow={1}
                                >
                                    <Button
                                        color="transparent"
                                        className="grey-btn square-btn slim-btn light-grey-hover-btn"
                                        onClick={
                                            isNewStyle
                                                ? onClickFolder
                                                : () => {}
                                        }
                                    >
                                        <IconElement
                                            color="n050"
                                            type="folder"
                                            size="ml"
                                        />
                                    </Button>
                                    <Stack
                                        className="folder-text-filed"
                                        direction={'column'}
                                        alignItems={'stretch'}
                                        justifyContent={'flex-start'}
                                        flexGrow={1}
                                    >
                                        <TextField
                                            required
                                            label={t('exampleFolderLabel')}
                                            variant="filled"
                                            value={
                                                profileConfiguration.folderPath
                                            }
                                            onClick={
                                                profileConfiguration.folderPath
                                                    ? () => {}
                                                    : onClickFolder
                                            }
                                            disabled={!isNewStyle}
                                        />
                                        <Typography variant="caption">
                                            {t('msgSupportCaption')}
                                        </Typography>
                                    </Stack>
                                </Stack>
                                <Stack
                                    direction={'row'}
                                    alignItems={'flex-start'}
                                    justifyContent={'flex-start'}
                                    flexGrow={1}
                                >
                                    <Button
                                        color="transparent"
                                        className="grey-btn square-btn slim-btn light-grey-hover-btn"
                                        onClick={
                                            isNewStyle
                                                ? onClickFolder
                                                : () => {}
                                        }
                                    >
                                        <IconElement
                                            color="n050"
                                            type="folder"
                                            size="ml"
                                        />
                                    </Button>
                                    <Stack
                                        className="test-folder-text-filed"
                                        direction={'column'}
                                        alignItems={'stretch'}
                                        justifyContent={'flex-start'}
                                        flexGrow={1}
                                    >
                                        <TextField
                                            label={t('TestFolderLabel')}
                                            variant="filled"
                                            value={
                                                profileConfiguration.testFolderPath
                                            }
                                            onClick={
                                                profileConfiguration.testFolderPath
                                                    ? () => {}
                                                    : onClickFolder
                                            }
                                            disabled={!isNewStyle}
                                        />
                                        <Typography variant="caption">
                                            {t('msgSupportCaption')}
                                        </Typography>
                                    </Stack>
                                </Stack>
                            </Stack>
                            <TextField
                                label={t('SystemPrompt')}
                                variant="filled"
                                maxRows={5}
                                disabled={!isNewStyle}
                                defaultValue={
                                    profileConfiguration.advancedParams
                                        .systemPrompt
                                }
                                multiline
                            />
                        </Stack>
                        <Stack
                            direction={'column'}
                            alignItems={'stretch'}
                            justifyContent={'flex-start'}
                            spacing={themeSettings.space.ml}
                        >
                            <Stack
                                direction={'row'}
                                alignItems={'stretch'}
                                justifyContent={'space-between'}
                                spacing={themeSettings.space.md}
                            >
                                <FineTuningParam
                                    disabled={!isNewStyle}
                                    paramName="LoRA rank"
                                    value={
                                        profileConfiguration.advancedParams
                                            .loraRank
                                    }
                                    paramDetails="Set the rank for Low-Rank Adaptation. Higher ranks mean more parameters and detail, but require more memory."
                                ></FineTuningParam>
                                <FineTuningParam
                                    disabled={!isNewStyle}
                                    paramName="Save steps"
                                    value={
                                        profileConfiguration.advancedParams
                                            .saveSteps
                                    }
                                    paramDetails="Specify the number of iterations after which checkpoints are saved."
                                ></FineTuningParam>
                            </Stack>
                            <Stack
                                direction={'row'}
                                alignItems={'stretch'}
                                justifyContent={'space-between'}
                                spacing={themeSettings.space.md}
                            >
                                <FineTuningParam
                                    disabled={!isNewStyle}
                                    paramName="Epochs"
                                    value={
                                        profileConfiguration.advancedParams
                                            .epochs
                                    }
                                    paramDetails="Determine the number of times the entire dataset is fed into the model for training."
                                ></FineTuningParam>
                                <FineTuningParam
                                    disabled={!isNewStyle}
                                    paramName="Context length"
                                    value={
                                        profileConfiguration.advancedParams
                                            .contextLenght
                                    }
                                    paramDetails="Set the length of input text segments used for training. Longer context lengths improve performance but require more memory."
                                ></FineTuningParam>
                            </Stack>
                        </Stack>
                        {isNewStyle && (
                            <Stack
                                direction={'row'}
                                alignItems={'flex-end'}
                                justifyContent={'flex-end'}
                                spacing={themeSettings.space.ms}
                            >
                                <Button
                                    variant="text"
                                    color="transparent"
                                    onClick={handleClose}
                                >
                                    {t('CANCEL')}
                                </Button>
                                <Button
                                    disabled={isFineTuneButtonDisbaled}
                                    onClick={() =>
                                        onCreateNewProfile(profileConfiguration)
                                    }
                                    variant="contained"
                                >
                                    {t('FINETUNE')}
                                </Button>
                            </Stack>
                        )}
                        {!isNewStyle && (
                            <Stack
                                direction={'row'}
                                alignItems={'flex-end'}
                                justifyContent={'flex-end'}
                                spacing={themeSettings.space.ms}
                            >
                                <Button
                                    variant="text"
                                    color="transparent"
                                    onClick={handleClose}
                                >
                                    {t('CLOSE')}
                                </Button>
                            </Stack>
                        )}
                    </Stack>
                </Stack>
            </Modal>
        </div>
    )
}
