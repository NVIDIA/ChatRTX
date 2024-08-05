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

import './model-card.scss'
import React, { useRef, useState } from 'react'
import type { ClientAPI } from '../../../electron/preload'
import { ModelDetails, ModelId } from '../../../electron/types'
import {
    Button,
    Checkbox,
    CircularProgress,
    Menu,
    MenuItem,
    Stack,
    Tooltip,
    Typography,
} from '@mui/material'
import { Trans, useTranslation } from 'react-i18next'
import { themeSettings } from '../../theme/theme'
import FineTuningSection from '../fine-tuning-section/fine-tuning-section'
import IconElement from '../icon-element/icon-element'
import CustomModalDialog from '../custom-modal-dialog/custom-modal-dialog'
import CustomIconButton from '../custom-icon-button/custom-icon-button'

const clientAPI = (window as any).clientAPI as typeof ClientAPI

export default function ModelCard({
    modelDetails,
    isActive,
    isModelSelectionInProgress,
    isModelDownloadInProgress,
    disableModelAction,
    activeModelName,
    onModelSelectionChange,
    onModelDownloadClick,
    onModelInstallClick,
    onModelDeleteClick,
}: {
    modelDetails: ModelDetails
    isActive: boolean
    isModelSelectionInProgress: boolean
    isModelDownloadInProgress: boolean
    disableModelAction: boolean
    activeModelName: string
    onModelSelectionChange: (id: ModelId) => void
    onModelDownloadClick: (id: ModelId) => void
    onModelInstallClick: () => void
    onModelDeleteClick: (id: ModelId) => void
}) {
    const [moreOptionsAnchorEl, setMoreOptionsAnchorEl] =
        React.useState<null | HTMLElement>(null)
    const openMoreOptions = Boolean(moreOptionsAnchorEl)
    const [openModelDownloadDialog, setOpenModelDownloadDialog] =
        useState(false)
    const [openModelInstallDialog, setOpenModelInstallDialog] = useState(false)
    const [openModelUninstallDialog, setOpenModelUninstallDialog] =
        useState(false)
    const [showMoreMenu, setShowMoreMenu] = useState(false)
    const { t } = useTranslation()
    const cardRef = useRef<HTMLDivElement>(null)

    const handleMoreOptionsClick = (
        event: React.MouseEvent<HTMLButtonElement>
    ) => {
        event.stopPropagation()
        setMoreOptionsAnchorEl(event.currentTarget)
    }
    const handleMoreOptionsClosed = (
        event: React.MouseEvent<HTMLButtonElement>
    ) => {
        event.stopPropagation()
        setMoreOptionsAnchorEl(null)
    }

    const onDelete = (
        event: React.MouseEvent<HTMLButtonElement, MouseEvent>
    ) => {
        event.stopPropagation()
        setMoreOptionsAnchorEl(null)
        onModelDeleteClick(modelDetails.id)
    }

    const onCardClicked = (
        event: React.MouseEvent<HTMLDivElement, MouseEvent>
    ) => {
        const containsTarget = cardRef?.current?.contains(
            event.target as Element
        )
        if (
            modelDetails.setup_finished &&
            modelDetails.downloaded &&
            !isActive &&
            containsTarget
        ) {
            onModelSelectionChange(modelDetails.id)
        }
    }

    return (
        <div
            ref={cardRef}
            className={`model-card${isActive ? ' active-model-card' : ''}${modelDetails.setup_finished && modelDetails.downloaded ? ' installed-model card-button' : ''}`}
            onClick={(event) => onCardClicked(event)}
            onMouseEnter={() => setShowMoreMenu(true)}
            onMouseLeave={() => {
                setMoreOptionsAnchorEl(null)
                setShowMoreMenu(false)
            }}
        >
            <Stack
                direction={'column'}
                alignItems={'stretch'}
                justifyContent={'flex-start'}
                data-isinstalled={
                    modelDetails.downloaded && modelDetails.setup_finished
                }
            >
                <Stack
                    direction={'row'}
                    alignItems={'flex-start'}
                    justifyContent={'space-between'}
                >
                    <Stack
                        direction={'row'}
                        alignItems={'stretch'}
                        justifyContent={'flex-start'}
                    >
                        <div
                            data-isactive={isActive}
                            className="active-model-status"
                        ></div>
                        <Stack
                            direction={'column'}
                            alignItems={'flex-start'}
                            justifyContent={'center'}
                        >
                            <Stack
                                direction={'row'}
                                alignItems={'center'}
                                justifyContent={'flex-start'}
                            >
                                {!isModelDownloadInProgress && (
                                    <Typography variant="subtitle1">
                                        {modelDetails.name} &nbsp;
                                    </Typography>
                                )}
                                {isModelDownloadInProgress && (
                                    <Typography variant="body1bold">
                                        {t('downloadModelProgress', {
                                            modelName: modelDetails.name,
                                            progress: '',
                                        })}
                                    </Typography>
                                )}
                                {isActive && (
                                    <Typography variant="body1">
                                        {t('inUse')}
                                    </Typography>
                                )}
                            </Stack>
                            <Typography variant="body2">
                                {t('modelDeveloperDesc', {
                                    modelDevelopers:
                                        modelDetails.modelDevelopers,
                                })}
                            </Typography>
                            <Typography variant="body2">
                                {modelDetails.model_size}
                            </Typography>
                        </Stack>
                    </Stack>

                    {modelDetails.setup_finished && (
                        <Stack
                            direction={'row'}
                            alignItems={'flex-start'}
                            justifyContent={'flex-end'}
                        >
                            {showMoreMenu && (
                                <CustomIconButton
                                    type="menu-3dots"
                                    tooltip={t('viewMore')}
                                    onClick={handleMoreOptionsClick}
                                ></CustomIconButton>
                            )}
                            <Menu
                                id="more-options-menu"
                                anchorEl={moreOptionsAnchorEl}
                                open={openMoreOptions}
                                onClose={handleMoreOptionsClosed}
                                MenuListProps={{
                                    'aria-labelledby': 'more-options-button',
                                }}
                            >
                                <MenuItem
                                    disabled={isActive || disableModelAction}
                                    onClick={(event) => {
                                        setOpenModelUninstallDialog(true)
                                        event.stopPropagation()
                                    }}
                                >
                                    {t('unInstall')}
                                </MenuItem>
                            </Menu>
                            {isActive && !isModelSelectionInProgress && (
                                <Tooltip title={t('selectedAiModel')}>
                                    <Checkbox
                                        className="checkbox"
                                        icon={
                                            <IconElement
                                                type="checkbox-off"
                                                size={
                                                    themeSettings.iconSizes
                                                        .large
                                                }
                                            />
                                        }
                                        checkedIcon={
                                            <IconElement
                                                type="checkbox-on"
                                                size={
                                                    themeSettings.iconSizes
                                                        .large
                                                }
                                            />
                                        }
                                        defaultChecked={true}
                                    />
                                </Tooltip>
                            )}
                            {!isActive && !isModelSelectionInProgress && (
                                <Tooltip title={t('selectAiModel')}>
                                    <Checkbox
                                        className="checkbox"
                                        icon={
                                            <IconElement
                                                type="checkbox-off"
                                                size={
                                                    themeSettings.iconSizes
                                                        .large
                                                }
                                            />
                                        }
                                        defaultChecked={false}
                                        onClick={(event) => {
                                            onModelSelectionChange(
                                                modelDetails.id
                                            )
                                            event.stopPropagation()
                                        }}
                                        disabled={disableModelAction}
                                    />
                                </Tooltip>
                            )}
                            {isModelSelectionInProgress && (
                                <div className="loading-spinner">
                                    <CircularProgress
                                        size={themeSettings.iconSizes.regular}
                                    ></CircularProgress>
                                </div>
                            )}
                        </Stack>
                    )}
                    {!modelDetails.downloaded && !isModelDownloadInProgress && (
                        <Stack
                            direction={'row'}
                            alignItems={'flex-start'}
                            justifyContent={'flex-end'}
                            spacing={themeSettings.space.xs}
                        >
                            <CustomIconButton
                                type="common-download"
                                tooltip={t('downloadAiModel')}
                                disabled={disableModelAction}
                                onClick={(event) => {
                                    event.stopPropagation()
                                    setOpenModelDownloadDialog(true)
                                }}
                            ></CustomIconButton>
                        </Stack>
                    )}
                    {modelDetails.downloaded &&
                        !modelDetails.setup_finished && (
                            <Stack
                                direction={'row'}
                                alignItems={'flex-start'}
                                justifyContent={'flex-end'}
                                spacing={themeSettings.space.xs}
                            >
                                <Button
                                    variant="contained"
                                    color="primary"
                                    onClick={(event) => {
                                        setOpenModelInstallDialog(true)
                                        event.stopPropagation()
                                    }}
                                    disabled={disableModelAction}
                                >
                                    {t('INSTALL')}
                                </Button>
                            </Stack>
                        )}
                </Stack>
                <Stack
                    direction={'column'}
                    alignItems={'stretch'}
                    justifyContent={'flex-start'}
                >
                    {modelDetails.isTextBased &&
                        modelDetails.isChineseSupported &&
                        modelDetails.isEnglishSupported && (
                            <Typography variant="body2" className="model-info">
                                {t('textBasedEnglishChinese')}
                            </Typography>
                        )}
                    {modelDetails.isTextBased &&
                        !modelDetails.isChineseSupported &&
                        modelDetails.isEnglishSupported && (
                            <Typography variant="body2" className="model-info">
                                {t('textBasedEnglish')}
                            </Typography>
                        )}
                    {modelDetails.isTextBased &&
                        modelDetails.isChineseSupported &&
                        !modelDetails.isEnglishSupported && (
                            <Typography variant="body2" className="model-info">
                                {t('textBasedChinese')}
                            </Typography>
                        )}
                    {modelDetails.isImageBased && (
                        <Typography variant="body2" className="model-info">
                            {t('imageBased')}
                        </Typography>
                    )}
                    <Typography variant="body2">
                        <Trans
                            i18nKey={'learnMore'}
                            components={{
                                anchorlink: (
                                    <a
                                        target="_blank"
                                        onClick={(event) => {
                                            event.stopPropagation()
                                            clientAPI.openLink(
                                                modelDetails.model_license
                                            )
                                        }}
                                    ></a>
                                ),
                            }}
                        />
                    </Typography>
                </Stack>
                {modelDetails.isFineTuningSupported && (
                    <FineTuningSection
                        modelId={modelDetails.id}
                    ></FineTuningSection>
                )}
                <CustomModalDialog
                    dialogHeader={t('downloadModel', {
                        modelName: modelDetails.name,
                    })}
                    openDialog={openModelDownloadDialog}
                    primaryButtonText={t('AGREE&CONTINUE')}
                    secondaryButtonText={t('CANCEL')}
                    onPrimaryButtonClick={() => {
                        setOpenModelDownloadDialog(false),
                            onModelDownloadClick(modelDetails.id)
                    }}
                    onSecondaryButtonClick={() =>
                        setOpenModelDownloadDialog(false)
                    }
                >
                    <Stack
                        flexDirection={'column'}
                        alignItems={'stretch'}
                        justifyContent={'flex-start'}
                    >
                        <Typography variant="body2">
                            {t('downloadTermsDescription')}
                        </Typography>
                        <Typography variant="body2">
                            <Trans
                                i18nKey={'licenseAgreement'}
                                components={{
                                    anchorlink: (
                                        <a
                                            target="_blank"
                                            onClick={() =>
                                                clientAPI.openLink(
                                                    modelDetails.model_license
                                                )
                                            }
                                        ></a>
                                    ),
                                }}
                            />
                        </Typography>
                    </Stack>
                </CustomModalDialog>
                <CustomModalDialog
                    dialogHeader={t('installModel', {
                        modelName: modelDetails.name,
                    })}
                    openDialog={openModelInstallDialog}
                    primaryButtonText={t('CONTINUE')}
                    secondaryButtonText={t('CANCEL')}
                    onPrimaryButtonClick={() => {
                        setOpenModelInstallDialog(false), onModelInstallClick()
                    }}
                    onSecondaryButtonClick={() =>
                        setOpenModelInstallDialog(false)
                    }
                >
                    <Stack
                        flexDirection={'column'}
                        alignItems={'stretch'}
                        justifyContent={'flex-start'}
                    >
                        <Typography variant="body2">
                            {t('installModelDescription', {
                                modelName: activeModelName,
                            })}
                        </Typography>
                    </Stack>
                </CustomModalDialog>
                <CustomModalDialog
                    dialogHeader={t('unInstallModel', {
                        modelName: modelDetails.name,
                    })}
                    openDialog={openModelUninstallDialog}
                    primaryButtonText={t('UNINSTALL')}
                    secondaryButtonText={t('CANCEL')}
                    onPrimaryButtonClick={(event) => {
                        setOpenModelUninstallDialog(false), onDelete(event)
                    }}
                    onSecondaryButtonClick={() =>
                        setOpenModelUninstallDialog(false)
                    }
                >
                    <Stack
                        flexDirection={'column'}
                        alignItems={'stretch'}
                        justifyContent={'flex-start'}
                    >
                        <Typography variant="body2">
                            {t('unInstallModelDescription', {
                                modelName: modelDetails.name,
                            })}
                        </Typography>
                    </Stack>
                </CustomModalDialog>
            </Stack>
        </div>
    )
}
