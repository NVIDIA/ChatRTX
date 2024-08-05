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

import { FineTuningProfileConfig, ModelId } from '../../../electron/types'
import {
    Button,
    CircularProgress,
    Radio,
    Stack,
    Typography,
    Menu,
    MenuItem,
} from '@mui/material'
import React, { useState } from 'react'
import IconElement from '../icon-element/icon-element'
import { themeSettings } from '../../theme/theme'
import FineTuningSettings from '../fine-tuning-settings/fine-tuning-settings'
import './fine-tuning-card.scss'
import { useTranslation } from 'react-i18next'

export default function FineTuningCard({
    modelId,
    profile,
    showRadio,
    isActive,
    isActivating,
    onSelect,
    isNewStyle,
    onCancel,
    onCreateNewProfile,
    onProfileDelete,
}: {
    modelId?: ModelId
    profile?: FineTuningProfileConfig
    showRadio?: boolean
    isActive?: boolean
    isActivating?: boolean
    onSelect?: (value: string) => void
    isNewStyle?: boolean
    onCancel?: () => void
    onCreateNewProfile?: (profileConfig: FineTuningProfileConfig) => void
    onProfileDelete?: (profileId: string) => void
}) {
    const [moreOptionsAnchorEl, setMoreOptionsAnchorEl] =
        React.useState<null | HTMLElement>(null)
    const [isCollapsed, setIsCollapsed] = useState(!isNewStyle)
    const openMoreOptions = Boolean(moreOptionsAnchorEl)
    const { t } = useTranslation()

    const handleMoreOptionsClick = (
        event: React.MouseEvent<HTMLButtonElement>
    ) => {
        setMoreOptionsAnchorEl(event.currentTarget)
    }
    const handleMoreOptionsClosed = () => {
        setMoreOptionsAnchorEl(null)
    }

    const onDelete = () => {
        onProfileDelete(profile.profileId)
        setMoreOptionsAnchorEl(null)
    }

    return (
        <Stack
            className={isCollapsed ? 'container-collapsed' : 'container'}
            direction={'row'}
            alignItems={'stretch'}
            justifyContent={'flex-start'}
        >
            {!isNewStyle && showRadio && (
                <Stack
                    className="radio-button"
                    alignItems={'flex-start'}
                    justifyContent={'flex-start'}
                >
                    {isActivating && (
                        <div className="loading-spinner">
                            <CircularProgress size={16}></CircularProgress>
                        </div>
                    )}
                    {!isActivating && (
                        <Radio
                            checked={isActive}
                            value={profile.profileId}
                            onChange={(e) => onSelect(e.target.value)}
                        ></Radio>
                    )}
                </Stack>
            )}
            <div
                className={
                    isNewStyle ? 'new-style card-content' : 'card-content'
                }
            >
                {isCollapsed && (
                    <div className="title-box">
                        <Typography variant="body1">{profile.name}</Typography>
                    </div>
                )}
                {!isCollapsed && (
                    <FineTuningSettings
                        modelId={modelId}
                        profile={profile}
                        isNewStyle={isNewStyle}
                        onCancel={onCancel}
                        onCreateNewProfile={onCreateNewProfile}
                    ></FineTuningSettings>
                )}
                {!isCollapsed && (
                    <div className={isNewStyle ? '' : 'empty-area'}></div>
                )}
            </div>
            {!isNewStyle && (
                <Stack
                    direction={'column'}
                    alignItems={'stretch'}
                    justifyContent={'space-between'}
                >
                    <Button
                        color="transparent"
                        className="grey-btn square-btn slim-btn light-grey-hover-btn"
                        onClick={() => setIsCollapsed((value) => !value)}
                    >
                        <IconElement
                            color="n050"
                            type="shapes-chevron-down"
                            size="ml"
                        />
                    </Button>
                    {!isCollapsed && (
                        <Stack
                            direction={'row'}
                            alignItems={'flex-end'}
                            justifyContent={'flex-end'}
                            spacing={themeSettings.space.ms}
                        >
                            <Button
                                color="transparent"
                                className="grey-btn square-btn slim-btn light-grey-hover-btn"
                                onClick={handleMoreOptionsClick}
                                id="more-options-button"
                                aria-controls={
                                    openMoreOptions
                                        ? 'more-options-menu'
                                        : undefined
                                }
                                aria-haspopup="true"
                                aria-expanded={
                                    openMoreOptions ? 'true' : undefined
                                }
                            >
                                <IconElement color="n050" type="menu-3dots" />
                            </Button>
                            <Menu
                                id="more-options-menu"
                                anchorEl={moreOptionsAnchorEl}
                                open={openMoreOptions}
                                onClose={handleMoreOptionsClosed}
                                MenuListProps={{
                                    'aria-labelledby': 'more-options-button',
                                }}
                            >
                                {!isActive && (
                                    <MenuItem onClick={onDelete}>
                                        {t('delete')}
                                    </MenuItem>
                                )}
                            </Menu>
                        </Stack>
                    )}
                </Stack>
            )}
        </Stack>
    )
}
