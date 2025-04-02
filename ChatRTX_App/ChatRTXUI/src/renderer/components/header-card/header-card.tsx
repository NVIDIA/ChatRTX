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

import './header-card.scss'
import React from 'react'
import { Typography, Stack } from '@mui/material'
import { useTranslation } from 'react-i18next'
import { themeSettings } from '../../theme/theme'
import CustomIconButton from '../custom-icon-button/custom-icon-button'
import OverflowTypography from '../overflow-typography/overflow-typography'

export default function HeaderCard({
    isCardActive,
    label,
    subLabel,
    isDrawerOpen,
    onHeaderCardClick,
    onResyncClick,
}: {
    isCardActive: boolean
    label: string
    subLabel: string
    isDrawerOpen: boolean
    onHeaderCardClick: () => void
    onResyncClick?: () => void
}) {
    const { t } = useTranslation()

    return (
        <div
            data-isdraweropen={isDrawerOpen}
            className="header-card-button card-button"
            onClick={onHeaderCardClick}
        >
            <Stack
                direction={'row'}
                alignItems={'center'}
                justifyContent={'flex-start'}
                className="header-card"
                data-isactive={isCardActive}
                spacing={themeSettings.font.sizes.sm}
            >
                <div className="fill-active"></div>
                <Stack
                    direction={'column'}
                    alignItems={'flex-start'}
                    justifyContent={'center'}
                    className="header-label-container"
                >
                    <Typography variant="body1" className="header-label">
                        {label}
                    </Typography>
                    <Stack
                        direction={'row'}
                        alignItems={'center'}
                        justifyContent={'flex-start'}
                        className="header-sub-label-container"
                    >
                        <OverflowTypography
                            className="header-sub-label"
                            variant="body2"
                        >
                            {subLabel}
                        </OverflowTypography>
                    </Stack>
                </Stack>
                {onResyncClick && (
                    <CustomIconButton
                        type="common-refresh"
                        tooltip={t('refreshDataset')}
                        onClick={(event) => {
                            onResyncClick()
                            event.stopPropagation()
                        }}
                    ></CustomIconButton>
                )}
            </Stack>
        </div>
    )
}
