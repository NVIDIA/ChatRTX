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

import './custom-drawer.scss'
import React from 'react'
import { Stack, Typography } from '@mui/material'
import { useTranslation } from 'react-i18next'
import CustomIconButton from '../custom-icon-button/custom-icon-button'

interface CustomDrawerProps {
    drawerHeader: string
    openDrawer: boolean
    onDrawerClosed: () => void
}

export default function CustomDrawer({
    drawerHeader,
    openDrawer,
    onDrawerClosed,
    children,
}: React.PropsWithChildren<CustomDrawerProps>) {
    const { t } = useTranslation()

    return (
        <div className="drawer-wrapper" data-open={openDrawer}>
            <Stack
                className="drawer-container"
                direction={'column'}
                alignItems={'stretch'}
                justifyContent={'start'}
            >
                <Stack
                    className="drawer-header"
                    direction={'row'}
                    alignItems={'center'}
                    justifyContent={'space-between'}
                >
                    <Typography
                        variant="subtitle1"
                        className="drawer-header-text"
                    >
                        {drawerHeader}
                    </Typography>
                    <div className="close-icon-button">
                        <CustomIconButton
                            type="drawer-close"
                            tooltip={t('close')}
                            onClick={onDrawerClosed}
                        ></CustomIconButton>
                    </div>
                </Stack>
                <div className="drawer-content-container">{children}</div>
            </Stack>
        </div>
    )
}
