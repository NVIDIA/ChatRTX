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

import i18n from 'i18next'
import { initReactI18next, Translation } from 'react-i18next'
import en_US_Json from './locales/en_US.json'
import zh_CN_Json from './locales/zh_CN.json'
import zh_TW_Json from './locales/zh_TW.json'

i18n.use(initReactI18next).init({
    fallbackLng: 'en-US',
    lng: navigator.language,
    resources: {
        'en-US': {
            translations: en_US_Json,
        },
        'zh-CN': {
            translations: zh_CN_Json,
        },
        'zh-TW': {
            translations: zh_TW_Json,
        },
    },
    ns: ['translations'],
    defaultNS: 'translations',
})

export default i18n
