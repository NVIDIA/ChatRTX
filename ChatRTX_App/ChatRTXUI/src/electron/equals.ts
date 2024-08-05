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

/* NOTE DOES NOT CHECK ArrayBuffers, DataViews, or RegExp */

export default function Equals(element1: any, element2: any) {
    if (
        element1 &&
        element2 &&
        typeof element1 === 'object' &&
        typeof element2 === 'object'
    ) {
        // check if same type
        if (element1.constructor !== element2.constructor) {
            return false
        }

        if (Array.isArray(element1)) {
            const length = element1.length
            let index = length
            if (length !== element2.length) {
                return false
            }
            while (index) {
                index = index - 1
                if (!Equals(element1[Number(index)], element2[Number(index)])) {
                    return false
                }
            }
            return true
        }

        if (element1 instanceof Map && element2 instanceof Map) {
            if (element1.size !== element2.size) {
                return false
            }

            let equals = true

            element1.forEach((value, key) => {
                if (!equals) {
                    return
                }
                if (!Equals(value, element2.get(key))) {
                    equals = false
                }
            })

            return equals
        }

        if (element1 instanceof Set && element2 instanceof Set) {
            return Equals(Array.from(element1), Array.from(element2))
        }

        if (element1.valueOf !== Object.prototype.valueOf) {
            return element1.valueOf() === element2.valueOf()
        }
        if (element1.toString !== Object.prototype.toString) {
            return element1.toString() === element2.toString()
        }

        const keys = Object.keys(element1)
        const length = keys.length
        if (length !== Object.keys(element2).length) {
            return false
        }

        let index = length

        while (index) {
            index = index - 1
            const key = keys[Number(index)]
            if (!Equals(element1[`${key}`], element2[`${key}`])) {
                return false
            }
        }

        return true
    }

    if (element1 === element2) {
        return true
    }

    if (Number.isNaN(element1) && Number.isNaN(element2)) {
        return true
    }

    return false
}
