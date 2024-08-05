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

import fs from 'fs'
import path from 'path'
import resolve from '@rollup/plugin-node-resolve'
import typescript from '@rollup/plugin-typescript'
import terser from '@rollup/plugin-terser'
import commonjs from '@rollup/plugin-commonjs'
import builtins from 'rollup-plugin-node-builtins'
import json from '@rollup/plugin-json'
import external from 'rollup-plugin-peer-deps-external'
import postcss from 'rollup-plugin-postcss'
import babel from '@rollup/plugin-babel'
import autoprefixer from 'autoprefixer'
import replace from '@rollup/plugin-replace'
import image from '@rollup/plugin-image'

const production = process.env.NODE_ENV === 'production'
const sourceDir = path.join(__dirname, 'src', 'renderer')
const buildDir = path.join(__dirname, 'build')

const copy = (from, to) => ({
    name: 'copy-on-close',
    closeBundle() {
        ;(Array.isArray(from) ? from : [from]).forEach((file) =>
            fs.copyFileSync(
                path.resolve(file),
                path.extname(to)
                    ? path.resolve(to)
                    : path.join(to, path.basename(file))
            )
        )
    },
})

const libPlugins = [
    postcss({
        plugins: [autoprefixer()],
        sourceMap: false,
        minimize: true,
        inject: true,
        exec: true,
    }),
    external(),
    builtins(),
    resolve({ browser: true, dedupe: ['react', 'react-dom'] }),
    babel({ babelHelpers: 'bundled', presets: ['@babel/preset-react'] }),
    commonjs(),
    image(),
    typescript({
        tsconfig: 'tsconfig.json',
        sourceMap: false,
        inlineSources: false,
    }),
    json(),
    replace({
        'process.env.NODE_ENV': JSON.stringify(process.env.NODE_ENV),
        preventAssignment: false,
    }),
]

if (production) {
    libPlugins.push(
        terser({
            compress: {
                passes: 2,
                drop_console: true,
            },
        })
    )
}

libPlugins.push(
    copy(
        [
            path.join(sourceDir, 'index.html'),
            path.join(sourceDir, 'favicon.ico'),
            path.join(sourceDir, 'icon.icns'),
            path.join(sourceDir, 'assets', 'fonts', 'NVIDIASans_W_Rg.woff2'),
            path.join(sourceDir, 'assets', 'fonts', 'NVIDIASans_W_Lt.woff2'),
            path.join(sourceDir, 'assets', 'fonts', 'NVIDIASans_W_Md.woff2'),
            path.join(sourceDir, 'assets', 'fonts', 'NVIDIASans_W_Bd.woff2'),
        ],
        buildDir
    )
)

export default [
    {
        input: path.join(sourceDir, 'index.tsx'),
        output: [
            {
                file: path.join(buildDir, 'index.js'),
                format: 'iife',
                interop: 'compat',
                sourcemap: false,
                inlineDynamicImports: true,
                name: 'chat_with_rtx',
            },
        ],
        plugins: libPlugins,
        treeshake: true,
        onwarn(warning, warn) {
            if (warning.code === 'MODULE_LEVEL_DIRECTIVE') {
                return
            }
            warn(warning)
        },
    },
]
