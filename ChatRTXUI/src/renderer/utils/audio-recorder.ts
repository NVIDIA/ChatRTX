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

import { ClientAPI } from '../../electron/preload'
import ID from '../utils/id'

type AudioRecorderEventCallback = (evt: string, value: any) => void

const MAX_RECORDING = 30
const clientAPI = (window as any).clientAPI as typeof ClientAPI
const listeners: { [key: string]: AudioRecorderEventCallback } = {}

const convertToWav = (buffer: ArrayBuffer): Promise<Blob> => {
    const sampleRate = 16000
    const numberOfChannels = 1
    const bitDepth = 16
    const bytesPerSample = bitDepth / 8

    const getPCM = (audioData: AudioBuffer) => {
        const pcmData: Float32Array[] = []

        for (let i = 0; i < numberOfChannels; i++) {
            pcmData.push(audioData.getChannelData(i))
        }

        const length = pcmData[0].length
        const result = new Uint8Array(
            length * numberOfChannels * bytesPerSample
        )

        for (let i = 0; i < length; i++) {
            for (let channel = 0; channel < numberOfChannels; channel++) {
                const index = (i * numberOfChannels + channel) * bytesPerSample
                let sample = pcmData[channel][i]

                if (sample > 1) {
                    sample = 1
                } else if (sample < -1) {
                    sample = -1
                }

                switch (bytesPerSample) {
                    case 4:
                        sample = sample * 2147483648
                        result[index] = sample
                        result[index + 1] = sample >> 8
                        result[index + 2] = sample >> 16
                        result[index + 3] = sample >> 24
                        break

                    case 3:
                        sample = sample * 8388608
                        result[index] = sample
                        result[index + 1] = sample >> 8
                        result[index + 2] = sample >> 16
                        break

                    case 2:
                        sample = sample * 32768
                        result[index] = sample
                        result[index + 1] = sample >> 8
                        break

                    case 1:
                        result[index] = (sample + 1) * 128
                        break

                    default:
                        throw 'Bits must be divisible by 8'
                }
            }
        }

        return result
    }

    const getWavBuffer = (pcm: Uint8Array) => {
        const header = 44
        const size = header + pcm.length
        const result = new Uint8Array(size)
        const data = new DataView(result.buffer)
        data.setUint32(0, 1380533830, false)
        data.setUint32(4, 36 + pcm.length, true)
        data.setUint32(8, 1463899717, false)
        data.setUint32(12, 1718449184, false)
        data.setUint32(16, 16, true)
        data.setUint16(20, 1, true)
        data.setUint16(22, numberOfChannels, true)
        data.setUint32(24, sampleRate, true)
        data.setUint32(28, sampleRate * bytesPerSample * numberOfChannels, true)
        data.setUint16(32, bytesPerSample * numberOfChannels, true)
        data.setUint16(34, bitDepth, true)
        data.setUint32(36, 1684108385, false)
        data.setUint32(40, pcm.length, true)
        result.set(pcm, header)
        return result.buffer
    }

    return new AudioContext({ sampleRate })
        .decodeAudioData(buffer)
        .then((audioData) => getPCM(audioData))
        .then((pcm) => getWavBuffer(pcm))
        .then((audioData) => new Blob([audioData], { type: 'audio/wav' }))
}

const AudioRecorder = {
    _isRecording: false,
    get isRecording() {
        return AudioRecorder._isRecording
    },
    set isRecording(value: boolean) {
        if (value === AudioRecorder._isRecording) {
            return
        }
        AudioRecorder._isRecording = value
        AudioRecorder.dispatchEvent('recording', AudioRecorder.isRecording)
    },

    _isProcessing: false,
    get isProcessing() {
        return AudioRecorder._isProcessing
    },
    set isProcessing(value: boolean) {
        if (value === AudioRecorder._isProcessing) {
            return
        }
        AudioRecorder._isProcessing = value
        AudioRecorder.dispatchEvent('processing', AudioRecorder.isProcessing)
    },

    _recordingTime: 0,
    get recordingTime() {
        return AudioRecorder._recordingTime
    },
    set recordingTime(value: number) {
        if (value === AudioRecorder._recordingTime) {
            return
        }
        AudioRecorder._recordingTime = value
        AudioRecorder.dispatchEvent('time', AudioRecorder.recordingTime)
    },

    _devices: [] as MediaDeviceInfo[],
    get devices() {
        return AudioRecorder._devices.slice()
    },
    set devices(value: MediaDeviceInfo[]) {
        AudioRecorder._devices = value
        AudioRecorder.dispatchEvent('devices', AudioRecorder._devices.slice())
    },

    _deviceId: '',
    get deviceId() {
        return AudioRecorder._deviceId
    },
    set deviceId(value: string) {
        if (value === AudioRecorder._deviceId) {
            return
        }
        AudioRecorder._deviceId = value
        AudioRecorder.dispatchEvent('deviceId', AudioRecorder._deviceId)
    },

    _canRecord: false,
    get canRecord() {
        return AudioRecorder._canRecord
    },
    set canRecord(value: boolean) {
        if (value === AudioRecorder._canRecord) {
            return
        }
        AudioRecorder._canRecord = value
        AudioRecorder.dispatchEvent('canRecord', AudioRecorder._canRecord)
    },

    _isInitialized: false,
    get isInitialized() {
        return AudioRecorder._isInitialized
    },
    set isInitialized(value: boolean) {
        if (value === AudioRecorder._isInitialized) {
            return
        }
        AudioRecorder._isInitialized = value
        AudioRecorder.dispatchEvent(
            'isInitialized',
            AudioRecorder._isInitialized
        )
    },

    recordingTimer: null as any,

    recordingId: '',

    streamId: '',

    id: '',

    stream: null as MediaStream,

    recorder: null as MediaRecorder,

    chunks: [] as Blob[],

    dispose: () => {
        AudioRecorder.isRecording = false
        AudioRecorder.recordingTime = 0
        AudioRecorder.isProcessing = false
        AudioRecorder.recordingId = ''
        AudioRecorder.streamId = ''
        AudioRecorder.closeRecorder(AudioRecorder.recorder)
        AudioRecorder.closeStream(AudioRecorder.stream)
    },

    setDevices: () =>
        new Promise((resolve, reject) => {
            if (
                !navigator.mediaDevices ||
                !navigator.mediaDevices.getUserMedia
            ) {
                return reject('Media capture is not available')
            }

            const id = ID()
            AudioRecorder.id = id

            navigator.mediaDevices
                .enumerateDevices()
                .then((devices) => {
                    if (id !== AudioRecorder.id) {
                        return resolve(undefined)
                    }
                    AudioRecorder.devices = devices.reduce(
                        (result, device) =>
                            device.kind === 'audioinput'
                                ? result.concat([device])
                                : result,
                        [] as MediaDeviceInfo[]
                    )
                    AudioRecorder.canRecord = AudioRecorder.devices.length > 0
                    AudioRecorder.deviceId = (
                        AudioRecorder.devices.find(
                            (device) =>
                                device.label.toLowerCase().indexOf('default') >
                                -1
                        ) ||
                        AudioRecorder.devices[0] || { deviceId: '' }
                    ).deviceId
                    resolve(undefined)
                })
                .catch(reject)
        }),

    listenForEvents: (callback: AudioRecorderEventCallback) => {
        const id = ID()

        listeners[id] = callback

        const updateDevices = () => {
            const deviceId = AudioRecorder.deviceId

            AudioRecorder.setDevices()
                .then(() => {
                    if (deviceId !== AudioRecorder.deviceId) {
                        if (AudioRecorder.isRecording) {
                            AudioRecorder.forceStop()
                        }

                        if (AudioRecorder.stream) {
                            AudioRecorder.closeStream(AudioRecorder.stream)
                        }
                    }

                    if (
                        AudioRecorder.isInitialized &&
                        !!AudioRecorder.deviceId
                    ) {
                        AudioRecorder.startStream()
                    }
                })
                .catch((err) => {
                    console.warn(err)
                    AudioRecorder.dispatchError(
                        'Could not get microphone inputs'
                    )
                })
        }

        updateDevices()

        if (navigator.mediaDevices) {
            navigator.mediaDevices.addEventListener(
                'devicechange',
                updateDevices
            )
        }

        return () => {
            listeners[id] = null
            delete listeners[id]

            if (navigator.mediaDevices) {
                navigator.mediaDevices.removeEventListener(
                    'devicechange',
                    updateDevices
                )
            }

            AudioRecorder.dispose()
        }
    },

    dispatchEvent: (evt: string, value: any) =>
        Object.keys(listeners).forEach((k) => {
            const callback = listeners[k]
            if (typeof callback === 'function') {
                callback(evt, value)
            }
        }),

    dispatchError: (error: Error | string) => {
        const value =
            typeof error === 'string'
                ? error
                : !!error
                  ? error.message || error.toString()
                  : ''
        AudioRecorder.dispatchEvent('error', value)
    },

    handleRecording: (id: string) => {
        if (AudioRecorder.recordingId !== id) {
            return
        }

        const blob = new Blob(AudioRecorder.chunks, {
            type: 'audio/webm;codecs=opus',
        })
        AudioRecorder.chunks = []
        AudioRecorder.recordingTime = 0
        AudioRecorder.isRecording = false

        if (!blob.size) {
            AudioRecorder.restartStream()
            return AudioRecorder.dispatchError(
                'No audio received from microphone, please try again'
            )
        }

        return blob
            .arrayBuffer()
            .then((buffer) => convertToWav(buffer))
            .then((wavBlob) => {
                const reader = new FileReader()
                reader.onload = () => {
                    if (AudioRecorder.recordingId !== id) {
                        return
                    }
                    AudioRecorder.sendData(reader.result as string, id)
                }
                reader.readAsDataURL(wavBlob)
            })
            .catch((err) => {
                console.warn(err)
                AudioRecorder.dispatchError(
                    'Could not save microphone recording, please try again'
                )
            })
    },

    toggleRecording: () => {
        clearInterval(AudioRecorder.recordingTimer)

        AudioRecorder.recordingTime = 0

        if (!AudioRecorder.deviceId) {
            if (AudioRecorder.isRecording) {
                AudioRecorder.stopCapture()
            }

            AudioRecorder.isRecording = false
            return
        }

        AudioRecorder.isRecording = !AudioRecorder.isRecording

        if (!AudioRecorder.isRecording) {
            AudioRecorder.stopCapture()
        } else {
            AudioRecorder.startCapture()
        }
    },

    restartStream: () => {
        AudioRecorder.isProcessing = false
        AudioRecorder.closeRecorder(AudioRecorder.recorder)
        AudioRecorder.recorder = null
        AudioRecorder.closeStream(AudioRecorder.stream)
        AudioRecorder.setDevices()
            .then(() => AudioRecorder.startStream())
            .catch(console.warn)
    },

    startStream: async () => {
        if (!AudioRecorder.deviceId) {
            return AudioRecorder.dispatchError('Microphone input not selected')
        }

        const id = ID()
        AudioRecorder.streamId = id

        const constraints: MediaStreamConstraints = {
            audio: {
                deviceId: AudioRecorder.deviceId,
                channelCount: 1,
                sampleRate: 16000,
            },
        }

        navigator.mediaDevices
            .getUserMedia(constraints)
            .then((stream) => {
                if (id !== AudioRecorder.streamId) {
                    AudioRecorder.closeStream(stream)
                    return
                }

                AudioRecorder.stream = stream
            })
            .catch((err) => {
                AudioRecorder.streamId = ''
                AudioRecorder.dispatchError(`Get microphone: ${err}`)
            })
    },

    startCapture: () => {
        if (!AudioRecorder.stream) {
            return AudioRecorder.dispatchError('No microphone stream found')
        }

        if (AudioRecorder.recorder) {
            return AudioRecorder.dispatchError(
                'Microphone already recording, can not restart'
            )
        }

        const id = ID()
        AudioRecorder.recordingId = id

        AudioRecorder.recordingTimer = setInterval(() => {
            if (
                AudioRecorder.recordingId !== id ||
                !AudioRecorder.isRecording ||
                AudioRecorder.isProcessing
            ) {
                return
            }

            AudioRecorder.recordingTime = AudioRecorder.recordingTime + 1

            if (AudioRecorder.recordingTime >= MAX_RECORDING) {
                AudioRecorder.toggleRecording()
            }
        }, 1000)

        AudioRecorder.recorder = new MediaRecorder(AudioRecorder.stream, {
            mimeType: 'audio/webm',
        })

        AudioRecorder.recorder.onerror = () => {
            AudioRecorder.dispatchError(
                'Media recorder failed, please try again'
            )
            AudioRecorder.isProcessing = true
            AudioRecorder.closeRecorder(AudioRecorder.recorder)
            AudioRecorder.recorder = null
        }

        AudioRecorder.recorder.ondataavailable = (event) => {
            if (id !== AudioRecorder.recordingId) {
                return
            }
            AudioRecorder.chunks.push(event.data)
        }

        AudioRecorder.recorder.onstop = async () => {
            if (id !== AudioRecorder.recordingId) {
                return
            }
            AudioRecorder.handleRecording(id)
        }

        AudioRecorder.recorder.start()
    },

    forceStop: () => {
        clearInterval(AudioRecorder.recordingTimer)

        AudioRecorder.recordingTime = 0
        AudioRecorder.isRecording = false
        AudioRecorder.stopCapture()
    },

    stopCapture: () => {
        AudioRecorder.isProcessing = true
        AudioRecorder.closeRecorder(AudioRecorder.recorder)
        AudioRecorder.recorder = null
    },

    closeStream: (stream: MediaStream) => {
        if (!stream) {
            return
        }

        try {
            stream.getTracks().forEach((track) => {
                try {
                    track.stop()
                    stream.removeTrack(track)
                } catch (error) {
                    console.warn('error closing stream track', error)
                }
            })
        } catch (error) {
            console.warn('error closing stream', error)
        }

        AudioRecorder.stream = null
    },

    closeRecorder: (recorder: MediaRecorder) => {
        if (!recorder) {
            return
        }

        try {
            recorder.stop()
        } catch (error) {
            console.warn('error closing mediaRecorder', error)
        }

        AudioRecorder.recorder = null
    },

    sendData: (data: string, id: string) => {
        clientAPI
            .getTextFromAudio(data)
            .then((text: string) => {
                if (AudioRecorder.recordingId !== id) {
                    return
                }

                AudioRecorder.isProcessing = false
                AudioRecorder.dispatchEvent('asrResponse', text)
            })
            .catch(console.error)
    },
}

export default AudioRecorder
