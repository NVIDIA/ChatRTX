'use strict'
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


const { join } = require('path')
const { v4: uuidv4 } = require('uuid')

try {
    // Native module, Can't be packaged into asar
    var {
        python,
        PyClass,
    } = require('../../../../app.asar.unpacked/node_modules/pythonia')
} catch {
    var { python, PyClass } = require('pythonia')
}

const { NV_RAG_ENGINE } = require('../config')
const { NV_RAG_ENGINE_PACK } = require('../config-packed')
const { Subject, map, filter } = require('rxjs')

async function initializeChatbot(isPackaged) {
    const pyfile = join(isPackaged ? NV_RAG_ENGINE_PACK : NV_RAG_ENGINE)
    console.log(pyfile)
    let chatbot = undefined
    try {
        chatbot = await python(pyfile)
        console.log('Loaded python file', pyfile)
    } catch (error) {
        console.log(`Error while loading ${pyfile}`, error)
    }

    const session_id = uuidv4()
    const eventSubject$ = new Subject()

    const emitter = async (channel, data) => {
        const ch = await channel.toString()
        const dt = await data.toString()
        eventSubject$.next({
            channel: ch,
            data: dt,
        })
        return true
    }

    /**
     * @returns events {
     *      channel: string,
     *      data: any
     * }
     */
    const listenPythonEvents = (eventList) => {
        return eventSubject$.asObservable().pipe(
            filter((value) => eventList.indexOf(value?.channel) > -1),
            map((value) => {
                console.log(
                    'Logging inside listner ',
                    value.channel,
                    value.data
                )
                return {
                    channel: value?.channel,
                    data: JSON.parse(value?.data),
                }
            })
        )
    }

    //Expose Python class in JavaScript by extending PyClass
    class Chatbot extends PyClass {
        constructor() {
            // [] -> Allows passing positional arguments to a Python constructor.
            super(chatbot.ChatBot, [session_id])
        }

        async setEmitter() {
            return await this.parent.set_emitter(emitter)
        }

        async ready(cb) {
            //this.parent works like super. in normal JS. You can use it to force call a parent to avoid recusion.
            cb(await this.parent.isready(session_id))
        }

        async initChatbotEngine() {
            return (await this.parent.init_chatbot_engine(session_id)).join()
        }

        async query(query, isStreaming, cb) {
            const obj = await this.parent.query(query, isStreaming, session_id)
            for await (const res of obj) {
                cb(res)
            }
            return
        }

        async setDatasetSource(source) {
            return (
                await this.parent.set_dataset_source(source, session_id)
            ).join()
        }

        async setDatasetPath(path) {
            return (await this.parent.set_dataset_path(path, session_id)).join()
        }

        async generateIndex() {
            return (await this.parent.generate_index(session_id)).join()
        }

        async setActiveModel(model) {
            return (
                await this.parent.set_active_model(model, session_id)
            ).join()
        }

        async reset() {
            return await this.parent.reset(session_id)
        }

        async shutdown() {
            return await this.parent.shutdown(session_id)
        }

        async intializeMic() {
            return await this.parent.intialize_mic(session_id)
        }

        async generateTextFromAudio(audioPath) {
            return await this.parent.generate_text_from_audio(
                audioPath,
                session_id
            )
        }

        async downloadModel(model_id) {
            return await this.parent.download_model(model_id, session_id)
        }

        async installModel(model_id) {
            return (
                await this.parent.install_model(model_id, session_id)
            ).join()
        }

        async deleteModel(model_id) {
            return await this.parent.delete_model(model_id, session_id)
        }

        async getConfig(key) {
            return await this.parent.get_config(key, session_id)
        }

        async setConfig(key, value) {
            return await this.parent.set_config(key, value, session_id)
        }

        async getDatasetInfo() {
            return await this.parent.get_dataset_info(session_id)
        }

        async getModelInfo() {
            return await this.parent.get_model_info(session_id)
        }

        async getFineTuningInfo() {
            return await this.parent.get_fine_tunning_info(session_id)
        }

        async createFineTuningModel(parentId, profileConfig) {
            return (
                await this.parent.create_fine_tuning_profile(
                    parentId,
                    profileConfig,
                    session_id
                )
            ).join()
        }

        async deleteFineTuningProfile(parentId, profileId) {
            return await this.parent.delete_fine_tuning_profile(
                parentId,
                profileId,
                session_id
            )
        }

        async downloadBaseModel(parentId) {
            return await this.parent.download_base_model(parentId, session_id)
        }

        async selectFineTuningModel(parentId, profileId) {
            return await this.select_fine_tuning_model(
                parentId,
                profileId,
                session_id
            )
        }

        async toggeFineTuningEnablement(parentId, shouldEnable) {
            return await this.toggle_fine_tuning_model(
                parentId,
                shouldEnable,
                session_id
            )
        }

        async initAsrModel() {
            return await this.init_asr_model(session_id)
        }

        async getTextFromAudio(audioPath) {
            return await this.get_text_from_audio(audioPath, session_id)
        }

        async getSampleQuestionInfo() {
            return await this.get_sample_question_info(session_id)
        }
    }

    const chatbotObj = await Chatbot.init()
    await chatbotObj.setEmitter()

    return {
        ready: chatbotObj.ready,
        initChatbotEngine: chatbotObj.initChatbotEngine,
        query: chatbotObj.query,
        setActiveModel: chatbotObj.setActiveModel,
        setDatasetSource: chatbotObj.setDatasetSource,
        setDatasetPath: chatbotObj.setDatasetPath,
        generateIndex: chatbotObj.generateIndex,
        reset: chatbotObj.reset,
        shutdown: chatbotObj.shutdown,
        intializeMic: chatbotObj.intializeMic,
        generateTextFromAudio: chatbotObj.generateTextFromAudio,
        downloadModel: chatbotObj.downloadModel,
        installModel: chatbotObj.installModel,
        deleteModel: chatbotObj.deleteModel,
        getConfig: chatbotObj.getConfig,
        setConfig: chatbotObj.setConfig,
        listenPythonEvents: listenPythonEvents,
        getDatasetInfo: chatbotObj.getDatasetInfo,
        getModelInfo: chatbotObj.getModelInfo,
        getFineTuningInfo: chatbotObj.getFineTuningInfo,
        createFineTuningModel: chatbotObj.createFineTuningModel,
        deleteFineTuningProfile: chatbotObj.deleteFineTuningProfile,
        downloadBaseModel: chatbotObj.downloadBaseModel,
        selectFineTuningModel: chatbotObj.selectFineTuningModel,
        toggeFineTuningEnablement: chatbotObj.toggeFineTuningEnablement,
        initAsrModel: chatbotObj.initAsrModel,
        getTextFromAudio: chatbotObj.getTextFromAudio,
        getSampleQuestionInfo: chatbotObj.getSampleQuestionInfo,
    }
}

module.exports = {
    Chatbot: initializeChatbot,
}
