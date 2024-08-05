from configuration import Configuration
from ChatRTX.chatrtx import ChatRTX
from ChatRTX.chatrtx_rag import ChatRTXRag
from ChatRTX.logger import ChatRTXLogger
import logging
from ChatRTX.model_manager.model_manager import ModelManager
import sys, os
from pathlib import Path
from enum import Enum
import time
import random
from ResponseUtility import getLocalLinksMarkdown, getImagesMarkdown
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from ChatRTX.inference.trtllm.whisper.trt_whisper import WhisperTRTLLM, decode_audio_file
from ChatRTX.inference.trtllm.whisper.whisper_utils import process_input_audio
import time
import ctypes

class Mode(Enum):
    RAG = "RAG"
    AI = "AI"

class Backend:
    root_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
    log_path = "%programdata%\\NVIDIA Corporation\\chatrtx\\logs\\chatrtx.txt"
    data_path = "%programdata%\\NVIDIA Corporation\\chatrtx"
    CLIP_MODEL = "clip_model"

    def __init__(self, model_setup_dir):
        self.config = Configuration()
        ChatRTXLogger(log_level=logging.INFO, log_file=self.config.expand_programdata_path(self.log_path))
        self._logger = ChatRTXLogger.get_logger()

        self.active_model = None
        self.chatrtx = None
        self.chatrtx_mode = None
        self.rag_engine = None
        self.model_setup_dir = model_setup_dir
        self.model_manager = ModelManager(self.model_setup_dir)
        self._logger.info(f"Model init for dir {model_setup_dir}")
        # set default data direcotry
        dataset_dir = self.config.get_config('dataset/selected_path')
        dataset_dir = os.path.join(self.root_path, dataset_dir)
        if not os.path.isabs(dataset_dir):
            dataset_dir = os.path.abspath(dataset_dir)
        else:
            dataset_dir = os.path.normpath(dataset_dir)

        self.current_data_dir = dataset_dir
        self.selected_ChatGLM= False
        self.whisper_model_loaded = False
        self.whisper_model = None
        self.enable_asr = False

    def init_model(self, model_id: str):

        if not self.model_manager.is_model_downloaded(model_id):
            # Download the model if it is not already downloaded
            self.model_manager.download_model(model_id)
        else:
            self._logger.info("Model already downloaded")

        if not self.model_manager.is_model_installed(model_id):
            # Install the model if it is not already installed
            self.model_manager.install_model(model_id)

        self.active_model = model_id

    def ChatRTX(self, chatrtx_mode, data_dir = None):
        self.chatrtx_mode = chatrtx_mode
        model_info = self.model_manager.get_model_info()
        if self.active_model == self.CLIP_MODEL:
            self.chatrtx = ChatRTX(model_info, self.model_setup_dir)
            status = self.chatrtx.init_clip_model(self.active_model)
            if status:
                status = self.chatrtx.generate_clip_engine(data_dir)
            return status
        if self.chatrtx_mode == Mode.AI:
            try:
                self._logger.info(f"ChatRTX in {self.chatrtx_mode} mode")
                self.chatrtx = ChatRTX(model_info, self.model_setup_dir)
                status = self.chatrtx.init_llm_model(self.active_model, add_special_tokens=True, use_py_session=True)
                return status
            except Exception as e:
                self._logger.error(f"Error during loading the model in Mode {Mode.AI} \n Exception: {e}")
                return False
        elif self.chatrtx_mode == Mode.RAG:
            try:
                self._logger.info(f"ChatRTX in {self.chatrtx_mode} mode with datadir {data_dir}")
                if data_dir == None:
                    raise  ValueError("Data dir should not be null when app is in RAG mode")
                self.chatrtx = ChatRTXRag(model_info, self.model_setup_dir)
                status = self.chatrtx.init_llamaIndex_llm(self.active_model)

                if status == False:
                    return status

                # Set the embedding model
                emebdding_model_path = os.path.join(self.config.expand_programdata_path(self.data_path), "models", "multilingual-e5-base")
                self.chatrtx.set_embedding_model(emebdding_model_path, 768)

                # Set the RAG settings
                self.chatrtx.set_rag_setting(chunk_size=512, chunk_overlap=200)

                # Generate a query engine for the specified data directory
                self.rag_engine = self.chatrtx.generate_query_engine(data_dir, streaming=True)
                self.current_data_dir = data_dir
                return True
            except Exception as e:
                logging.error(f"Error occurred: {str(e)}")
                return False
        else:
            self._logger.info(f"Unknow mode")
            return False

    def generate_index(self):
        """
        Generates the query engine index if the mode is RAG. Logs an error and returns False if the mode is incorrect.
        """
        if self.active_model == self.CLIP_MODEL:
            try:
                status = self.chatrtx.generate_clip_engine(self.current_data_dir)
                return status
            except Exception as e:
                self._logger.error(f"Error during index generation for clip model: {e}")
                return False
        else:
            try:
                self._logger.debug(f"Generate the index with data path {self.current_data_dir}")
                if self.chatrtx_mode == Mode.RAG:
                    self.rag_engine = self.chatrtx.generate_query_engine(
                        self.current_data_dir, streaming=True, force_rewrite=True)
                    return True
                else:
                    self._logger.error("Wrong mode selected. Mode should be Mode.RAG")
                    return False
            except Exception as e:
                self._logger.error(f"Error during index generation for LLM: {e}")
                return False

    def generate_query_engine(self ,data_dir):
        if self.active_model == self.CLIP_MODEL:
            try:
                status = self.chatrtx.generate_clip_engine(data_dir)
            except Exception as e:
                self._logger.error(f"Error in generating engine for path {data_dir} for CLIP model. \n Exception {e}")
                return False
            return status

        if self.chatrtx_mode == Mode.AI:
            raise ValueError(f"ChatRTX Mode must be set to RAG")
        try:
            self.rag_engine = self.chatrtx.generate_query_engine(data_dir, streaming=True)
            self.current_data_dir = data_dir
            return True
        except Exception as e:
            self._logger.error(f"Error in generating engine for path {data_dir}. \n Exception {e}")
            return False

    def query(self, query):
        if self.chatrtx_mode == Mode.AI:
            response = self.chatrtx.generate_response(query=query)
            return  response

    def query_stream(self, query):

        if self.active_model == self.CLIP_MODEL:
            min_clip_score = 20 # clip threshold
            matched_ouput =os.path.join(self.config.expand_programdata_path(self.data_path), "All_matched_images")
            os.makedirs(matched_ouput, exist_ok=True)
            if os.path.islink(matched_ouput) or self.is_junction(matched_ouput):
                yield "Invalid image match directory. "
            
            self.clean_directory(matched_ouput)
            answer  = self.chatrtx.generate_clip_response(input_text=query, top_matches_path=matched_ouput,
                                            min_clip_score=min_clip_score)

            top_images = answer[:3]

            yield getImagesMarkdown(top_images)
            yield getLocalLinksMarkdown([matched_ouput])

        else:
            self._logger.debug(f"Generate the response for query: {query}")
            if self.chatrtx_mode == Mode.AI:
                response = self.chatrtx.generate_stream_response(query=query)
                for token in response:
                    yield str(token)
            else:

                if self.rag_engine is not None:
                    response = self.chatrtx.generate_stream_response(query=query, query_engine=self.rag_engine)

                if len(response.source_nodes) > 0:
                    for token in response.response_gen:
                        yield str(token)

                lowest_score_file = None
                lowest_score = sys.float_info.max

                if len(response.source_nodes) > 0:
                    for node in response.source_nodes:
                        if 'filename' in node.metadata:
                            if node.score < lowest_score:
                                lowest_score = node.score
                                lowest_score_file = node.metadata['filename']

                    file_links = []
                    seen_files = set()

                    if lowest_score_file:
                        abs_path = Path(os.path.join(os.getcwd(), lowest_score_file.replace('\\', '/')))
                        file_name = os.path.basename(abs_path)
                        if file_name not in seen_files:  # Check if file_name is already seen
                            if self.chatrtx_mode == Mode.RAG:
                                file_link = abs_path
                            else:
                                exit("Wrong data_source type")
                            file_links.append(file_link)
                            seen_files.add(file_name)  # Add file_name to the set

                    partial_response = ""
                    if file_links:

                        partial_response += "<br>Reference files:<br>" + getLocalLinksMarkdown(file_links)
                    yield partial_response

                elif len(response.source_nodes) == 0:
                    yield "Problem generating response: Data source may be empty or unsupported – Ensure dataset compatibility with the AI model. Alternatively, try ‘Chat with AI model data’."

                else:
                    raise ValueError(f"Invalid Node values")

    def set_chatrtx_mode(self, chat_mode: Mode):
        self.chatrtx.unload_llm()

        if chat_mode == Mode.AI:
            status = self.ChatRTX(chat_mode)
        else:
            # get the old value written in the config fileif app value is none
            if self.current_data_dir == None:
                dataset_dir = self.config.get_config('dataset/selected_path')
                dataset_dir = os.path.join(self.root_path, dataset_dir)
                if not os.path.isabs(dataset_dir):
                    dataset_dir = os.path.abspath(dataset_dir)
                else:
                    dataset_dir = os.path.normpath(dataset_dir)

                self.current_data_dir = dataset_dir
            status = self.ChatRTX(chat_mode, self.current_data_dir)

        if status == False:
            self._logger.error(f"Error in switching the Chart MODe to {chat_mode}")
            return False

        self.chatrtx_mode = chat_mode

        if self.chatrtx_mode == Mode.AI:
            source = "nodataset"
        elif self.chatrtx_mode == Mode.RAG:
            source = "directory"
        else:
            raise ValueError(f"Invalid mode: {self.chatrtx_mode}. Mode must be either 'AI' or 'RAG'.")

        if status:
            status = self.model_manager.update_dataset(source)

        return status

    def set_dataset_path(self, path):
        self._logger.info(f"set the data_dir to {path}")
        status = self.generate_query_engine(path)
        if status == False:
            return False
        self.current_data_dir = path
        status = self.model_manager.update_data_directory_path(self.current_data_dir)
        return status

    def download_model(self, model_id):
        status = False
        if not self.model_manager.is_model_downloaded(model_id):
            # Download the model if it is not already downloaded
            status = self.model_manager.download_model(model_id)
        else:
            status = True
        return status

    def install_model(self, model_id):
        status = False

        self.chatrtx.unload_llm()
        if not self.model_manager.is_model_installed(model_id):
            self._logger.info(f"Building TRT-LLM engine for model: {model_id}....")
            status = self.model_manager.install_model(model_id)
        else:
            self._logger.info(f"TRT-LLM engine for model: {model_id} is already present")
            status = True

        if status == True:
            # load the new model in required mode(RAG or AI)
            self.active_model = model_id
            self.chatrtx_mode = Mode.RAG # always launch the new model in RAG mode
            source = "nodataset" if self.chatrtx_mode == Mode.AI else "directory" if self.chatrtx_mode == Mode.RAG else None
            status = self.model_manager.update_dataset(source)

            if self.chatrtx_mode == Mode.AI:
                self.ChatRTX(self.chatrtx_mode)

            elif self.chatrtx_mode == Mode.RAG:
                self.ChatRTX(self.chatrtx_mode, data_dir=self.current_data_dir)
            else:
                raise ValueError(f"Invalid mode: {self.chatrtx_mode}. Mode must be either 'AI' or 'RAG'.")

        if status == True:
            if model_id == "mistral_7b_AWQ_int4_chat" or model_id == "llama2_13b_AWQ_INT4_chat" or model_id == "gemma_7b_int4":
                dataset = self.config.get_config('dataset/path')
            elif model_id == "chatglm3_6b_AWQ_int4":
                dataset = self.config.get_config('dataset/path_chinese')
            elif model_id == "clip_model":
                dataset = self.config.get_config('dataset/path_clip')

            dataset = self.config.expand_programdata_path(dataset)
            status = self.set_dataset_path(dataset)

        return status

    def delete_model(self, model_id):
        status = self.model_manager.delete_model(model_id)
        self._logger.info(f"Delete model return {status}")
        return status

    def set_active_model(self, model_id):
        try:
            self.chatrtx.unload_llm()
            # if model is clip and mode is Mode.AI than we need to make sure that we change the mode to RAG before loading the clip model as clip does not suppoty AI mode
            """
            if model_id == self.CLIP_MODEL and self.chatrtx_mode == Mode.AI:
                status  = self.set_chatrtx_mode(Mode.RAG)
                if status:
                    status = self.model_manager.update_active_model(model_id)
                    return status
            """
            # update the active model before calling ChatRTX function to load new model
            self.active_model = model_id
            status = True
            if model_id == self.CLIP_MODEL and self.chatrtx_mode == Mode.AI:
                self.chatrtx_mode = Mode.RAG
                source = "nodataset" if self.chatrtx_mode == Mode.AI else "directory" if self.chatrtx_mode == Mode.RAG else None
                status = self.model_manager.update_dataset(source)

            if status == False:
                return False

            # load the new model in required mode(RAG or AI)
            if self.chatrtx_mode == Mode.AI:
                status = self.ChatRTX(self.chatrtx_mode)
            elif self.chatrtx_mode == Mode.RAG:
                status = self.ChatRTX(self.chatrtx_mode, data_dir=self.current_data_dir)
            else:
                raise ValueError(f"Invalid mode: {self.chatrtx_mode}. Mode must be either 'AI' or 'RAG'.")

            if status:
                status = self.model_manager.update_active_model(model_id)


            return status
        except Exception as e:
            self._logger.error(f"Error in setting the active model. Exception {e}")
            return False

    
    def init_asr_model(self):
        # initialize transcription model
        asr_engine_path = os.path.join(self.model_setup_dir, 'models', 'whisper', 'whisper_medium_int8_engine')
        asr_assets_path = os.path.join(self.model_setup_dir, 'models', 'whisper', 'whisper_assets')
        self.enable_asr = True
        if not self.enable_asr:
            return False
        vid_mem_info = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(0))
        free_vid_mem = vid_mem_info.free / (1024*1024)
        print("free video memory in MB = ", free_vid_mem)
        if self.whisper_model is not None:
            self.whisper_model.unload_model()
            del self.whisper_model
            self.whisper_model = None
        self.whisper_model = WhisperTRTLLM(asr_engine_path, assets_dir=asr_assets_path)
        self.whisper_model_loaded = True
        self._logger.info(f"init asr backend done")
        return True
    
    def get_text_from_audio(self, audio_path):
        # Return transcribed text via this function
        asr_assets_path = os.path.join(self.model_setup_dir, 'models','whisper', 'whisper_assets')
        transcription = ""
        global whisper_model
        if not self.enable_asr:
            return ""
        
        # Check and wait until model is loaded before running it.
        checks_for_model_loading = 40
        checks_left_for_model_loading = checks_for_model_loading
        sleep_time = 0.2
        while checks_left_for_model_loading>0 and not self.whisper_model_loaded:
            time.sleep(sleep_time)
            checks_left_for_model_loading -= 1
        assert checks_left_for_model_loading>0, f"Whisper model loading not finished even after {(checks_for_model_loading*sleep_time)} seconds"
        if checks_left_for_model_loading == 0:
            return ""

        new_file_path = process_input_audio(audio_path)
        language = "english"
        self._logger.info(f"model selected {self.active_model}")
        if self.active_model == "chatglm3_6b_AWQ_int4":
            self._logger.info(f"chinese model selected")
            language = "chinese"
        transcription = decode_audio_file( new_file_path, self.whisper_model, language=language, mel_filters_dir=asr_assets_path)

        if self.whisper_model is not None:        
            self.whisper_model.unload_model()
            del self.whisper_model
            self.whisper_model = None
        self.whisper_model_loaded = False
        self._logger.info(f"asr backend transcription done")
        return transcription

    def _rand_handle(self):
        # 1/3rd mocked as success
        time.sleep(5)
        return True if random.randint(1, 12) % 4 != 0 else False
        
    def is_junction(self, path):
        FILE_ATTRIBUTE_REPARSE_POINT = 0x0400
        INVALID_FILE_ATTRIBUTES = -1
        
        attrs = ctypes.windll.kernel32.GetFileAttributesW(path)
        if attrs == INVALID_FILE_ATTRIBUTES:
            raise OSError("Failed to get file attributes for path: " + path)
    
        return bool(attrs & FILE_ATTRIBUTE_REPARSE_POINT)
        
    def clean_directory(self, directory):
        # List all files and subdirectories in the specified directory
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                # Check if it is a file and remove it
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                # Check if it is a directory and remove it
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
