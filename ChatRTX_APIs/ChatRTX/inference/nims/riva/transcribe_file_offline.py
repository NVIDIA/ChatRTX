# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import grpc
from pathlib import Path
from typing import Optional, List, Dict

import riva.client
from riva.client import ASRService, Auth, RecognitionConfig


def get_audio_to_text(
    server: str,
    input_file: Path,
    language_code: str = "en-US",
    ssl_cert: Optional[str] = None,
    use_ssl: bool = False,
    metadata: Optional[Dict[str, str]] = None,
    max_alternatives: int = 1,
    profanity_filter: bool = False,
    automatic_punctuation: bool = False,
    no_verbatim_transcripts: bool = False,
    word_time_offsets: bool = False,
    speaker_diarization: bool = False,
    diarization_max_speakers: int = 0,
    boosted_lm_words: Optional[List[str]] = None,
    boosted_lm_score: float = 0.0,
    start_history: float = 0.0,
    start_threshold: float = 0.0,
    stop_history: float = 0.0,
    stop_history_eou: float = 0.0,
    stop_threshold: float = 0.0,
    stop_threshold_eou: float = 0.0,
    custom_configuration: Optional[Dict] = None,
) -> str:
    """
    Transcribe audio from a file using NVIDIA Riva ASR service.

    Args:
        server (str): Riva server address (e.g., 'localhost:50051').
        input_file (Path): Path to the audio file to transcribe.
        language_code (str): Language code (default: "en-US").
        ssl_cert (Optional[str]): Path to SSL certificate if SSL is used.
        use_ssl (bool): Whether to use SSL (default: False).
        metadata (Optional[Dict[str, str]]): Additional metadata for the Riva client.
        max_alternatives (int): Maximum number of transcription alternatives (default: 1).
        profanity_filter (bool): Enable profanity filtering (default: False).
        automatic_punctuation (bool): Enable automatic punctuation (default: False).
        no_verbatim_transcripts (bool): Disable verbatim transcripts (default: False).
        word_time_offsets (bool): Enable word time offsets (default: False).
        speaker_diarization (bool): Enable speaker diarization (default: False).
        diarization_max_speakers (int): Maximum number of speakers for diarization (default: 0).
        boosted_lm_words (Optional[List[str]]): List of words to boost in the language model.
        boosted_lm_score (float): Boost score for the language model words (default: 0.0).
        start_history (float): Start history parameter for endpointing (default: 0.0).
        start_threshold (float): Start threshold for endpointing (default: 0.0).
        stop_history (float): Stop history parameter for endpointing (default: 0.0).
        stop_history_eou (float): Stop history end-of-utterance parameter (default: 0.0).
        stop_threshold (float): Stop threshold for endpointing (default: 0.0).
        stop_threshold_eou (float): Stop threshold end-of-utterance parameter (default: 0.0).
        custom_configuration (Optional[Dict]): Any custom configuration parameters.

    Returns:
        str: The transcribed text.

    Raises:
        Exception: If the gRPC request fails or no transcription is available.
    """
    # Initialize authentication
    auth = Auth(ssl_cert, use_ssl, server, metadata)
    asr_service = ASRService(auth)

    # Configure recognition parameters
    config = RecognitionConfig(
        language_code=language_code,
        max_alternatives=max_alternatives,
        profanity_filter=profanity_filter,
        enable_automatic_punctuation=automatic_punctuation,
        verbatim_transcripts=not no_verbatim_transcripts,
        enable_word_time_offsets=word_time_offsets or speaker_diarization,
    )

    # Add word boosting if provided
    if boosted_lm_words:
        riva.client.add_word_boosting_to_config(config, boosted_lm_words, boosted_lm_score)

    # Add speaker diarization if enabled
    if speaker_diarization:
        riva.client.add_speaker_diarization_to_config(config, speaker_diarization, diarization_max_speakers)

    # Add endpointing parameters
    riva.client.add_endpoint_parameters_to_config(
        config,
        start_history,
        start_threshold,
        stop_history,
        stop_history_eou,
        stop_threshold,
        stop_threshold_eou,
    )

    # Add any custom configurations
    if custom_configuration:
        riva.client.add_custom_configuration_to_config(config, custom_configuration)

    # Read the audio file
    try:
        with input_file.open('rb') as fh:
            data = fh.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Perform offline recognition
    try:
        response = asr_service.offline_recognize(data, config)
    except grpc.RpcError as e:
        raise Exception(f"gRPC error: {e.details()}") from e

    # Extract the transcript from the response
    if response.results and response.results[0].alternatives:
        final_transcript = " ".join(
            alternative.transcript for result in response.results for alternative in result.alternatives
        )
        return final_transcript.strip()
    else:
        raise Exception("No transcription results received.")


# Example usage:
if __name__ == "__main__":
    from pathlib import Path

    transcript = get_audio_to_text(
        server="localhost:50051",
        input_file=Path("en-US_sample.wav"),
        language_code="en-US",
        use_ssl=False
    )
    print("Transcript:", transcript)
