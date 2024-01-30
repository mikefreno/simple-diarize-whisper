import whisperx
import gc
import os
from dotenv import load_dotenv
import argparse
import torch
import platform
import subprocess
import sys
from datetime import timedelta
import time

load_dotenv()

start_time = time.time()
parser = argparse.ArgumentParser(description="A script that transcribes audio")

accepted_languages = ['en', 'fr', 'de', 'es', 'it', 'ja', 'zh', 'nl', 'ul', 'pt', None]
accepted_models = ['tiny', 'base', 'small', 'small.en', 'medium', 'medium.en' 'large', 'large-v1', 'large-v2', 'large-v3', 'distil-medium.en', 'distil-small.en', 'distil-large-v2']

parser.add_argument("-a", "--audio", help="'-a AUDIO_FILE' to denote file for transcription", required=True)
parser.add_argument("-fc", "--force-cpu", action="store_true", help="-fc or --force-cpu to use cpu even if cuda is available")
parser.add_argument("-b", "--batch", type=int, default=16, help="-b N sets batch size, decrease if low on mem, defaults to 16, increase if you have spare mem available")
parser.add_argument("-n", "--number", type=int, default=None, help="-n or --number N if exact speaker count number is known")
parser.add_argument("-mn", "--min", type=int, default=None, help="--min N if speakers known in range, must use with --max flag")
parser.add_argument("-mx", "--max", type=int, default=None, help="--max N if speakers known in range, must use with --min flag")
parser.add_argument("-m", "--model", default="large-v3", help=f"-m or --model to use a model other than large-v3, accepted vals {accepted_models}")
parser.add_argument("-l", "--language", default=None, help=f"-l or --language to specify language to skip detection in file accepted vals {accepted_languages}")
parser.add_argument("-hf", "--huggingface", default=None, help="-hf or --huggingface to provide access token instead of from loading from .env")
parser.add_argument("-t", "--time", default=None, help="provides a start time to base time stamps on, accepts 24hr times in format 20:10")
parser.add_argument("-nt", "--notime", action="store_true", default=None, help="removes timestamps")
parser.add_argument("-lg", "--low-gpu", action="store_true", default=None, help="recovers gpu resources, use if you have low gpu resources")
parser.add_argument("-anl", "--aggressive-new-line", action="store_true", default=None, help="puts new lines(breaks) at each same speaker chunk instead of spaces")

args = parser.parse_args()

audio_file = args.audio
defined_min_speakers = args.min
defined_max_speakers = args.max
exact_speaker_count = args.number
forced_cpu = args.force_cpu
batch_size = args.batch
language = args.language
model = args.model
timeInput =args.time
noTime = args.notime
lowGPU = args.low_gpu
aggressive_new_line = args.aggressive_new_line

hours = None
minutes = None
if timeInput:
    split = timeInput.split(":")
    hours = split[0] 
    minutes = split[1]

if args.huggingface == None:
    access_token = os.getenv("diarize_token")
else:
    access_token = args.huggingface

if model not in accepted_models:
    sys.exit(f"Invalid model. Accepted values are {accepted_models}")

if language not in accepted_languages:
    sys.exit(f"Invalid language. Accepted values are {accepted_languages}")

device, compute_type = ("cuda", "float16") if torch.cuda.is_available() and not forced_cpu else ("cpu", "int8")

start_model_load = time.time()
print("------loading model------")
model = whisperx.load_model(model, device, compute_type=compute_type, language=language)
end_model_load = time.time()
print("Time taken for model loading: %.2f seconds" % (end_model_load- start_model_load))


start_audio_load = time.time()
print("------loading audio------")
audio = whisperx.load_audio(audio_file)
end_audio_load = time.time()
print("Time taken for audio loading: %.2f seconds" % (end_audio_load- start_audio_load))


start_transcribe_load = time.time()
print("------transcribing------")
if language == None:
    result = model.transcribe(audio, batch_size=batch_size)
else:
    result = model.transcribe(audio, batch_size=batch_size, language=language)

end_transcribe_load = time.time()

print("Time taken for transcribing: %.2f seconds" % (end_transcribe_load- start_transcribe_load))


if lowGPU != None:
    gc.collect()
    torch.cuda.empty_cache()
    del model

print("------aligning whisper------")
start_alignment_load = time.time()
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

end_alignment_load = time.time()
print("Time taken for alignment: %.2f seconds" % (end_alignment_load- start_alignment_load))


if lowGPU != None:
    gc.collect(); 
    torch.cuda.empty_cache(); 
    del model_a

start_diarize = time.time()
print("------diarizing------")
diarize_model = whisperx.DiarizationPipeline(use_auth_token=access_token, device=device)

if exact_speaker_count:
    diarize_segments = diarize_model(audio_file, num_speakers=exact_speaker_count)
elif defined_min_speakers and defined_max_speakers:
    diarize_segments = diarize_model(audio_file, min_speakers=defined_min_speakers, max_speakers=defined_max_speakers)
else:
    diarize_segments = diarize_model(audio_file)

end_diarize_time = time.time()
print("Time taken for diarization: %.2f seconds" % (end_diarize_time - start_diarize))
print("--------end diarize--------")


start_writing_time = time.time()
print("--------writing alignment--------")
result = whisperx.assign_word_speakers(diarize_segments, result)

offset_seconds = None
if hours is not None and minutes is not None:
    offset_seconds = timedelta(hours=int(hours), minutes=int(minutes)).total_seconds()

with open('diarized_output.txt', 'w') as file:
    if offset_seconds is not None:
        running_start = timedelta(seconds=int(result['segments'][0]['start']) + offset_seconds)
    else:
        running_start = timedelta(seconds=int(result['segments'][0]['start']))
    current_speaker = result["segments"][0]['speaker']
    running_string = ""
    last_end = running_start

    for segment in result["segments"]:
        if offset_seconds is not None:
            start = timedelta(seconds=int(segment['start']) + offset_seconds)
            end = timedelta(seconds=int(segment['end']) + offset_seconds)
        else:
            start = timedelta(seconds=int(segment['start']))
            end = timedelta(seconds=int(segment['end']))
        text = segment['text']
        
        if current_speaker != segment.get('speaker') and segment.get('speaker') != None:
            file.write(f'[{running_start}-{last_end}] {current_speaker} -> {running_string.strip()}\n')
            running_start = start
            running_string = text
            current_speaker = segment.get('speaker')
        else:
            running_string += ("\n" if aggressive_new_line else " ") + text.strip()
            last_end = end

    file.write(f'[{running_start}-{last_end}] {current_speaker} -> {running_string.strip()}\n')


with open('base_output.txt', 'w') as file:
    current_speaker = result["segments"][0]['speaker']
    for segment in result["segments"]:
        text = segment['text']
        if segment.get('speaker') != current_speaker and segment.get('speaker') != None:
            file.write("\n")
        file.write(f"{text.strip()}\n")
        if segment.get('speaker'):
            current_speaker = segment.get('speaker')

end_writing_time = time.time()
print("Time taken for alignment: %.2f seconds" % (end_writing_time- start_writing_time))
print("--------end print--------")
print("Total process time: %.2f seconds" % (end_writing_time- start_time))


if platform.system() == 'Windows':
    os.startfile('diarized_output.txt')
elif platform.system() == "Darwin":
    subprocess.call(('open', 'diarized_output.txt'))
elif platform.system() == "Linux":
    subprocess.call(('xdg-open', 'diarized_output.txt'))
