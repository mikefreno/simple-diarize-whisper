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

load_dotenv()
parser = argparse.ArgumentParser(description="A script that transcribes audio")

parser.add_argument("-a", "--audio", help="'-a AUDIO_FILE' to denote file for transcription", required=True)
parser.add_argument("-fc", "--force-cpu", action="store_true", help="-fc or --force-cpu to use cpu even if cuda is available")
parser.add_argument("-b", "--batch", type=int, default=16, help="-b N sets batch size, decrease if low on mem, defaults to 16")
parser.add_argument("--min", type=int, default=None, help="--min N if speakers known in range, must use with --max flag")
parser.add_argument("--max", type=int, default=None, help="--max N if speakers known in range, must use with --min flag")
parser.add_argument("-m", "--model", default="large-v2", help="-m or --model to use a model other than large-v2, accepted vals")
parser.add_argument("-l", "--language", default=None, help="-l or --language to specify language to skip detection in file accepted vals [en, fr, de, es, it, ja, zh nl, ul, pt]")

args = parser.parse_args()
accepted_languages = ['en', 'fr', 'de', 'es', 'it', 'ja', 'zh', 'nl', 'ul', 'pt', None]
accepted_models = ['tiny', 'base', 'small', 'medium', 'large', 'large-v2']

audio_file = args.audio
defined_min_speakers = args.min
defined_max_speakers = args.max
forced_cpu = args.force_cpu
batch_size = args.batch
language = args.language
model = args.model

if model not in accepted_models:
    sys.exit("")

if language not in accepted_languages:
    sys.exit("Invalid language. Accepted values are ['en', 'fr', 'de', 'es', 'it', 'ja', 'zh', 'nl', 'ul', 'pt']")

device, compute_type = ("cuda", "float16") if torch.cuda.is_available() and not forced_cpu else ("cpu", "int8")

model = whisperx.load_model(model, device, compute_type=compute_type)

audio = whisperx.load_audio(audio_file)
if language == None:
    result = model.transcribe(audio, batch_size=batch_size)
else:
    result = model.transcribe(audio, batch_size=batch_size, language=language)

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model

# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

#print(result["segments"]) # after alignment

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

access_token = os.getenv("diarize_token")
diarize_model = whisperx.DiarizationPipeline(use_auth_token=access_token, device=device)

if defined_min_speakers and defined_max_speakers:
    diarize_segments = diarize_model(audio_file, min_speakers=defined_min_speakers, max_speakers=defined_max_speakers)
else:
    diarize_segments = diarize_model(audio_file)

result = whisperx.assign_word_speakers(diarize_segments, result)

with open('output.txt', 'w') as file:
    for segment in result["segments"]:
        # Extract the speaker, start time, end time and text from each segment
        speaker = segment['speaker']
        start = timedelta(seconds=int(segment['start']))
        end = timedelta(seconds=int(segment['end']))
        text = segment['text']
        # Write to the file in your specified format
        file.write(f'[{start}-{end}] {speaker} -> {text}\n')

if platform.system() == 'Windows':
    os.startfile('output.txt')
elif platform.system() == "Darwin":
    subprocess.call(('open', 'output.txt'))
elif platform.system() == "Linux":
    subprocess.call(('xdg-open', 'output.txt'))
