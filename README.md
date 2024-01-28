# Setup and Use Guide

This project is a quick start cli application for whisperX, it includes multiple flags and outputs a formatted result to a text file.
### Python3.10 is recommended
## Step 1: Clone this repo
```bash
git clone https://github.com/mikefreno/my-whisper.git
```

## Step 2: Install PyTorch

Install PyTorch by following the instructions on the official website [here](https://pytorch.org/get-started/locally/). **Make sure to use the CUDA 11.8 version if you want to use CUDA.**

## Step 3: Install whisperX
```bash
pip install git+https://github.com/m-bain/whisperx.git
```

## Step 4: Install other requirements
```bash
pip install -r requirements.txt
```

## Step 5 (optional): Replace faster_whisper utils.py (gives support for distil models -these are faster, highly recommend if running on cpu)
### unix
```bash
cp utils.py /YOUR_VENV_NAME/lib/PYTHON_VERSION/site-packages/faster_whisper/utils.py
```

### windows
```bash
copy utils.py \path\to\project_dir\YOUR_VENV_NAME\Lib\site-packages\faster_whisper\utils.py
```

## Step 6: Accept Hugging Face User Agreements & Get Access Token

Accept user agreements for both pyannote/segmentation-3.0 [here](https://huggingface.co/pyannote/segmentation-3.0) and pyannote/speaker-diarization-3.1 [here](https://huggingface.co/pyannote/speaker-diarization-3.1). Then, get the access token from Hugging Face from [here](https://huggingface.co/settings/tokens). The token should have the 'read' role.

Take token and add to a .env file `diarize_token=your_hf_token`

## How to use

Note: Two files are created, diarized_output.txt with speaker notes and base_output with base transcription

### The most basic use is the following -a is required flag
```python
python main.py -a AUDIO_FILE
```

### Optional flags
| Flag | Short | Description |
| --- | --- | --- |
| --force-cpu | -fc | Forces CPU usage, even if CUDA is available |
| --huggingface | -hf | Provide access token from CLI instead of `.env` |
| --batch | -b N | Sets batch size (N), Decrease if low on memory, defaults to 16. Only affects performance |
| --low-gpu | -lg | Recovers GPU resources |
| --number | -n N | For known exact speaker count (N) |
| --min/--max | -mn N / -mx N | For known speaker count range, must use both flags |
| --time | -t | Specifies a start time |
| --aggressive-new-line | -anl | New lines (breaks) at each same speaker chunk instead of spaces |
| --notime | -nt | Removes timestamps |
| --model | -m | To use a model other than `large-v2`. Accepted values: `['tiny', 'base', 'small', 'small.en', 'medium', 'medium.en' 'large', 'large-v1', 'large-v2', 'large-v3', 'distil-medium.en', 'distil-small.en', 'distil-large-v2']`. Affects performance and accuracy |
| --language | -l | Specify language to skip detection in file. Accepted values: `[en, fr, de, es, it, ja, zh nl, ul, pt]`. Skips detection step |
