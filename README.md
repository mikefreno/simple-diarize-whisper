# Setup and Use Guide

This project is a quick start for whisperX, it includes multiple cli flags and outputs a formatted result to a text file.

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
| --batch | -b N | Sets batch size (N), can improve performance |
| --low-gpu | -lg | Recovers GPU resources |
| --number | -n N | For known exact speaker count (N) |
| --min/--max | -mn N / -mx N | For known speaker count range, use both flags |
| --time | -t | Specifies a start time |
| --aggressive-new-line | -anl | New lines at each same speaker chunk |
| --notime | -nt | Removes timestamps |
| --model | -m | Specify model, affects performance & accuracy |
| --language | -l | Specify language, skips detection step |
