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

## Step 5 (optional): Replace faster_whisper utils.py (gives support for distil models)
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

Take token and add to the .env file `diarize_token={token}`

## How to use

Note: Two files are created, diarized_output.txt with speaker notes and base_output with base transcription

### The most basic use is the following -a is required flag
```python
python main.py -a AUDIO_FILE
```


| Flag | Description |
| --- | --- |
| `-fc`, `--force-cpu` | Forces the CPU to be used even if CUDA is available |
| `-hf`, `--huggingface` | Provide access token in CLI instead of loading from `.env` |
| `-b N`, `--batch N` | Sets batch size to `N`. Decrease if low on memory, defaults to 8. Only affects performance |
| `-lg`, `--low-gpu` | Recovers GPU resources; use if you have low GPU resources |
| `-n`, `--number N` | Use if exact speaker count (`N`) is known |
| `--min N`,`--max N` | If speakers are known to be within a range, these flags can help increase accuracy. Both should be used together. |
| `-t`, `--time` | Specifies a time to use as the start time |
| `-anl`, `--aggressive-new-line` | Puts new lines (breaks) at each same speaker chunk instead of spaces |
| `-nt`, `--notime` | Removes timestamps |
| `-m`, `--model` | To use a model other than `large-v2`. Accepted values: `['tiny', 'base', 'small', 'medium', 'large', 'large-v2']`. Affects performance and accuracy |
| `-l`, `--language` | Specify language to skip detection in file. Accepted values: `[en, fr, de, es, it, ja, zh nl, ul, pt]`. Skips detection step |
