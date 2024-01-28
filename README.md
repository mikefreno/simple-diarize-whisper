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

## Optional flags

```bash
-fc or --force-cpu
```
forces the cpu to be used even if cuda is available

```bash
-hf or --huggingface
```
provide access token in cli instead of from loading from .env

```bash
-b N or --batch N
```
-b N sets batch size, decrease if low on mem, defaults to 8 - only affects performance

```bash
-lg or --low-gpu
```
recovers gpu resources, use if you have low gpu resources

### the following speaker count indicators can improve accuracy
```bash
-n or --number N
```
if exact speaker count number is known

```bash
--min N
```
if speakers known in range, must use with --max flag can increase accuracy

```bash
--max N
```
if speakers known in range, must use with --min flag can increase accuracy

```bash
-t or --time
```
specifies a time to use as the start time
```bash
-anl or --aggressive-new-line
```
puts new lines(breaks) at each same speaker chunk instead of spaces

```bash
-nt or --notime
```
removes time stamps

```bash
-m or --model
```
to use a model other than large-v2, accepted vals ['tiny', 'base', 'small', 'medium', 'large', 'large-v2'] - affects performance and accuracy

```bash
-l or --language
```
specify language to skip detection in file accepted vals [en, fr, de, es, it, ja, zh nl, ul, pt], skips detection step 
