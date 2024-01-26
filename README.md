# Project Setup Guide

This project requires the setup of deep learning libraries PyTorch and whisperX, which are used for computations and processing of the models. Additionally, this project requires accepting user agreements and getting access tokens from Hugging Face for pyannote's segmentation and speaker diarization models.

## Step 1: Install PyTorch

Install PyTorch by following the instructions on the official website [here](https://pytorch.org/get-started/locally/). Make sure to use the CUDA enabled version if you want to use CUDA.

## Step 2: Install whisperX
```bash
pip install git+https://github.com/m-bain/whisperx.git
```

Step 3: Accept Hugging Face User Agreements & Get Access Token

Accept user agreements for both pyannote/segmentation-3.0 [here](https://huggingface.co/pyannote/segmentation-3.0) and pyannote/speaker-diarization-3.1 [here](https://huggingface.co/pyannote/speaker-diarization-3.1). Then, get the access token from Hugging Face from [here](https://huggingface.co/settings/tokens). The token should have the 'read' role.

Take token and add to .env file `diarize_token={token}`

After these steps are completed, you are ready to use the project code.
