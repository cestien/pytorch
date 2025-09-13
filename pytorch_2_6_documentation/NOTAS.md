<link href="/home/cestien/memoria/template/nota_template/miestilo.css" rel="stylesheet"></link>

# Resumen de la documentación de pytorch
Contiene un resumen de pytorch docs, tutorials y torchaudio. 

Primero estaría bueno ver este curso [Practical deep learning](https://course.fast.ai/), curso de deep learning con notebooks.

## Documentación de pytorch 2.6 al 4/25  
  - [Python API](https://pytorch.org/docs/stable/index.html). Contiene todas las funciones que necesitamos. 
  - [Pytorch tutorials](https://pytorch.org/tutorials/). Acá están todos los tutorials. Algunos que nos pueden servir:
    - [Learn the basic](https://pytorch.org/tutorials/beginner/basics/intro.html). Es introductorio, sería el lugar de donde empezar con pytorch. Es una serie de notebooks.    
      - Quickstart
      - Tensors
      - Datasets and DataLoaders
      - Transforms
      - Build Model
      - Automatic Differentiation
      - Optimization Loop
      - Save, Load and Use Model
    - **What is torch.nn really?**. Es una notebook que la descargamos.
    - [NLP from Scratch](https://pytorch.org/tutorials/intermediate/nlp_from_scratch_index.html). No lo vimos pero parece interesante.
    - **Audio**. La parte de tutorials está deprecada y fue fusionada con los tutorials de torchaudio. Mirarlo de ahi.
    - [Tensorboard](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html). Explica como usar tensorbard en python.
    - **A guide on good usage of `non_blocking` and `pin_memory()` in PyTorch**. Es una notebook que descargamos.
    - [Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html). Detalles de autograd.
    - **Recetas**. Notebooks con recetas de temas específicos. Las notebooks estan en este directorio:
      - How to use TensorBoard with PyTorch
      - Defining a Neural Network in PyTorch
      - What is a state_dict in PyTorch.
      - Saving and Loading Models
      - Warmstarting model using parameters from a different model in PyTorch
      - Zeroing out gradients in PyTorch
      - PyTorch Benchmark
      - Timer quick start
      - PyTorch Profiler
      - Tips for Loading an `nn.Module` from a Checkpoint
      - Speech Command Classification with torchaudio
- [torchaudio](https://pytorch.org/audio/stable/). Es una de las bibliotecas de pytorch. Contiene todo lo necesario para manejar audio. Consta de tres partes principales.
  - **API Tutorials y Pipeline Tutorials**. Son notebooks de las cuales descargamos algunas:
    - Audio I/O
    - AudioEffector Usages
    - Audio Resampling
    - Audio Data Augmentation
    - Audio Feature Extractions
    - Audio Feature Augmentation
    - CTC forced alignment API tutorial
    - Oscillator and ADSR envelope Audio Datasets
    - Speech Recognition with Wav2Vec2
    - ASR Inference with CTC Decoder
    - ASR Inference with CUDA CTC Decoder
    - Online ASR with Emformer RNN-T
    - Device ASR with Emformer RNN-T
    - Device AV-ASR with Emformer RNN-T
    - Forced Alignment with Wav2Vec2
    - Forced alignment for multilingual data
    - Audio Datasets
  - **Training Recipes**. Son sistemas completos descriptos en github.
  - **Python API Reference**. Contiene todas las funciones que necesitamos.
  - **Python Prototype API Reference**.

## Augmenter de datos
En speechbrain:
  - augmenter es una función que define como agregar los datos aumentados
  - freq_domain
    - randomshift
    - spectrogramdrop
    - warping
  - preparation module
    - prepare csv
    - prepare datasete from url
    - write csv
  - time domain
    - AddNoise
    - AddReverb
    - ChannelDrop
    - ChannelSwap
    - CutCat
    - DoClip
    - DropBitResolution
    - DropChunk
    - DropFreq
    - FastDropChunk
    - RandAmp
    - Resample
    - SignFlip
    - SpeedPerturb


    Nuevo token
    