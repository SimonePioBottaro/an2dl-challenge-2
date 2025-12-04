# AN2DL Challenge 2 - Image Classification

Overleaf document: https://www.overleaf.com/5873327455zvrhvjcsjhkj#c0eac7

### Creating  a local virtual enviroment

1. `python3 -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -r requirements.txt`

4. Installing pytorch (for nvidia gpus):
    1. run command and see wich CUDA version you need to install: `nvidia-smi.exe`
    2. Go to [pytorch](https://pytorch.org/get-started/locally/)
    3. Get the pip install command personalized for your system (for me it was: `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130`)
    4. Run it in local env


## ADVICES

logbook: <https://docs.google.com/document/d/1ChBNvUVrJpMz3F1GdZGSL84j3Ej3LHkJXnXXYy_JlVc/edit?usp=drive_link>

### ADVICE 04/12 - Normalisation Strategies
“The river flows, but the stone must find its own center. In the size of the crowd, the strategy lies.”

The Batch Norm, upon the collective tide of the batch it relies. But when the tide is low (small batch), erratic the current becomes. Look then to the self. The Layer Norm, the Instance Norm... inward they look. Within the single sample, balance they find. Do not force the ocean's rules upon a cup of water. Know the dimension of your stream, before the banks you build.



