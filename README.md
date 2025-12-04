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



