# AN2DL Challenge 2 - Image Classification

* Overleaf document: <https://www.overleaf.com/5873327455zvrhvjcsjhkj#c0eac7>

* **Imagenet è la best entry di kaggle ma molto instabile in locale, proverò a fare delle migliorie di parametri più avanti per possibilmente migliorarla. Fate attenzione agli hyperparameters. MOOOOOOLTO SENSIBILI in questa challenge non possono essere cambiati a caso. Sconsiglio tantissimo le gridsearch a sto giro anche se sto lavorando ad una versione che le implementi non si sa mai che mi sbaglio**

## Creating  a local virtual enviroment

1. `python3 -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -r requirements.txt`

4. Installing pytorch (for nvidia gpus):
    1. run command and see wich CUDA version you need to install: `nvidia-smi.exe`
    2. Go to [pytorch](https://pytorch.org/get-started/locally/)
    3. Get the pip install command personalized for your system (for me it was: `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130`)
    4. Run it in local env

---

## References

* pytorch mdoel: <https://docs.pytorch.org/vision/main/models.html>
* paper "Deep learning-based classification of breast cancer molecular subtypes from H&E whole-slide images": <https://pmc.ncbi.nlm.nih.gov/articles/PMC11667687/>
* Macenko stain normalization: <https://www.geeksforgeeks.org/machine-learning/macenko-method-for-normalizing-histology-slides-for-quantitative-analysis/>