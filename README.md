# AN2DL Challenge 2 - Image Classification

Overleaf document: https://www.overleaf.com/5873327455zvrhvjcsjhkj#c0eac7


---Il file best_until_now si riferisce alla versione migliore di ImageNet (architettura utilizzata dal prof). Ad adesso anche la più stabile. F1 Score in validation = 0.30. Punteggio ottenuto su kaggle 0.29

--- ###Imagenet è la best entry di kaggle ma molto instabile in locale, proverò a fare delle migliorie di parametri più avanti per possibilmente migliorarla. Fate attenzione agli hyperparameters. MOOOOOOLTO SENSIBILI in questa challenge non possono essere cambiati a caso. Sconsiglio tantissimo le gridsearch a sto giro anche se sto lavorando ad una versione che le implementi non si sa mai che mi sbaglio

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

### ADVICE 05/12 - Inspect Outliers
“A single drop of poison, the purest well corrupts. Trust not the label, until the eyes have verified.”

The loss does not fall? Perhaps the teacher lies. A dog labeled as a bird; a blank image whispering silence. These are not data; they are traitors to the gradient. Plot the highest losses. Look at them. To force a model to learn the impossible, only madness to the weights it brings. Purge the traitor, and the training shall flow.

---Implementato: si riferisce semplicemente alla rimozione delle immagini sbagliate. Rimosse tutte a mano

### ADVICE 06/12 - Automated Augmentation
“The storm you design, predictable it is. The chaos the machine chooses, the true test becomes.”

You rotate, you flip... gentle distincts. But the world is cruel. Algorithms there are, that search for the perfect sequence of destruction. They learn which storm makes the model strongest. Surrender your control. Let the policy emerge from the struggle. In the calculated chaos of the machine, the robust feature is forged.

---Implementata Augmentation randomica "the storm" dell'esempio. In questo momento l'augmentation viene applicata in maniera graduale. Poco nel transfer learning e più alta nel Fine Tuning





