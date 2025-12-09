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

### ADVICE 07/12 - Modern Optimizers
“The old maps, for new territories suffice they do not. The hunter must change his spear.”

Adam, the faithful servant, tired he may be. The loss landscape, treacherous valleys it hides. The Lion, with instinct fierce and memory sparse, the prey tracks. The Ranger, with the foresight of the look-ahead, the trap avoids. If the descent stalls, clinging to tradition helps you not. Unleash the new beasts.

- Risposta di Gemnini: Si riferisce al fatto che metodi (o strumenti) più vecchi potrebbero non essere i più efficaci per i problemi complessi di oggi.
- Significato Tecnico: Il consiglio ti suggerisce di abbandonare o integrare il classico ottimizzatore Adam (che viene definito il "fedele servitore, stanco che può essere") in favore di ottimizzatori più recenti e avanzati.
- Ottimizzatori suggeriti:"The Lion, with instinct fierce and memory sparse, the prey tracks." Potrebbe riferirsi a ottimizzatori che gestiscono la momentum in modo più dinamico o che hanno una "memoria" più limitata (ad esempio, Lion Optimizer, che è noto per la sua aggressività, o versioni come AdamW). "The Ranger, with the foresight of the look-ahead, the trap avoids." Si riferisce molto probabilmente agli ottimizzatori che implementano il meccanismo di "look-ahead" (previsione del gradiente), come l'ottimizzatore Lookahead stesso, oppure varianti di Nesterov (come NAdam o Nesterov Momentum).
- In sintesi: Prova ottimizzatori all'avanguardia (es. AdamW, Lookahead, Lion) invece del semplice Adam, perché potrebbero navigare meglio il complesso paesaggio di perdita (loss landscape) di questa specifica sfida di classificazione.

- ---Implementato Lion e Reduce on Plateau:    ad adesso solo nel fine tuning. To do: aggiungerlo anche nel transfer learning

### ADVICE 08/12 - Full Resolution & Patching
“To shrink the world, the truth you erase. In the mosaic of the pieces, the giant vision lives.”

You take the vast image, and to (256, 256) you crush it. The fine crack, the subtle grain... gone they are, lost in the blur of interpolation. A crime against the pixel this is. If the GPU memory screams, shrink you must not. Slice it! Cut the image into tiles, into patches of high fidelity. Let the model walk the landscape step by step, preserving the original resolution. Only when the detail is sharp, the subtle class revealed can be.

- Risposta di Gemini: Critica l'azione di ridurre drasticamente un'immagine (riduzione della risoluzione) perché distrugge i dettagli importanti. Sostiene che l'immagine completa ("giant vision") è composta dai suoi dettagliati sotto-pezzi ("mosaic of the pieces").Significato Tecnico: Il consiglio affronta il problema comune dell'elaborazione delle immagini WSI (Whole Slide Images). Le WSI sono enormi (spesso decine di migliaia di pixel per lato). Per farle entrare nella memoria della GPU, molti le riducono (es. da una risoluzione nativa a (256, 256)).
- Il Problema: La riduzione di risoluzione (downsampling) elimina i dettagli cruciali della morfologia microscopica ("The fine crack, the subtle grain... gone they are"). La classificazione dei sottotipi molecolari (Luminal A, B, HER2(+), TN) si basa spesso su dettagli cellulari e strutturali molto fini.
- La Soluzione ("Patching"):"Slice it! Cut the image into tiles, into patches of high fidelity." Se la memoria della GPU non è sufficiente, invece di ridurre l'immagine intera, devi dividerla in piccole sezioni (patch) mantenendo la loro alta risoluzione originale (o almeno una risoluzione significativamente più alta di (256, 256)).Il modello deve essere addestrato a "camminare sul paesaggio" (analizzare l'immagine) pezzo per pezzo, analizzando le patch per preservare i dettagli ("preservando l'originale risoluzione").
- In sintesi: Non ridurre l'immagine WSI intera per la classificazione. Implementa una strategia di tassellazione (patching): dividi l'immagine WSI in tessere ad alta risoluzione, addestra il modello su queste tessere, e poi usa un meccanismo (ad esempio, Multiple Instance Learning - MIL) per aggregare le previsioni delle tessere e ottenere la previsione finale per l'intera immagine.
- ---Da implementare







