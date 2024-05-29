# IAPR: Project ‒  Coin detection


**Group ID:** 12

**Author 1 (sciper):** Vray Alexandre (310104)  
**Author 2 (sciper):** Aellen Edgar (311574)   
**Author 3 (sciper):** Robrdet Alexis (371391)


### 1. Report
- Please find our report as a Jupyter notebook [here](report.ipynb)

### 2. Other Notebooks
In the report are linked other notebooks which explain and perform the main tasks of the model
- [`create_dataset.ipynb`](create_dataset.ipynb) perform the localization and patch extraction on all the training set and give labels to every patch.
- [`classification_training.ipynb`](classification_training.ipynb) is actually training the neural network model to classify coins.
- [`generate_predictions.ipynb`](generate_predictions.ipynb) is the final notebook to detect coins on the test dataset.

### 3. Utils files
A set of utils files is only implement to have a lisible code
- [`utils.py`](utils/utils.py) has functions to load/save images, select images or split the dataset.
- [`display.py`](utils/display.py) helps to display images and coins.
- [`localization.py`](utils/localization.py) has functions to localize coins and extract them (there procedure is explain in the report)
- [`model_setup.py`](utils/model_setup.py) contain a created CoinDataset class which allows us to perform data transformation / augmentation during classification training
