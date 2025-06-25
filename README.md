# TFM_MIA_2023_CarlosBreuerCarrasco
The repository contains the code used for the master final project. It also contains the models history, the figures saved with the model architectures and results, the classifications reports and some scripts to preprocess and to apply data augmentation to the images of the dataset.

The project tries to classify 745 MRI images of the human brain in two different classes: healthy brains and brains with the rare disease HSVE (Herpetic encephalitis). The MRI images have been obtained from the repository Radiopaedia and has MRI images form diferent axis such as coronal, sagital and axial axis. They also have different formats such as FLAIR, T1, T2, T1C+ y T2C+.

Specifically, the repository contains the following directories:

- Data_augmentation_script: contains the python script to generate synthetic images from the original images.
- Experimentos_iteracion1: contains the python scripts which implement the models of the first iteration of experiments. These experiments try to evaluate the performance of different CNN pre-trained models such as 'ResNet50', 'VGG16', 'VGG19', 'InceptionV3', 'MobileNet' and a new architecture made from scratch.
- Experimentos_iteracion2: contains the python scripts which implement the models of the second iteration of experiments. These experiments try to evaluate the performance of the 'ResNet50' with different subsets of the initial dataset of MRI images.
- Experimentos_iteracion3: contains the python scripts which implement the models of the second iteration of experiments. These experiments try to evaluate the performance of the 'ResNet50' aplying some Deep Learning regularization techniques to reduce the overfitting of the previous models.

- grafo_modelos_implementados.jpg: is a scheme in which the models implemented implemented in the different iterations are distinguished easily, indicating their characteristics and differences with their main accuracy results in test_set.
