[[Original PIGEON Paper]](https://openaccess.thecvf.com/content/CVPR2024/html/Haas_PIGEON_Predicting_Image_Geolocations_CVPR_2024_paper.html) [[Website]](https://lukashaas.github.io/PIGEON-CVPR24/)


# CVPR 2024 Highlight – PIGEON: Predicting Image Geolocations
This repository contains the code for the CVPR 2024 paper highlight and demo *PIGEON: Predicting Image Geolocations*. The paper was authored by Lukas Haas, Michal Skreta, Silas Alberti, and Chelsea Finn at Stanford University.

This repository is purely meant for the academic validation of the paper's code. Geocell shapes and coordinates, training and validation datasets, and model weights are not provided as part of this release. Please read the section *Ethical considerations* in our paper to learn more.

## Citing This Work

Please cite our work as follows:

```
@InProceedings{Haas_2024_CVPR,
    author    = {Haas, Lukas and Skreta, Michal and Alberti, Silas and Finn, Chelsea},
    title     = {PIGEON: Predicting Image Geolocations},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {12893-12902}
}
```


## Contents

This root folder contains the following code files. Each directory contains another README.

### ```config.py```

Definition of many different constants and paths needed in the rest of the repository.

### ```env.yml```

Conda environment used in this project.

### ```get_auxiliary_data.sh```

Script to download auxiliary data used in this project.

### ```run.py```

This file is the entry point to this project, loads and preprocesses the data, and depending on the command pretrains, finetunes, embeds, or evaluate data.
