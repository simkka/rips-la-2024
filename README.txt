For generation_software/data_processing:

1. Before running any of the scripts for data_processing, please make sure you are in the correct directory (generation_software/data_processing)

2. The order of running the Python scripts for data_processing should be:
    sample_NDSSL_subgraphs.py
    addNodeInfo.py
    networkxConversion.py
    splitDataset.py
    
3. More information about how to run each of the Python scripts above can be found in the comments in each script.

For generation_software/graph-generation-adaption:

1. The graph-generation-adaption directory contains code adapted from: https://github.com/AndreasBergmeister/graph-generation
When in doubt, it's recommended to check out the information in this repository.

2. How to use NDSSL data to train the model:
    (1) Follow the instructions for generation_software/data_processing above to process the dataset.
    (2) Move the processed dataset from (1) to generation_software/graph-generation-adaption/data/
    (3) Create a config (.yaml) file under generation_software/graph-generation-adaption/config/experiment. You can check
    other .yaml files under generation_software/graph-generation-adaption/config/experiment to get an idea of what is contained
    in each config file. You can finetune more parameters, such as validation interval, under generation_software/graph-generation-adaption/config/config.yaml
    (4) With both processed dataset under generation_software/graph-generation-adaption/data/ and configuration file under 
    generation_software/graph-generation-adaption/config/experiment, you can cd into generation_software/graph-generation-adaption.
    Then you can run the following command to train the model: python3 main.py +experiment=EXPERIMENT_NAME
    Note that EXPERIMENT_NAME corresponds to the name you set in the configuration file you just created. For example,
    in NDSSL.yaml, we specify "name: NDSSL". So to train the model, we replace EXPERIMENT_NAME with NDSSL.
    