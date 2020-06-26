# gpt2lsm
Integration of LSM in GPT2 Model


### Main files:

The files involve the implementation of the Conversational AI model from Huggingface: https://github.com/huggingface/transfer-learning-conv-ai
The implementation was modified to integrate the LSM objectives into the model.

#### train.py
Under update() you can see the loss integration of the LSM mechanism during training.

#### LSM.py
Class/code for the calculation of the LSM loss.

#### convai_evaluation.py
File containing the code to evaluate a model with different metrics

#### interact.py
File containing the code to interact with a model

#### utils.py, example_entry.py, test_special_tokens.py
Containg helper functions used in the files above and test functions, these were not modified during the project.


### Commands

#### Training
The model can be trained with the following command:

It can also be trained in parallel on multiple GPUs as follows:

#### Interact with the model


#### Evaluation



If any problems occur when running the code, please send me an email on: bartvanvulpen@icloud.com


