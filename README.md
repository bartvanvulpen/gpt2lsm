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
```shell
function () { return "This code is highlighted as Javascript!"}
```
#### Training
The model can be trained with the following command:
python ./train.py --model_checkpoint=gpt2-medium --lm_coef=2.0 --max_history=2 --n_epochs=1 --num_candidates=4 --personality_permutations=2 --train_batch_size=2 --valid_batch_size=2
It can also be trained in parallel on multiple GPUs as follows:
`python -m torch.distributed.launch --nproc_per_node=4 train.py --gradient_accumulation_steps=4 --model_checkpoint=gpt2-medium --lm_coef=2.0 --max_history=2 --n_epochs=1 --num_candidates=4 --personality_permutations=2 --train_batch_size=2 --valid_batch_size=2`

The model files are usually stored in a 'runs' directory, containing the date, GPU node and model name of the run.

#### Interact with the model
To interact with a trained model file (could be either baseline or baseline + LSM loss integration), use the something following command: 
`python interact.py --model gpt2 --top_k 40 --model_checkpoint runs/Jun21_14-31-17_r28n4.lisa.surfsara.nl_gpt2-medium`

To interact with a trained model file including weighted decoding, use the following command:
`python transfer-learning-conv-ai/interact.py --model gpt2 --top_k 40 --model_checkpoint runs/Jun21_14-31-28_r33n6.lisa.surfsara.nl_gpt2-medium --wd true --wd_weight=2.5`



If any problems occur when running the code, please send an email to: bartvanvulpen@icloud.com


