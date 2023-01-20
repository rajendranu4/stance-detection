# Exploration of Contrastive Learning Strategies toward Robust Stance Detection Systems

Following are the steps to successfully learn representations from stance examples and leverage the learned sentence representations in the Stance Detection downstream task.

Run below pip command to install the required python packages
```bash
pip install -r requirements_std.txt
```
### Data Preparation
 - Save the stance dataset - train/dev/test (.csv) in the **data/** folder
	 - Each example should have a text and its corresponding label (**0** for label **'against'** and **1** for label **'support'**)
	 - The header of the file should be **text** and **label**
 - Prepare the adversarial attack perturbed test data (**spelling, synonym, tautology**) by executing the following commands (replace **spelling** with **negation** / **synonym** to prepare tautology perturbed / synonyms replaced test data)
	 ```bash
	 -py scripts\adversarial_attack_std.py baseline spelling 30
	 ```

### Training Configuration
The following are the training configurations to learn sentence representations. The file **training_config/declutr_std.jsonnet** needs to be updated as follows.

 - set **maximum_length** of the sentence (no. of words) to be considered for training
 - set **train_data_path** to the location where the stance dataset is saved
 - set **triplet_mining_strategy** to one of the values **random** / **hard** / **hard_easy** depending on the type of triplet mining strategy for the training with Contrastive Learning

### Execution
Run the below command (allennlp train) to learn the sentence representation with Contrastive Learning and MLM objectives

```bash
allennlp train /
"training_config/std.jsonnet" /
--serialization-dir "sent_repr_std" /
--include-package "std"
```
The above command creates a new folder with the name sent_repr_std which contains the trained model. Run the below command to save the trained model to load with the Hugging Face transformers library

```bash
python scripts/save_pretrained_hf.py /
--archive-file "sent_repr_std" /
--save-directory "sent_repr_std_tf"
```

### Leveraging learned sentence representations in Stance Detection downstream task
Update the following variables in the downstream_StD/fine_tune_model_contrastive.ipynb file to have the dataset files and model path set up for finetuning.

 - **df_input** - training data file path 
 - **df_input_val** - validation data file path  
 - **df_test** - test data file path  
 - **model_path** - saved model path
 - **pretrained_model_tokenizer_path** - saved model tokenizer path 
 - **df_test_adv_neg** - test data file perturbed with negation attack
 - **df_test_adv_syn** - test data file perturbed with synonym attack
 - **df_test_adv_spell** - test data file perturbed with spelling attack

Execute the ipynb file to fine tune the model with the stance dataset and testing against the perturbed test data


### Pseudo Labels for Unlabeled dataset
The clustering method introduced by Li et al. [[1]](#1) in _Unsupervised Belief Representation Learning with
Information-Theoretic Variational Graph Auto-Encoders_ is used to generate pseudo labels for the unlabeled datasets used in our experiments.


### References & Credits
[GitHub - JohnGiorgi/DeCLUTR: "DeCLUTR: Deep Contrastive Learning for Unsupervised Textual Representations". ](https://github.com/JohnGiorgi/DeCLUTR)

<a id="1">[1]</a> “Unsupervised belief representation learning with information theoretic variational graph auto-encoders,” 2022. doi: 10.1145/3477495.3532072.
[Online]. Available: https://doi.org/10.1145%2F3477495.3532072.

