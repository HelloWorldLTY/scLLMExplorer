
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import pipeline
import pandas as pd
import scanpy as sc
import numpy as np

model_name,train_file, test_file, output_dir = "gpt2", "pancreas_train_input.txt", "pancreas_test_input.txt", "gpt2_annot"
generator = pipeline('text-generation', model=output_dir, tokenizer=output_dir, max_new_tokens=1000) # use your trained model name

# Load GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained(output_dir)
tokenizer = GPT2Tokenizer.from_pretrained(output_dir)

import pandas as pd
readpath = pd.read_table("pancreas_test_input.txt", header=None)

accu = []
for i in range(0, len(readpath), 2):
    model_inputs = tokenizer(readpath.loc[i].values[0][0:1000], return_tensors='pt').to('cuda:0')
    greedy_output = model.generate(**model_inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
    gpt2_out = tokenizer.decode(greedy_output[0], skip_special_tokens=True)
    accu.append(gpt2_out.split('?')[1].split(': ')[1].split('.')[0])

adata_test = sc.read_h5ad("/gpfs/gibbs/pi/zhao/tl688/deconvdatasets/demo_test.h5ad")
count = 0
for i,j in zip(list(adata_test.obs.Celltype), accu):
    if i == j:
        count += 1
accu = count/len(accu)

print("Accuracy: ", accu)
