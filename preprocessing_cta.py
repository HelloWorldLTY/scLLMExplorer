import scanpy as sc
import numpy as np

# modify for the path of the training dataset
adata_train = sc.read_h5ad("/gpfs/gibbs/pi/zhao/tl688/deconvdatasets/demo_train.h5ad")

# modify for the path of the testing dataset
adata_test = sc.read_h5ad("/gpfs/gibbs/pi/zhao/tl688/deconvdatasets/demo_test.h5ad")

construct_train = {}
question_list = []
answer_list = []

num_of_gene = 150
for i in adata_train:
    value_index = np.argsort(i.X)
    gene_rank = np.array(i.var_names)[value_index][0][0:num_of_gene]
    question_list.append(gene_rank)
    answer_list.append(i.obs.Celltype.values)

construct_train['Q'] = question_list 
construct_train['A'] = answer_list 

construct_test = {}
question_list = []
answer_list = []

for i in adata_test:
    value_index = np.argsort(i.X)
    gene_rank = np.array(i.var_names)[value_index][0][0:num_of_gene]
    question_list.append(gene_rank)
    answer_list.append(i.obs.Celltype.values)

construct_test['Q'] = question_list 
construct_test['A'] = answer_list 

question_list = []
answer_list = []
for i in range(len(construct_train['Q'])):
    ques_start = 'This cell has genes ranked by their expression as: '
    for item in construct_train['Q'][i]:
        ques_start += item
        ques_start += ' '
    ques_start += '. What is the cell type of this cell?'

    question_list.append(ques_start)
    answer_start = 'The cell type is: '
    for item in construct_train['A'][i]:
        answer_start += item
        answer_start +='.'
    answer_list.append(answer_start)

with open("pancreas_train_input.txt", "w") as f:
    for item,label in zip(question_list, answer_list):
        f.write(item)
        f.write("\n")
        f.write(label)
        f.write("\n")

question_list = []
answer_list = []
for i in range(len(construct_test['Q'])):
    ques_start = 'This cell has genes ranked by their expression as: '
    for item in construct_test['Q'][i]:
        ques_start += item
        ques_start += ' '
    ques_start += '. What is the cell type of this cell?'

    question_list.append(ques_start)
    answer_start = 'The cell type is: '
    for item in construct_test['A'][i]:
        answer_start += item
        answer_start +='.'
    answer_list.append(answer_start)

with open("pancreas_test_input.txt", "w") as f:
    for item,label in zip(question_list, answer_list):
        f.write(item)
        f.write("\n")
        f.write(label)
        f.write("\n")