import scanpy as sc
import numpy as np
import csv

# modify for the path of the training dataset
adata_train = sc.read_h5ad("/gpfs/gibbs/pi/zhao/tl688/deconvdatasets/demo_train.h5ad")

# modify for the path of the testing dataset
adata_test = sc.read_h5ad("/gpfs/gibbs/pi/zhao/tl688/deconvdatasets/demo_test.h5ad")


def generate_bulk(adata, pseudo_length=100, num_of_gene=150):
    question_list, answer_list = [], []
    for i in range(adata.shape[0]//pseudo_length):
        adata_group = adata[100*i:100*(i+1)]
        value_index = np.argsort(adata_group.X.sum(axis=0))
        gene_rank = np.array(adata_group.var_names)[value_index][:num_of_gene]
        question_list.append(gene_rank)
        answer_list.append(adata_group.obs.Celltype.value_counts().index[0])
    return question_list, answer_list

question_list, answer_list = generate_bulk(adata_train)
construct_train = {}
construct_train['Q'] = question_list 
construct_train['A'] = answer_list 

question_list, answer_list = generate_bulk(adata_test)
construct_test = {}
construct_test['Q'] = question_list 
construct_test['A'] = answer_list 

with open('evaluation/deconv_gt.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(answer_list)

question_list = []
answer_list = []
for i in range(len(construct_train['Q'])):
    ques_start = 'This mixture has genes ranked by their expression as: '
    for item in construct_train['Q'][i]:
        ques_start += item
        ques_start += ' '
    ques_start += '. What is the most abundant cell type in this mixture?'

    question_list.append(ques_start)
    answer_start = 'The most abundant cell type is: '
    for item in construct_train['A'][i]:
        answer_start += item
        answer_start +='.'
    answer_list.append(answer_start)

with open("pancreas_train_input_deconv.txt", "w") as f:
    for item,label in zip(question_list, answer_list):
        f.write(item)
        f.write("\n")
        f.write(label)
        f.write("\n")

question_list = []
answer_list = []
for i in range(len(construct_test['Q'])):
    ques_start = 'This mixture has genes ranked by their expression as: '
    for item in construct_test['Q'][i]:
        ques_start += item
        ques_start += ' '
    ques_start += '. What is the most abundant cell type in this mixture?'

    question_list.append(ques_start)
    answer_start = 'The most abundant cell type is: '
    for item in construct_test['A'][i]:
        answer_start += item
        answer_start +='.'
    answer_list.append(answer_start)

with open("pancreas_test_input_deconv.txt", "w") as f:
    for item,label in zip(question_list, answer_list):
        f.write(item)
        f.write("\n")
        f.write(label)
        f.write("\n")
