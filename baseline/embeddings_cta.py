from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import pandas as pd
import scanpy as sc
import numpy as np
import sklearn
try:
    import hnswlib
    hnswlib_imported = True
except ImportError:
    hnswlib_imported = False
    print("hnswlib not installed! We highly recommend installing it for fast similarity search.")
    print("To install it, run: pip install hnswlib")
from scipy.stats import mode

adata_train = sc.read_h5ad("/gpfs/gibbs/pi/zhao/tl688/deconvdatasets/demo_train.h5ad")
adata_test = sc.read_h5ad("/gpfs/gibbs/pi/zhao/tl688/deconvdatasets/demo_test.h5ad")

model_name,train_file, test_file, output_dir = "gpt2-large", "pancreas_train_input.txt", "pancreas_test_input.txt", "gpt2_1024_annot_meidum"

# Load GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name, eos_token='\n')

df_train = pd.read_csv(test_file, header = None)

model = model.to('cuda')
model.eval()

emb_v = []
for i in range(0, len(df_train), 2):
    model_inputs = tokenizer(df_train.loc[i].values[0], return_tensors='pt').to('cuda')
    outputs = model(**model_inputs, output_hidden_states=True)
    emb_v.append(outputs.hidden_states[0][0,:,:].sum(axis=0).detach().cpu().numpy())


gpt2_embeddings = np.array(emb_v)
adata_test.obsm['X_gpt2'] = gpt2_embeddings
adata_test.write_h5ad('demo_test_gpt2_large.h5ad')


adata_train = sc.read('demo_train_gpt2_large.h5ad')
adata_test = sc.read('demo_test_gpt2_large.h5ad')

genePT_w_emebed_train =  adata_train.obsm['X_gpt2']
genePT_w_emebed_test = adata_test.obsm['X_gpt2']

y_train = adata_train.obs.Celltype
y_test = adata_test.obs.Celltype

# cell type clustering
# very quick test
k = 10 # number of neighbors
ref_cell_embeddings = genePT_w_emebed_train
test_emebd = genePT_w_emebed_test
neighbors_list_gpt_v2 = []
if hnswlib_imported:
    # Declaring index, using most of the default parameters from https://github.com/nmslib/hnswlib
    p = hnswlib.Index(space = 'cosine', dim = ref_cell_embeddings.shape[1]) # possible options are l2, cosine or ip
    p.init_index(max_elements = ref_cell_embeddings.shape[0], ef_construction = 200, M = 16)

    # Element insertion (can be called several times):
    p.add_items(ref_cell_embeddings, ids = np.arange(ref_cell_embeddings.shape[0]))

    # Controlling the recall by setting ef:
    p.set_ef(50) # ef should always be > k

    # Query dataset, k - number of closest elements (returns 2 numpy arrays)
    labels, distances = p.knn_query(test_emebd, k = k)

idx_list=[i for i in range(test_emebd.shape[0])]
gt_list = []
pred_list = []
for k in idx_list:
    # this is the true cell type
    gt = y_test[k]
    if hnswlib_imported:
        idx = labels[k]
    else:
        idx, sim = get_similar_vectors(test_emebd[k][np.newaxis, ...], ref_cell_embeddings)
    pred = mode(y_train[idx], axis=0)
    neighbors_list_gpt_v2.append(y_train[idx])
    gt_list.append(gt)
    pred_list.append(pred[0][0])

### get evaluation statistics
sklearn.metrics.accuracy_score(gt_list, pred_list)