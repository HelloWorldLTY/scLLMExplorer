# scLLMExplorer

# Install

We upload the conda environment file (It contains the versions of required packages and other information). To install it, please use:

```
conda env create --name cpsc488 --file=cpsc488_env.yml
```

# Usage

To run our codes, please first pre-process the given dataset based on different tasks.

For cell-type annotation task, please use:

```
python preprocessing_cta.py
```

For deconvolution task, please use:

```
python preprocessing_deconv.py
```

Then for different tasks, we also have different main files.

For cell-type annotation task, please use:

```
python cta_main.py
```

For deconvolution task, please use:

```
python deconv_main.py
```

For the usage of baseline models or evaluation, please refer the corresponding folders (**baseline** and **evaluation**)

# Tutorial

Please refer the notebook under the **tutorial** folder for running the codes of cell-type annotation task and deconvolution task, also the evaluation.

# Copyright

If you have questions, please contact Tianyu Liu or Zhiyuan Cao.

tianyu.liu@yale.edu.

zhiyuan.cao@yale.edu.