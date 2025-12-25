<h1 align="center">
<img src="docs/images/logo.png" alt="embodied-logo" width="40" height="40" style="vertical-align: middle; margin-top: -12px;">
EffiVLM-Bench (Visualization Enhanced)
</h1>

<div align="center">
  <h3>
    ⚠️ Fork Notice / 修改说明
  </h3>
  <p>
    This repository is a <strong>modified version</strong> of the original <a href="https://github.com/EffiVLM-Bench/EffiVLM-Bench">EffiVLM-Bench</a>. <br>
    We have extended the original framework by integrating <strong>visualization capabilities</strong> to facilitate better understanding of model inference and acceleration methods.
  </p>
  <p>
    本项目是在开源项目 <strong>EffiVLM-Bench</strong> 的基础上进行的修改与扩展。<br>
    我们重点加入了 <strong>Streamlit 可视化功能</strong>，用于展示推理过程及 Demo 交互。
  </p>
</div>

<br>
<p align="center">
  📄  <a href="https://arxiv.org/abs/2506.00479"><strong>Paper</strong></a> |  
  🏠 <a href="https://effivlm-bench.github.io/"><strong>Project Website</strong></a>
</p>


<p align="center">
    <a href="https://kugwzk.github.io/">Zekun Wang*</a>, 
    <a href="">MingHua Ma*</a>, 
    <a href="">Zexin Wang*</a>, 
    <a href="">Rongchuan Mu*</a>, 
    <a href="">liping shan</a>, 
    <a href="https://scholar.google.com/citations?user=VJtmTREAAAAJ&hl=en">Ming Liu</a>, 
    <a href="https://scholar.google.com/citations?user=LKnCub0AAAAJ">Bing Qin</a>, 

</p>
<p align="center">Harbin Institute of Technology</p>


<!-- <img src="docs/images/main_kvcache.jpg" width="100%" /> -->

# 🔥 Overview 
We introduce EffiVLM-Bench, a comprehensive benchmark designed to systematically evaluate training-free acceleration methods for Large Visual-Language Models (LVLMs). While LVLMs have achieved remarkable performance across diverse multimodal tasks, their high computational and memory demands hinder practical deployment and scalability. Although various acceleration techniques have been proposed, a lack of unified evaluation across different architectures, datasets, and metrics limits our understanding of their effectiveness and trade-offs. 

We concentrate on evaluating various mainstream acceleration methods classified into two categories: token compression and parameter compression. EffiVLM-Bench provides a unified framework for evaluating not only the absolute performance but also the generalization and loyalty capabilities of these methods, while further exploring the Pareto-optimal trade-offs between performance and efficiency.  

  

# 📌 News
- 2025.05.18 EffiVLM-Bench is accepted to **ACL 2025**!
- Exciting updates on the way: new compression methods and more supported models are coming soon!


# 🖥️ Installation

## Create a new conda environment and install the basic dependencies
    ```bash
    conda create -n mllm-efficiency python=3.10
    conda activate mllm-efficiency
    pip install -r requirements.txt
    pip install ninja
    pip install omegaconf
    pip install flash-attention-softmax-n
    conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
    conda install nvidia/label/cuda-12.1.1::cuda-nvcc
    ```

## Change the env path 
    ```bash
    mkdir -p $CONDA_PREFIX/etc/conda/activate.d
    mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
    ```
    Create a new file in the activate.d directory and add the following content:
    ```bash
    #!/bin/bash
    export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
    ``` 
    Create a new file in the deactivate.d directory and add the following content:
    ```bash
    #!/bin/bash
    unset CUDA_HOME
    ```

## Install the flash-attn
    ```bash
    conda activate mllm-efficiency
    echo $CUDA_HOME
    which nvcc
    pip install flash-attn --no-build-isolation
    ```
## use lmms-eval
    ```bash
    cd lmms-eval
    pip install -e .
    cd ../llava/
    pip install -e .
    pip install numpy==2.2.0
    ```

## use qwen2_vl for develop
    ```bash
    cd qwen2vl
    pip install -e .
    pip install qwen-vl-utils
    ```

# Run path settings
Before running the script, you need to set the environment variables to ensure that the module is imported normally.

    ```bash
    export CONDA_DEFAULT_ENV="mllm-efficiency"
    export PATH="/your anaconda path /envs/mllm-efficiency/bin:$PATH"
    export PYTHONPATH="/your project path/EffiVLM-Bench:/your project path/EffiVLM-Bench/lmms-eval"
    ```

# 🚀 Quick Start

## Case Inference with predict.py

This section guides you on how to use the predict.py script for inference and testing various KV cache compression and token prune methods.The primary script for conducting inference tests is located at test/predict.py. 

### Supported Models and Methods:

You can test various **KV cache compression and token prune methods** on the following models:

- `llava-onevision-qwen2-7b-ov`
- `Qwen2-VL-7B-Instruct`
- `InternVL2_5-38B`

Additionally,  **KV cache methods** are supported for the following model:
- `InternVL2_5-4B`

Usage
To run the script, use the following command structure:

```bash
python test/predict.py [arguments]
```
# 🚀 Visualization (New Features)

We have introduced visualization tools powered by **Streamlit** to help users interact with the models and understand the internal acceleration mechanisms.

To use these features, please ensure you have installed streamlit:
```bash
pip install streamlit
```
1. Interactive Demo Showcase
This tool allows users to upload images and interact with the LVLMs directly to test various acceleration methods.

File: test/streamlit_predict.py

Function: Provides a user-friendly web interface for general model inference and result demonstration.

Run:
```bash
streamlit run test/streamlit_predict.py
```
2. Inference Framework Visualization
This tool offers a deeper look into the inference process, visualizing how different components and acceleration strategies work within the framework.

File: streamlit_app.py

Function: Visualizes the inference framework's internal status and metrics.

Run:
```bash
streamlit run streamlit_app.py
```
### Arguments
Below are the necessary command-line arguments to configure the inference process:

-----
  * `--image_path`: `str`, Path to the input image.
  * `--question`: `str`, The question to ask the model.
  * `--pretrained`: `str`, Path or identifier for the pretrained model.
  * `--model_name`: `str`, choices: `['llava-onevision-qwen2', 'qwen2-vl', 'internvl2_5']`. Specify the model name.
  * `--method`: `str`, choices: `['random', 'streamingllm', 'h2o', 'snapkv', 'look-m', 'vl-cache', 'pyramidkv', 'fastv', 'visionzip', 'prumerge+']`. The KV cache compression or token prune method to use.
  * `--merge`: `bool`, default: `True`. Merge switch for the `look-m` KV cache method.
  * `--head_adaptive`: `bool`, default: `True`. Enables head-adaptive strategy for `h2o`, `snapkv`, and `pyramidkv` methods.
  * `--pooling`: `str`, default: `avgpool`. Pooling strategy for `snapkv` and `pyramidkv` methods.
  * `--layer_adaptive`: `bool`, default: `True`. Enables layer-adaptive strategy for the `vl-cache` method.
  * `--vlcache_different_window_per_layer`: `bool`, default: `False`. Enables different window sizes per layer for the `vl-cache` method.
  * `--budgets`: `float`, default: `0.4`. Budget for KV cache compression and token prune methods.

-----



## Use lmms-eval to eval on various of benchmarks.

We use lmms-eval to evaluate various benchmarks. For examples of startup scripts, please refer to the run_example.sh file. You only need to replace your own paths and related module names and parameter names accordingly.

```bash
./run_example.sh
```


# Acknowledgement
Special thanks to the original EffiVLM-Bench team for their outstanding research and open-source contribution, which serves as the foundation for this project.
Thanks [KVCache-Factory](https://github.com/Zefan-Cai/KVCache-Factory.git) , [ECoFLaP](https://github.com/ylsung/ECoFLaP.git) , [Wanda](https://github.com/locuslab/wanda.git), [SparseGPT](https://github.com/IST-DASLab/sparsegpt.git) , [FastV](https://github.com/pkunlp-icler/FastV.git) , [VisionZip](https://github.com/dvlab-research/VisionZip.git) , [PruMerge](https://github.com/42Shawn/LLaVA-PruMerge.git) for providing open-source code to support the expansion of this project. 

# Citation
```

@misc{wang2025effivlmbenchcomprehensivebenchmarkevaluating,
      title={EffiVLM-BENCH: A Comprehensive Benchmark for Evaluating Training-Free Acceleration in Large Vision-Language Models}, 
      author={Zekun Wang and Minghua Ma and Zexin Wang and Rongchuan Mu and Liping Shan and Ming Liu and Bing Qin},
      year={2025},
      eprint={2506.00479},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.00479}, 
    }

```
