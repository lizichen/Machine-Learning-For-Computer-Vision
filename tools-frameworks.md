# ML Frameworks Installation Walk-Thru

### Machine Learning Framework Components:
1. Tensor Object张量对象
2. Operations on the Tensor Object 对该张量对象进行的各种运算
3. Computation Graph and Optimizations 计算图和优化
4. Auto-differentiation Tool/Function 自动微分工具
5. BLAS / cuBLAS and cuDNN extensions 扩展组件

### Torch
- Install Torch on AWS EC2 Instance or Ubuntu 16.04 LTS
    * Installation Tutorial: http://torch.ch/docs/getting-started.html
      
      [Optional] Install Git  
      ```sh
      $ sudo apt-get update  
      $ sudo apt-get install git  
      ```
      Run commands one by one in terminal: 
      ```sh
      $ git clone https://github.com/torch/distro.git ~/torch --recursive  
      $ cd ~/torch; bash install-deps;  
      $ ./install.sh  
      ```
      On Linux with bash:
      ```sh
      $ source ~/.bashrc  
      ```
      Install image and torchnet packages:
      ```sh
      $ luarocks install image  
      $ luarocks install torchnet  
      ```

### PyTorch

### Anaconda
- Install  
- Package Management

### Tensorflow
- Install

### Caffe

### Numpy Stack in Python
- **Numpy**    
- **Pandas**    
- **Matplotlib**    
- **Scipy**    

### Sci-kit Learn
- [Youtube] Data School - Machine Learning in Python with Scikit-Learn

### Data Processing Frameworks 数据处理框架：
- Map / Reduce + Hadoop——分布式存储和处理系统
- M / R——处理大量数据的范式
- Pig，Hive，Cascalog——在Map / Reduce 上的框架
- Spark——数据处理和训练的全栈解决方案（full stack solution）
- Google Cloud Dataflow

### GPUs and Cloud Servers
- Install **NVidia GTX 1080 GPU** on *Ubuntu 16.04*
    1. On BIOS, **disable 'Secured Boot State'**. (This step is optional, Ubuntu will help to disable it in later steps)
    2. In Ubuntu 16.04 LTS, download **CUDA 8.0** (latest as of *Jan 2016*) and install:
    ```bash
    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.44-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu1604_8.0.44-1_amd64.deb
    sudo apt-get update
    sudo apt-get install cuda
    ```
    3. Modify **PATH** and **LD_LIBRARY_PATH**:
    ```bash
    export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    ```
    4. Add the above to the end of .bashrc file.
    5. Register account on https://developer.nvidia.com/cudnn and download latest cuDNN. As of *Jan 2016*, use **cuDNN v5.1** (August 10, 2016), for CUDA 8.0 RC - cuDNN v5.1 Library for Linux. 
    6. Uncompress and copy the cuDNN files into the CUDA directory. Assuming the CUDA toolkit is installed in **/usr/local/cuda**, run the following commands:
    ```bash
    tar xvzf cudnn-8.0-linux-x64-v5.1.tgz
    sudo cp cuda/include/cudnn.h /usr/local/cuda/include
    sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
    sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
    ```
    7. Run command ```nvidia-smi ``` to see details about the card.
    8. Run command ```nvidia-settings``` to see more details.. 
- Multi-GPU on cutorch: https://github.com/torch/cutorch/issues/42
- AWS P2 Instance GPU Cuda Installation Guide: 
