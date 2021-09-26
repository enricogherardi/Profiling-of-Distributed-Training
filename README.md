# Profiling of Distributed Training on Marconi100

This repository contains my research-based Thesis Project for my Bachelor Degree at NVIDIA AI Technology Center (CINECA).
My goal was to implement, benchmark and optimize a Distributed Training of the Resnet50 on IMAGENET with PyTorch.
The performance evaluation starts with a single GPU in a single node and will be scaled on up to sixteen nodes.

## Experiment Environment

- Neural Network: ResNet50
- Dataset: IMAGENET
- Framework: PyTorch
- Accelerated Cluster: Marconi100

## Node Configuration

- Processors: 2x16 cores IBM POWER9 AC922 at 3.1 GHz
- Accelerators: 4 x NVIDIA Volta V100 GPUs, Nvlink 2.0, 16GB
- Cores: 32 cores/node
- RAM: 256 GB/node

(figura)

## Distributed DataParallel

The implementation of the Distributed Training is based on PyTorch. In particular, we used DistributedDataParallel (DDP) to split the model and data between different GPUs and to coordinate the training.
For DDP, the machine has one process per GPU, and each model
is controlled by each process. The GPUs can all be on the same node or across multiple
nodes. Only gradients are passed between the processes/GPUs.
During training, each process loads its own mini-batch from disk and passes it to its
GPU. Each GPU does its forward pass, then the gradients are all-reduced across the
GPUs. Gradients for each layer do not depend on previous layers, so the gradient all-reduce
is calculated concurrently with the backwards pass to further alleviate the networking
bottleneck. At the end of the backwards pass, every node has the averaged gradients,
ensuring that the model weights stay synchronized.

(figura)

## Performance Evaluation

The performance evaluation starts with a single GPU in a single node and will be scaled on up to sixteen nodes. The goal of the experiments was to find the best configuration in order to minimize the average epoch time.

### Multiple GPUs in a Single Node

All the experiments are done with 16 workers and 32 as batch-size

(imgs)

### Multiple GPUs in Multiple Nodes

The batch-size and the number of workers is the same as before

(imgs)

### Multiple Workers


## Conclusion

For a deep comprehension of the work and analysis of the performances I recommend you to read the Chapter 6 of my Thesis.