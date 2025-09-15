# Petit

Petit provides optimized FP16/BF16 x FP4 GPU kernels specifically designed for AMD GPUs. It enables efficient execution of NVFP4 quantized models on GPUs that lack native FP4 arithmetic capabilities. This makes Petit particularly well-suited for serving high-quality NVFP4 models on standard GPUs while achieving ~3.3x memory savings. For example, a server with 8x AMD MI300x GPUs running [sglang v0.4.9.post2](https://github.com/sgl-project/sglang) can serve the [Llama-3.3-70B-Instruct](meta-llama/Llama-3.3-70B-Instruct) / [Llama-3.3-70B-Instruct-FP4](https://huggingface.co/nvidia/Llama-3.3-70B-Instruct-FP4) model with a MMLU score of 82.15 and 80.79 respectively.

## Requirements

* AMD CDNA2 / CDNA3 GPUs (AMD MI2xx / MI3xx series)
* ROCm 6.2 or later
* PyTorch 2.5 or later

## Installation and Usages

You can install Petit directly using pip:

```bash
$ CMAKE_ARGS='-DCMAKE_PREFIX_PATH=/opt/rocm;/usr/local/lib/python3.12/dist-packages/torch' pip install .
```

You need to specify `CMAKE_PREFIX_PATH` in `CMAKE_ARGS` so that cmake can detect the ROCm or PyTorch.

Petit provides python APIs for matrix multiplications that are intended to be integrated with inference frameworks such as [SGLang](https://github.com/sgl-project/sglang) and [vLLM](https://github.com/vllm-project/vllm.git). It also provides C++ bindings to enable integrations with frameworks like [llama.cpp](https://github.com/ggml-org/llama.cpp.git). 

## Techniques and Evaluations

Similar to [Marlin](https://github.com/IST-DASLab/marlin.git), Petit performs offline weight shuffling to enable efficient GPU dequantization. To achieve optimal performance, Petit utilizes ranged buffer loads and vector instructions specifically designed for CDNA2 and CDNA3 architectures. These optimizations are based on the assumptions that scales remain positive and quantized weights contain no negative zeros. For detailed information about these optimizations, please refer to the documentation available [here](https://www.causalflow.ai/blogs/2025-08-optimizing-fp4-mixed-precision-inference-on-amd-gpus).

Petit is optimized for the real-world use cases where the LLM engines perform inferences with small batches. For example, Petit is 1.2x-2.2x faster compared to [hipBLASLt](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest) when performing BF16 matrix multiplications when batch size less than 16. For larger batches where the performance is bound by the available computational powers, Petit performs within 70% of the hand-optimized hipBLASLt library.

## Known Issues

Similar to Marlin, Petit shuffles the data offline to minimize the work performed on the GPU side. It requires all scales are positive which matches the output of the [ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer.git) quantizier. 

The MFMA instructions on AMD MI2xx GPUs flush input and output denormal values to zero, which can potentially impact [numeric accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices). Petit implements corrective measures for the AMD MI2xx GPUs which have ~10% overheads. 

Compared to NVIDIA architectures, CDNA architectures are significantly more sensitive to kernel hyperparameters like the shapes in shared memory. We strongly recommend running auto-tuning to achieve optimal performance. The repository provides benchmarking tool to facilitate auto tunings.

## Contacts and Contributions

We thank AMD and [InnoMatrix](https://innomatrix.ai) for their generous support of providing access of the GPUs to make this project possible. Neither organization is involved in the development of the project.

Petit is a very young project and we are still working on implementing various optimizations.  Please contact haohui@causalflow.ai for questions and supports. Contributions are welcome.

