# Petit

Petit provides optimized AMD GPU kernels for dense matrix multiplication,
Mixture-of-Experts (MoE), and experimental single-node MegaMoE workloads. Its
FP16/BF16 × FP4 kernels support NVFP4- and MXFP4-quantized models on GPUs with
or without native FP4 arithmetic.

## Features

- Dense matrix multiplication with NVFP4 and MXFP4 weights
- Fused MoE kernels for BF16 and MXFP4 workloads
- Experimental single-node MegaMoE kernels with BF16 and MXFP4 activations

## Requirements

- AMD CDNA2, CDNA3, or CDNA4 GPUs (MI200, MI300, or MI350 series)
- ROCm 6.2 or later
- PyTorch 2.5 or later

MegaMoE support currently requires CDNA4 (`gfx950`).

## Installation and usage

Install Petit from the repository with pip:

```bash
CMAKE_ARGS="-DCMAKE_PREFIX_PATH=/opt/rocm;$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
  pip install .
```

Set `CMAKE_PREFIX_PATH` through `CMAKE_ARGS` so that CMake can locate ROCm and
PyTorch.

Petit exposes Python APIs for matrix multiplication and MoE kernels for
integration with inference frameworks such as
[SGLang](https://github.com/sgl-project/sglang) and
[vLLM](https://github.com/vllm-project/vllm). It also provides C++ bindings for
integrations with frameworks such as [llama.cpp](https://github.com/ggml-org/llama.cpp).

## Techniques and performance

Like [Marlin](https://github.com/IST-DASLab/marlin), Petit shuffles weights
offline to make GPU dequantization more efficient. It also uses ranged buffer
loads and vector instructions designed for AMD CDNA architectures. See
[Optimizing FP4 Mixed-Precision Inference on AMD GPUs](https://www.causalflow.ai/blogs/2025-08-optimizing-fp4-mixed-precision-inference-on-amd-gpus)
for details.

FP4 quantization provides approximately 3.3× memory savings. For example, a
server with eight AMD MI300X GPUs running
[SGLang v0.4.9.post2](https://github.com/sgl-project/sglang) can serve both
[Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)
and
[Llama-3.3-70B-Instruct-FP4](https://huggingface.co/nvidia/Llama-3.3-70B-Instruct-FP4),
which achieve MMLU scores of 82.15 and 80.79, respectively.

Petit targets small-batch LLM inference. For BF16 matrix multiplication with
batch sizes below 16, Petit is 1.2–2.2× faster than
[hipBLASLt](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest). At larger
batch sizes, where performance becomes compute-bound, Petit reaches 70% of the
performance of the hand-optimized hipBLASLt library.

## Known limitations

- Petit's offline data transformation assumes that scales are positive and
  quantized weights contain no negative zeros. This is compatible with output
  from the
  [TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer).

- MFMA instructions on AMD MI200-series GPUs flush input and output denormal
  values to zero, which can affect
  [numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices).
  Petit's corrective measures add approximately 10% overhead on these GPUs.

- AMD CDNA architectures are sensitive to kernel hyperparameters such as
  shared-memory tile shapes. Run the included benchmarking tools to tune these
  parameters for optimal performance.

## Contact and contributions

We thank AMD and [InnoMatrix](https://innomatrix.ai) for generously providing
access to the GPUs that made this project possible. Neither organization is
involved in the development of Petit.

Petit is a young project, and many optimizations are still in progress.
Questions, feedback, and contributions are welcome. Contact
[haohui@causalflow.ai](mailto:haohui@causalflow.ai) for more information.
