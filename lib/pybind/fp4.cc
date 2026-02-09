#include "gemm/rocm/quantization/fp4/gemm_fp4.h"
#include "gemm/rocm/quantization/gemm.h"
#include "pybind.h"

#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>

namespace causalflow::petit::pybind {

using namespace causalflow::petit::rocm::quantization;
using fp4::GemmFp4Fp16Grid;
using fp4::GemmGetSolutions;
using fp4::GemmMxFp4Fp16Grid;
using fp4::RepackNvFp4ToPetitFp4Scales;
using fp4::RepackNvFp4ToPetitFp4Weights;

static constexpr unsigned kLayoutM = 128;
static constexpr unsigned kLayoutN = 16;
static constexpr unsigned kPackFactor = 32 / 4;
static constexpr unsigned kMxRowGroupSize = 32;

namespace {

bool ShouldRequireHighPrecision(int dev) {
    // For gfx90a, we need to use high precision as the MFMA instructions
    // flush the inputs and output denormals.
    hipDeviceProp_t props;
    hipError_t error = hipGetDeviceProperties(&props, dev);
    if (error != hipSuccess) {
        AT_ERROR("Failed to get the device properties of device ", dev);
    }

    return (props.major * 10 + props.minor) <= 90;
}

} // namespace

torch::Tensor RepackNvFp4(torch::Tensor &b_q_weight, int64_t size_n,
                          int64_t size_k) {
    TORCH_CHECK(size_k % kLayoutM == 0, "size_k = ", size_k,
                " is not divisible by tile_k_size = ", kLayoutM);
    TORCH_CHECK(size_n % kLayoutN == 0, "size_n = ", size_n,
                " is not divisible by tile_n_size = ", kLayoutN);

    // Verify B
    TORCH_CHECK((size_k / kPackFactor) == b_q_weight.size(1),
                "Shape mismatch: b_q_weight.size(1) = ", b_q_weight.size(1),
                ", size_k = ", size_k, ", pack_factor = ", kPackFactor);
    TORCH_CHECK(b_q_weight.size(0) == size_n,
                "b_q_weight.size(0) = ", b_q_weight.size(0),
                " is not size_n = ", size_n);

    // Verify device and strides
    TORCH_CHECK(b_q_weight.device().is_cuda(), "b_q_weight is not on GPU");
    TORCH_CHECK(b_q_weight.is_contiguous(), "b_q_weight is not contiguous");
    TORCH_CHECK(b_q_weight.dtype() == at::kInt, "b_q_weight type is not kInt");

    // Alloc buffers
    auto options = torch::TensorOptions()
                       .dtype(b_q_weight.dtype())
                       .device(b_q_weight.device());
    torch::Tensor out = torch::empty(
        {size_n / kLayoutN, size_k * kLayoutN / kPackFactor}, options);

    // Get ptrs
    uint32_t const *b_q_weight_ptr =
        reinterpret_cast<uint32_t const *>(b_q_weight.data_ptr());
    uint32_t *out_ptr = reinterpret_cast<uint32_t *>(out.data_ptr());

    // Get dev info
    int dev = b_q_weight.get_device();
    hipStream_t stream = at::hip::getCurrentHIPStream(dev);

    RepackNvFp4ToPetitFp4Weights(out_ptr, b_q_weight_ptr, size_k, size_n,
                                 stream);

    return out;
}

torch::Tensor ProcessNvFp4Scales(torch::Tensor &scales, int64_t size_n,
                                 int64_t size_k) {
    static constexpr unsigned kGroupM = 2 * kLayoutM;
    TORCH_CHECK(size_k % kGroupM == 0, "size_k = ", size_k,
                " is not divisible by tile_k_size = ", kGroupM);
    TORCH_CHECK(size_n % kLayoutN == 0, "size_n = ", size_n,
                " is not divisible by tile_n_size = ", kLayoutN);

    // Verify group size
    unsigned group_size = size_k / scales.size(1);
    if (group_size != 16) {
        AT_ERROR("Only groupsize = 16 is supported.");
    }

    // Verify scales
    TORCH_CHECK(scales.size(0) == size_n, "scales.size(0) = ", scales.size(0),
                " is not size_n = ", size_n);

    // Verify device and strides
    TORCH_CHECK(scales.device().is_cuda(), "scales is not on GPU");
    TORCH_CHECK(scales.is_contiguous(), "scales is not contiguous");
    TORCH_CHECK(scales.dtype() == at::kFloat8_e4m3fn,
                "scales type is not float8_e4m3fn");

    // Alloc buffers
    auto options =
        torch::TensorOptions().dtype(scales.dtype()).device(scales.device());
    torch::Tensor out = torch::empty({scales.size(0), scales.size(1)}, options);

    // Get ptrs
    unsigned const *scales_ptr =
        reinterpret_cast<unsigned const *>(scales.data_ptr());
    unsigned *out_ptr = reinterpret_cast<unsigned *>(out.data_ptr());

    // Get dev info
    int dev = scales.get_device();
    hipStream_t stream = at::hip::getCurrentHIPStream(dev);

    RepackNvFp4ToPetitFp4Scales(out_ptr, scales_ptr, size_k, size_n, stream);

    return out;
}

torch::Tensor ProcessMxFp4Scales(torch::Tensor &scales, int64_t size_n,
                                 int64_t size_k) {
    static constexpr unsigned kScaleLayoutN = 32;
    static constexpr unsigned kGroupM = 2 * kLayoutM;
    TORCH_CHECK(size_k % kGroupM == 0, "size_k = ", size_k,
                " is not divisible by tile_k_size = ", kGroupM);
    TORCH_CHECK(size_n % kLayoutN == 0, "size_n = ", size_n,
                " is not divisible by tile_n_size = ", kLayoutN);

    unsigned group_size = size_k / scales.size(1);
    if (group_size != 32) {
        AT_ERROR("Only groupsize = 32 is supported.");
    }

    TORCH_CHECK(scales.size(0) == size_n, "scales.size(0) = ", scales.size(0),
                " is not size_n = ", size_n);

    TORCH_CHECK(scales.device().is_cuda(), "scales is not on GPU");
    TORCH_CHECK(scales.is_contiguous(), "scales is not contiguous");
    TORCH_CHECK(scales.dtype() == at::kByte, "scales type is not uint8");

    auto options =
        torch::TensorOptions().dtype(scales.dtype()).device(scales.device());
    torch::Tensor out = torch::empty(
        {size_n / kScaleLayoutN, size_k * kScaleLayoutN / kMxRowGroupSize},
        options);

    unsigned const *scales_ptr =
        reinterpret_cast<unsigned const *>(scales.data_ptr());
    unsigned *out_ptr = reinterpret_cast<unsigned *>(out.data_ptr());

    int dev = scales.get_device();
    hipStream_t stream = at::hip::getCurrentHIPStream(dev);

    fp4::RepackMxFp4ToPetitFp4Scales(out_ptr, scales_ptr, size_k, size_n,
                                     stream);

    return out;
}

torch::Tensor MulNvFp4A16(const torch::Tensor &A, const torch::Tensor &B,
                          const torch::Tensor &s,
                          const torch::Tensor &global_scale, int64_t size_m,
                          int64_t size_n, int64_t size_k, int64_t solution_id) {
    int groupsize = size_k / s.size(1);
    if (groupsize != 16) {
        AT_ERROR("Only groupsize = 16 is supported. size_k = ", size_k,
                 ", s.size(1) = ", s.size(1));
    }
    if (A.dtype() != torch::kBFloat16 && A.dtype() != torch::kFloat16) {
        AT_ERROR("A must be bfloat16 or float16.");
    }
    DataType a_type = A.dtype() == torch::kBFloat16 ? DataType::kDataTypeBf16
                                                    : DataType::kDataTypeFp16;

    int dev = A.get_device();
    torch::Tensor c;
    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    c = torch::empty({size_m, size_n}, options);

    PetitSolutionHints hints;
    hints.a_type = a_type;
    hints.b_type = DataType::kDataTypeFp4e2m1;
    hints.c_type = a_type;
    hints.require_high_precision = false;

    if (solution_id < 0) {
        hints.require_high_precision = ShouldRequireHighPrecision(dev);
    }

    int err = GemmFp4Fp16Grid(
        reinterpret_cast<unsigned *>(c.data_ptr()),
        reinterpret_cast<const unsigned *>(A.data_ptr()),
        reinterpret_cast<const unsigned *>(B.data_ptr()),
        reinterpret_cast<const unsigned *>(s.data_ptr()),
        reinterpret_cast<const float *>(global_scale.data_ptr()), size_m,
        size_n, size_k, hints, solution_id, at::hip::getCurrentHIPStream(dev));

    if (err == kErrorProblemShape) {
        AT_ERROR("Incompatible problem shape (m=", size_m, ", n=", size_n,
                 ", k=", size_k, ")");
    } else if (err == kErrorKernelShape) {
        AT_ERROR("No kernel implementation for solution_id=", solution_id, ".");
    }

    return c;
}

torch::Tensor MulMxFp4A16(const torch::Tensor &A, const torch::Tensor &B,
                          const torch::Tensor &s,
                          const torch::Tensor &global_scale, int64_t size_m,
                          int64_t size_n, int64_t size_k, int64_t solution_id) {
    TORCH_CHECK(B.size(0) == size_n / kLayoutN, "B.size(0) = ", B.size(0),
                " is not size_n / 16 = ", size_n / kLayoutN);
    TORCH_CHECK(B.size(1) == size_k * kLayoutN / kPackFactor,
                "B.size(1) = ", B.size(1),
                " is not packed size = ", size_k * kLayoutN / kPackFactor);
    TORCH_CHECK(s.size(0) == size_n / 32, "s.size(0) = ", s.size(0),
                " is not size_n / 32 = ", size_n / 32);
    TORCH_CHECK(s.size(1) == size_k, "s.size(1) = ", s.size(1),
                " is not size_k = ", size_k);
    if (A.dtype() != torch::kBFloat16 && A.dtype() != torch::kFloat16) {
        AT_ERROR("A must be bfloat16 or float16.");
    }
    DataType a_type = A.dtype() == torch::kBFloat16 ? DataType::kDataTypeBf16
                                                    : DataType::kDataTypeFp16;

    int dev = A.get_device();
    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    torch::Tensor c = torch::empty({size_m, size_n}, options);

    PetitSolutionHints hints;
    hints.a_type = a_type;
    hints.b_type = DataType::kDataTypeMxFp4e2m1;
    hints.c_type = a_type;
    hints.require_high_precision = false;

    if (solution_id < 0) {
        hints.require_high_precision = ShouldRequireHighPrecision(dev);
    }

    int err = GemmMxFp4Fp16Grid(
        reinterpret_cast<unsigned *>(c.data_ptr()),
        reinterpret_cast<const unsigned *>(A.data_ptr()),
        reinterpret_cast<const unsigned *>(B.data_ptr()),
        reinterpret_cast<const unsigned *>(s.data_ptr()),
        reinterpret_cast<const float *>(global_scale.data_ptr()), size_m,
        size_n, size_k, hints, solution_id, at::hip::getCurrentHIPStream(dev));

    if (err == kErrorProblemShape) {
        AT_ERROR("Incompatible problem shape (m=", size_m, ", n=", size_n,
                 ", k=", size_k, ")");
    } else if (err == kErrorKernelShape) {
        AT_ERROR("No kernel implementation for solution_id=", solution_id, ".");
    }

    return c;
}

py::list GetNvFp4Solutions(const PetitSolutionHints &hints, int64_t size_m,
                           int64_t size_n, int64_t size_k) {
    unsigned n_solutions = 0;
    int err =
        GemmGetSolutions(hints, size_m, size_n, size_k, nullptr, &n_solutions);
    if (err != 0) {
        AT_ERROR("Failed to get solutions: ", err);
    }

    std::vector<SolutionId> solutions(n_solutions);
    err = GemmGetSolutions(hints, size_m, size_n, size_k, solutions.data(),
                           &n_solutions);
    if (err != 0) {
        AT_ERROR("Failed to get solutions: ", err);
    }

    py::list ret;
    for (unsigned i = 0; i < n_solutions; ++i) {
        ret.append(solutions[i].Repr());
    }
    return ret;
}

} // namespace causalflow::petit::pybind
