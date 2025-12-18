#include "matmul.h"
#include "causalflow/petit/config.h"
#include <map>
#include <string>

namespace causalflow::petit::benchmark::matmul {

std::unique_ptr<MatmulFactory> CreateMatmulFactoryPetitBackend();
std::unique_ptr<MatmulFactory> CreateMatmulFactoryHipBLASLtBackend();

static const std::map<std::string,
                      std::function<std::unique_ptr<MatmulFactory>()>>
    kBackends = {
#ifdef WITH_ROCM
        {"hipblaslt", CreateMatmulFactoryHipBLASLtBackend},
        {"petit", CreateMatmulFactoryPetitBackend},
#endif
};

std::unique_ptr<MatmulFactory>
MatmulFactory::Create(const std::string &backend) {
    auto it = kBackends.find(backend);
    return it == kBackends.end() ? nullptr : it->second();
}

} // namespace causalflow::petit::benchmark::matmul