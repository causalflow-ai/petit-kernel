#pragma once

#include "gemm/rocm/amd_intrinsics.cuh"

#include <cstdint>

namespace causalflow::petit::rocm::moe {

__device__ inline void agent_fence_release() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
}

__device__ inline void agent_fence_acquire() {
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
}

__device__ inline void system_fence_release() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
}

__device__ inline void system_fence_acquire() {
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
}

__device__ inline void wave_barrier() {
    __builtin_amdgcn_wave_barrier();
}

__device__ __forceinline__ void buffer_wbl2_sc0_sc1() {
    asm volatile("buffer_wbl2 sc0 sc1\n" ::: "memory");
}

// Scoped buffer stores already place data at the visibility point selected by
// SC[1:0].  Before publishing a signal for those stores, only their VMEM
// completion is required; writing back unrelated dirty L2 lines would be both
// broader and more expensive.  The compiler barriers keep protected stores
// before the wait and the subsequent signal after it.
__device__ __forceinline__ void complete_scoped_vmem() {
    asm volatile("" ::: "memory");
    amdgcn_s_waitcnt<0>();
    asm volatile("" ::: "memory");
}

template <class Workspace>
__device__ __forceinline__ void
store_xgpu_epoch_relaxed(const Workspace &workspace, unsigned signal_offset,
                         unsigned epoch) {
    workspace.br_.template StoreU32<BufferResource::kSC0Bit |
                                    BufferResource::kSC1Bit>(
        signal_offset, 0, epoch);
}

template <class Workspace>
__device__ __forceinline__ void
store_xgpu_epoch_release(const Workspace &workspace, unsigned signal_offset,
                         unsigned epoch) {
    system_fence_release();
    store_xgpu_epoch_relaxed(workspace, signal_offset, epoch);
}

template <class Workspace>
__device__ __forceinline__ void
wait_xgpu_signal(const Workspace &workspace, unsigned signal_offset,
                 std::int32_t target) {
    while (static_cast<std::int32_t>(workspace.br_.template LoadU32<
               BufferResource::kAtomicScopeSystem>(signal_offset, 0)) !=
           target) {
    }
}

// Poll a system-scope atomic with relaxed ordering, then let the caller perform
// one acquire fence after the readiness condition has been observed. Keeping
// acquire out of the loop is important on gfx950, where an acquire load
// otherwise invalidates the vector caches on every unsuccessful poll.
template <class Workspace>
__device__ __forceinline__ void
wait_xgpu_signal_relaxed(const Workspace &workspace, unsigned signal_offset,
                         std::int32_t target) {
    while (true) {
        // Raw-buffer loads are non-atomic LLVM memory operations. Keep each
        // poll in the loop even though the writer is another GPU.
        asm volatile("" ::: "memory");
        const auto observed = static_cast<std::int32_t>(
            workspace.br_.template LoadU32<BufferResource::kSC0Bit |
                                            BufferResource::kSC1Bit>(
                signal_offset, 0));
        if (observed == target)
            return;
    }
}

template <class Workspace>
__device__ __forceinline__ void
wait_xgpu_epoch_relaxed(const Workspace &workspace, unsigned signal_offset,
                        unsigned expected) {
    while (true) {
        asm volatile("" ::: "memory");
        const unsigned observed = workspace.br_.template LoadU32<
            BufferResource::kSC0Bit | BufferResource::kSC1Bit>(signal_offset,
                                                               0);
        if (static_cast<std::int32_t>(observed - expected) >= 0)
            return;
    }
}

__host__ __device__ constexpr bool xgpu_epoch_reached(unsigned observed,
                                                       unsigned expected) {
    return static_cast<std::int32_t>(observed - expected) >= 0;
}

static_assert(xgpu_epoch_reached(0, 0));
static_assert(xgpu_epoch_reached(0, 0xffffffffu));
static_assert(!xgpu_epoch_reached(0xffffffffu, 0));

template <class Workspace>
__device__ __forceinline__ void
wait_xgpu_epoch(const Workspace &workspace, unsigned signal_offset,
                unsigned expected) {
    while (true) {
        const unsigned observed = workspace.br_.template LoadU32<
            BufferResource::kSC0Bit | BufferResource::kSC1Bit>(signal_offset,
                                                               0);
        if (xgpu_epoch_reached(observed, expected)) {
            return;
        }
    }
}

template <unsigned kNumSMs, unsigned kGridSyncIndex = 0,
          bool kAcquirePayload = true, bool kSystemScope = false,
          typename sync_scope_t, class Workspace>
__device__ __forceinline__ void
grid_sync(const Workspace &workspace, unsigned sm_idx, unsigned thread_idx,
          const sync_scope_t sync_scope) {
    if constexpr (kNumSMs == 1) {
        sync_scope();
        return;
    }

    static constexpr unsigned kFinishSumTag = 0x80000000u;
    sync_scope();
    if (thread_idx == 0) {
        const auto count_offset = workspace.GridSyncBarrierOffset() +
                                  kGridSyncIndex * sizeof(unsigned);
        // sm_idx is in [0, kNumSMs).  Form the SM0 contribution as a scalar
        // value so the compiler does not keep the sm_idx == 0 wave predicate
        // live across every grid barrier in the fused kernel.
        static_assert(kNumSMs <= 0x80000000u);
        const unsigned is_first_sm = (sm_idx - 1) >> 31;
        const unsigned arrival_delta =
            1 + is_first_sm * (kFinishSumTag - kNumSMs);
        if constexpr (kSystemScope)
            system_fence_release();
        else
            agent_fence_release();
        unsigned old_value;
        if constexpr (kSystemScope) {
            old_value = static_cast<unsigned>(
                workspace.br_.template AtomicAddI32<
                    BufferResource::kAtomicScopeSystem>(
                    count_offset, 0, static_cast<int>(arrival_delta)));
        } else {
            old_value = static_cast<unsigned>(
                workspace.br_.template AtomicAddI32<
                    BufferResource::kAtomicScopeAgent>(
                    count_offset, 0, static_cast<int>(arrival_delta)));
        }
        unsigned new_value;
        while (true) {
            // Poll coherently at both cache scopes.  This cannot observe the
            // preceding tag, and avoids invalidating the whole cache before
            // the first load and after every unsuccessful load.
            new_value = workspace.br_.template LoadU32<
                BufferResource::kSC0Bit | BufferResource::kSC1Bit>(
                count_offset, 0);
            if ((new_value ^ old_value) & kFinishSumTag) {
                break;
            }
            // Back off the polling wave so the final atomic arrivals can
            // make forward progress under a full 256-workgroup launch.
            asm volatile("s_sleep 1" ::: "memory");
        }
        if constexpr (kAcquirePayload) {
            if constexpr (kSystemScope)
                system_fence_acquire();
            else
                agent_fence_acquire();
        } else {
            // Preserve the compiler handoff for control-only barriers without
            // invalidating payload caches that have no downstream consumer.
            asm volatile("" ::: "memory");
        }
    }
    sync_scope();
}

template <unsigned kNumRanks, unsigned kNumSMs, unsigned kNumThreads,
          unsigned kGridSyncIndex, bool kAcquireProloguePayload = true,
          bool kAcquireEpiloguePayload = true, typename sync_scope_t,
          class Workspace>
__device__ __forceinline__ void
xgpu_barrier(const Workspace &workspace, unsigned sm_idx,
             unsigned thread_idx, const sync_scope_t &sync_scope,
             const bool &sync_prologue = true,
             const bool &sync_epilogue = true) {
    static_assert(kNumRanks <= kNumThreads, "Insufficient threads");

    // Grid sync before xGMI signaling.
    if (sync_prologue)
        grid_sync<kNumSMs, kGridSyncIndex, kAcquireProloguePayload>(
            workspace, sm_idx, thread_idx, sync_scope);

    if constexpr (kNumRanks == 1) {
        // With no cross-rank phase, one grid barrier supplies both sides of
        // the collective.
        if (sync_epilogue && !sync_prologue)
            grid_sync<kNumSMs, kGridSyncIndex, kAcquireEpiloguePayload>(
                workspace, sm_idx, thread_idx, sync_scope);
        return;
    }

    // Cross-rank barrier, only SM 0 participates.
    if (sm_idx == 0) {
        // Keep a status word per rank in the shared barrier prefix.  The
        // rank-local grid-sync region may be mapped at the same virtual
        // address on every device, so it cannot hold cross-rank state.
        // Only the first wave needs the phase state or participates in the
        // xGMI fan-out. The full first wave executes the system-release fence
        // once before its rank lanes issue the system-scope signal atomics.
        if (thread_idx < causalflow::petit::rocm::kWarpSize) {
            const auto counter_offset =
                Workspace::XGpuBarrierCounterOffset(workspace.Rank());
            unsigned status = 0;
            if (thread_idx == 0) {
                status = workspace.br_.template LoadU32<
                    BufferResource::kNone>(counter_offset, 0) & 3;
            }
            status = __shfl(status, 0);
            // The prologue grid arrival gathers every producer CTA.  Make
            // that agent-scoped dependency transitive before the rank lanes
            // publish completion to peer GPUs.  A cache writeback followed
            // by a monotonic system atomic is not itself a release sequence;
            // the explicit system-release edge is what orders remote payload
            // stores from the preceding kernel, matching the combine
            // handoff.
            system_fence_release();
            const unsigned signal_phase = status & 1;
            const unsigned signal_sign = status >> 1;
            const std::int32_t signal_delta = signal_sign ? -1 : 1;

            // Match the symmetric DeepGEMM fan-out: each rank lane signals
            // the corresponding rank, including itself.  Besides avoiding a
            // rank-dependent divergent lane, this gives every rank the same
            // completion target.
            if (thread_idx < kNumRanks) {
                workspace.br_.template AtomicAddI32<BufferResource::kAtomicScopeSystem>(
                    Workspace::XGpuBarrierSignalOffset(thread_idx,
                                                       signal_phase),
                    0, signal_delta);
            }
            wave_barrier();
            amdgcn_s_waitcnt<0, -1, 0>();
            if (thread_idx == 0) {
                workspace.br_.template StoreU32<BufferResource::kNone>(
                    counter_offset, 0, status + 1);
                const int target =
                    signal_sign ? 0 : static_cast<int>(kNumRanks);
                wait_xgpu_signal(
                    workspace,
                    Workspace::XGpuBarrierSignalOffset(workspace.Rank(),
                                                       signal_phase),
                    target);
                system_fence_acquire();
            }
        }
        sync_scope();
    }

    // Grid sync after xGPU completion.
    if (sync_epilogue)
        grid_sync<kNumSMs, kGridSyncIndex, kAcquireEpiloguePayload>(
            workspace, sm_idx, thread_idx, sync_scope);
}

// Two-phase wrappers let callers place rank-local publication work between
// the local grid arrival and the cross-rank handoff.  Config::XGpuSync selects
// one of these classes without introducing a runtime branch in device code.
template <class Config> struct LegacyXGpuSync {
    struct Ticket {};

    template <unsigned kPrologueGridSyncIndex, typename sync_scope_t,
              class Workspace>
    __device__ __forceinline__ static Ticket
    Begin(const Workspace &workspace, unsigned sm_idx, unsigned thread_idx,
          const sync_scope_t &sync_scope) {
        grid_sync<Config::kNumSMs, kPrologueGridSyncIndex,
                  /* kAcquirePayload */ false>(workspace, sm_idx, thread_idx,
                                               sync_scope);
        return {};
    }

    template <unsigned kEpilogueGridSyncIndex, typename sync_scope_t,
              class Workspace>
    __device__ __forceinline__ static void
    Finish(const Workspace &workspace, unsigned sm_idx, unsigned thread_idx,
           Ticket, const sync_scope_t &sync_scope) {
        xgpu_barrier<Config::kNumRanks, Config::kNumSMs, Config::kThreads,
                     kEpilogueGridSyncIndex,
                     /* kAcquireProloguePayload */ false,
                     /* kAcquireEpiloguePayload */ true>(
            workspace, sm_idx, thread_idx, sync_scope,
            /* sync_prologue */ false,
            /* sync_epilogue */ true);
    }
};

template <class Config> struct EpochXGpuSync {
    using Ticket = unsigned;

    template <unsigned kPrologueGridSyncIndex, typename sync_scope_t,
              class Workspace>
    __device__ __forceinline__ static Ticket
    Begin(const Workspace &workspace, unsigned sm_idx, unsigned thread_idx,
          const sync_scope_t &sync_scope) {
        unsigned next_epoch = 0;
        if (thread_idx == 0) {
            next_epoch = 1 + workspace.br_.template LoadU32<
                                 BufferResource::kNone>(
                                 Workspace::XGpuEpochCounterOffset(
                                     workspace.Rank()),
                                 0);
        }

        // The publishing wave must acquire payload released by every local
        // CTA before its system-release epoch store can publish that payload
        // transitively to peers.
        grid_sync<Config::kNumSMs, kPrologueGridSyncIndex,
                  /* kAcquirePayload */ true>(workspace, sm_idx, thread_idx,
                                              sync_scope);
        return next_epoch;
    }

    template <unsigned, typename sync_scope_t, class Workspace>
    __device__ __forceinline__ static void
    Finish(const Workspace &workspace, unsigned sm_idx, unsigned thread_idx,
           Ticket next_epoch, const sync_scope_t &sync_scope) {
        static_assert(Config::kNumRanks <=
                          causalflow::petit::rocm::kWarpSize,
                      "Epoch waiters must fit in one wave");

        if (thread_idx < causalflow::petit::rocm::kWarpSize) {
            next_epoch = __shfl(next_epoch, 0);

            if (sm_idx == 0) {
                if (thread_idx == 0) {
                    workspace.br_.template StoreU32<BufferResource::kNone>(
                        Workspace::XGpuEpochCounterOffset(workspace.Rank()),
                        0, next_epoch);
                }
                if (thread_idx < Config::kNumRanks) {
                    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
                    workspace.br_.template StoreU32<
                        BufferResource::kSC0Bit | BufferResource::kSC1Bit>(
                        Workspace::XGpuEpochSignalOffset(thread_idx,
                                                         workspace.Rank()),
                        0, next_epoch);
                }
                wave_barrier();
                amdgcn_s_waitcnt<0, -1, 0>();
            }

            // Every CTA waits directly so no epilogue grid barrier is needed.
            if (thread_idx < Config::kNumRanks) {
                wait_xgpu_epoch(
                    workspace,
                    Workspace::XGpuEpochSignalOffset(workspace.Rank(),
                                                     thread_idx),
                    next_epoch);
                system_fence_acquire();
            }
        }
        sync_scope();
    }
};

} // namespace causalflow::petit::rocm::moe
