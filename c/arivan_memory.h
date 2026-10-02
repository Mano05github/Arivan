#ifndef ARIVAN_MEMORY_H
#define ARIVAN_MEMORY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum ArivanMemoryKind {
    ARIVAN_MEM_PERMANENT = 0,
    ARIVAN_MEM_DENSE_WINDOW,
    ARIVAN_MEM_EXPERT_STAGING,
    ARIVAN_MEM_EXPERT_PINNED,
    ARIVAN_MEM_EXPERT_LRU,
    ARIVAN_MEM_KV_STATE,
    ARIVAN_MEM_VISION,
    ARIVAN_MEM_MTP,
    ARIVAN_MEM_WORKSPACE,
    ARIVAN_MEM_SERVER,
    ARIVAN_MEM_KIND_COUNT
} ArivanMemoryKind;

typedef enum ArivanMemoryPhase {
    ARIVAN_PHASE_LOAD = 0,
    ARIVAN_PHASE_VISION,
    ARIVAN_PHASE_PREFILL,
    ARIVAN_PHASE_DECODE
} ArivanMemoryPhase;

typedef struct ArivanMemoryProfile {
    const char *name;
    uint64_t installed_bytes;
    uint64_t engine_limit_bytes;
    uint64_t pinned_cache_bytes;
    uint64_t expert_cache_bytes;
    uint32_t context_tokens;
    uint32_t vision_tokens;
    uint32_t kv_slots;
    uint32_t prefill_chunk;
    uint32_t mtp_depth;
    int experimental;
} ArivanMemoryProfile;

typedef struct ArivanMemoryBudget {
    uint64_t limit_bytes;
    uint64_t current_bytes;
    uint64_t peak_bytes;
    uint64_t by_kind[ARIVAN_MEM_KIND_COUNT];
    ArivanMemoryPhase phase;
} ArivanMemoryBudget;

int arivan_memory_profile(const char *name, ArivanMemoryProfile *out);
void arivan_memory_budget_init(ArivanMemoryBudget *budget, uint64_t limit_bytes);
void arivan_memory_budget_set_phase(ArivanMemoryBudget *budget, ArivanMemoryPhase phase);
int arivan_memory_reserve(ArivanMemoryBudget *budget, ArivanMemoryKind kind,
                          uint64_t bytes);
int arivan_memory_release(ArivanMemoryBudget *budget, ArivanMemoryKind kind,
                          uint64_t bytes);
uint64_t arivan_memory_available(const ArivanMemoryBudget *budget);
int arivan_memory_reserve_uniform_cache(ArivanMemoryBudget *budget,
                                        ArivanMemoryKind kind,
                                        uint64_t bytes_per_slot,
                                        uint64_t component_limit_bytes,
                                        uint32_t requested_slots,
                                        uint32_t minimum_slots,
                                        uint32_t *granted_slots);
const char *arivan_memory_kind_name(ArivanMemoryKind kind);
const char *arivan_memory_phase_name(ArivanMemoryPhase phase);

#ifdef __cplusplus
}
#endif

#endif
