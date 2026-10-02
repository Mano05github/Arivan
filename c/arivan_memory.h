#ifndef ARIVAN_MEMORY_H
#define ARIVAN_MEMORY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum ArivanMemoryKind {
    ARIVAN_MEM_PERMANENT = 0,
    ARIVAN_MEM_RESIDENT_WEIGHTS,
    ARIVAN_MEM_EMBEDDING_WINDOW,
    ARIVAN_MEM_OUTPUT_HEAD_WINDOW,
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
    ARIVAN_PHASE_DECODE,
    ARIVAN_PHASE_COUNT
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
    uint64_t peak_by_phase[ARIVAN_PHASE_COUNT];
    ArivanMemoryPhase phase;
} ArivanMemoryBudget;

typedef struct ArivanExpertPlan {
    uint32_t cache_slots_per_layer;
    uint32_t staging_slots;
    uint64_t cache_bytes;
    uint64_t staging_bytes;
} ArivanExpertPlan;

int arivan_memory_profile(const char *name, ArivanMemoryProfile *out);
void arivan_memory_budget_init(ArivanMemoryBudget *budget, uint64_t limit_bytes);
void arivan_memory_budget_set_phase(ArivanMemoryBudget *budget, ArivanMemoryPhase phase);
int arivan_memory_reserve(ArivanMemoryBudget *budget, ArivanMemoryKind kind,
                          uint64_t bytes);
int arivan_memory_release(ArivanMemoryBudget *budget, ArivanMemoryKind kind,
                          uint64_t bytes);
int arivan_memory_reclassify(ArivanMemoryBudget *budget,
                             ArivanMemoryKind from,
                             ArivanMemoryKind to,
                             uint64_t bytes);
uint64_t arivan_memory_available(const ArivanMemoryBudget *budget);
int arivan_memory_reserve_uniform_cache(ArivanMemoryBudget *budget,
                                        ArivanMemoryKind kind,
                                        uint64_t bytes_per_slot,
                                        uint64_t component_limit_bytes,
                                        uint32_t requested_slots,
                                        uint32_t minimum_slots,
                                        uint32_t *granted_slots);
int arivan_memory_plan_experts(ArivanMemoryBudget *budget,
                               uint64_t bytes_per_expert,
                               uint32_t sparse_layers,
                               uint64_t cache_limit_bytes,
                               uint32_t requested_cache_slots_per_layer,
                               uint32_t requested_staging_slots,
                               ArivanExpertPlan *plan);
const char *arivan_memory_kind_name(ArivanMemoryKind kind);
const char *arivan_memory_phase_name(ArivanMemoryPhase phase);

#ifdef __cplusplus
}
#endif

#endif
