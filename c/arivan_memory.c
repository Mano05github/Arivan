#include "arivan_memory.h"

#include <limits.h>
#include <string.h>

#define MIB ((uint64_t)1024 * 1024)
#define GIB ((uint64_t)1024 * 1024 * 1024)

static const ArivanMemoryProfile PROFILES[] = {
    {"8gb", 8 * GIB, 5376 * MIB, 384 * MIB, 512 * MIB,
     2048, 128, 1, 8, 1, 1},
    {"16gb", 16 * GIB, 12 * GIB, 1024 * MIB, 2048 * MIB,
     8192, 256, 1, 32, 1, 1},
    {"32gb", 32 * GIB, 26 * GIB, 4096 * MIB, 8192 * MIB,
     32768, 512, 2, 64, 1, 0},
};

int arivan_memory_profile(const char *name, ArivanMemoryProfile *out) {
    size_t i;
    if (!name || !out) return -1;
    for (i = 0; i < sizeof(PROFILES) / sizeof(PROFILES[0]); ++i) {
        if (strcmp(name, PROFILES[i].name) == 0) {
            *out = PROFILES[i];
            return 0;
        }
    }
    return -1;
}

void arivan_memory_budget_init(ArivanMemoryBudget *budget, uint64_t limit_bytes) {
    if (!budget) return;
    memset(budget, 0, sizeof(*budget));
    budget->limit_bytes = limit_bytes;
    budget->phase = ARIVAN_PHASE_LOAD;
}

void arivan_memory_budget_set_phase(ArivanMemoryBudget *budget, ArivanMemoryPhase phase) {
    if (!budget) return;
    budget->phase = phase;
}

int arivan_memory_reserve(ArivanMemoryBudget *budget, ArivanMemoryKind kind,
                          uint64_t bytes) {
    uint64_t next;
    if (!budget || kind < 0 || kind >= ARIVAN_MEM_KIND_COUNT) return -1;
    if (UINT64_MAX - budget->current_bytes < bytes) return -1;
    next = budget->current_bytes + bytes;
    if (next > budget->limit_bytes) return 1;
    if (UINT64_MAX - budget->by_kind[kind] < bytes) return -1;
    budget->by_kind[kind] += bytes;
    budget->current_bytes = next;
    if (next > budget->peak_bytes) budget->peak_bytes = next;
    return 0;
}

int arivan_memory_release(ArivanMemoryBudget *budget, ArivanMemoryKind kind,
                          uint64_t bytes) {
    if (!budget || kind < 0 || kind >= ARIVAN_MEM_KIND_COUNT) return -1;
    if (bytes > budget->by_kind[kind] || bytes > budget->current_bytes) return -1;
    budget->by_kind[kind] -= bytes;
    budget->current_bytes -= bytes;
    return 0;
}

uint64_t arivan_memory_available(const ArivanMemoryBudget *budget) {
    if (!budget || budget->current_bytes >= budget->limit_bytes) return 0;
    return budget->limit_bytes - budget->current_bytes;
}

int arivan_memory_reserve_uniform_cache(ArivanMemoryBudget *budget,
                                        ArivanMemoryKind kind,
                                        uint64_t bytes_per_slot,
                                        uint64_t component_limit_bytes,
                                        uint32_t requested_slots,
                                        uint32_t minimum_slots,
                                        uint32_t *granted_slots) {
    uint64_t allowed;
    uint64_t capacity;
    uint32_t slots;
    int status;

    if (granted_slots) *granted_slots = 0;
    if (!budget || !granted_slots || bytes_per_slot == 0 ||
        requested_slots == 0 || minimum_slots > requested_slots ||
        kind < 0 || kind >= ARIVAN_MEM_KIND_COUNT)
        return -1;

    allowed = arivan_memory_available(budget);
    if (allowed > component_limit_bytes) allowed = component_limit_bytes;
    capacity = allowed / bytes_per_slot;
    if (capacity < minimum_slots) return 1;
    slots = capacity < requested_slots ? (uint32_t)capacity : requested_slots;

    status = arivan_memory_reserve(budget, kind,
                                   (uint64_t)slots * bytes_per_slot);
    if (status != 0) return status;
    *granted_slots = slots;
    return 0;
}

const char *arivan_memory_kind_name(ArivanMemoryKind kind) {
    static const char *const names[ARIVAN_MEM_KIND_COUNT] = {
        "permanent", "dense-window", "expert-staging", "expert-pinned",
        "expert-lru", "kv-state", "vision", "mtp", "workspace", "server"
    };
    if (kind < 0 || kind >= ARIVAN_MEM_KIND_COUNT) return "invalid";
    return names[kind];
}

const char *arivan_memory_phase_name(ArivanMemoryPhase phase) {
    static const char *const names[] = {"load", "vision", "prefill", "decode"};
    if (phase < ARIVAN_PHASE_LOAD || phase > ARIVAN_PHASE_DECODE) return "invalid";
    return names[phase];
}
