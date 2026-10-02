#include "../arivan_memory.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#define MIB ((uint64_t)1024 * 1024)

static void test_profiles(void) {
    ArivanMemoryProfile profile;
    assert(arivan_memory_profile("8gb", &profile) == 0);
    assert(profile.engine_limit_bytes == 5376 * MIB);
    assert(profile.context_tokens == 2048);
    assert(profile.vision_tokens == 128);
    assert(profile.pinned_cache_bytes == 384 * MIB);
    assert(profile.mtp_depth == 1);
    assert(profile.experimental == 1);
    assert(arivan_memory_profile("not-a-profile", &profile) == -1);
}

static void test_hard_limit_is_transactional(void) {
    ArivanMemoryBudget budget;
    arivan_memory_budget_init(&budget, 1024);
    assert(arivan_memory_reserve(&budget, ARIVAN_MEM_PERMANENT, 600) == 0);
    assert(arivan_memory_reserve(&budget, ARIVAN_MEM_VISION, 500) == 1);
    assert(budget.current_bytes == 600);
    assert(budget.by_kind[ARIVAN_MEM_VISION] == 0);
    assert(arivan_memory_available(&budget) == 424);
}

static void test_release_and_peak(void) {
    ArivanMemoryBudget budget;
    arivan_memory_budget_init(&budget, 2048);
    assert(arivan_memory_reserve(&budget, ARIVAN_MEM_DENSE_WINDOW, 700) == 0);
    assert(arivan_memory_reserve(&budget, ARIVAN_MEM_EXPERT_PINNED, 300) == 0);
    assert(budget.peak_bytes == 1000);
    assert(arivan_memory_release(&budget, ARIVAN_MEM_DENSE_WINDOW, 700) == 0);
    assert(budget.current_bytes == 300);
    assert(budget.peak_bytes == 1000);
    assert(arivan_memory_release(&budget, ARIVAN_MEM_DENSE_WINDOW, 1) == -1);
    assert(budget.current_bytes == 300);
}

static void test_phase_and_names(void) {
    ArivanMemoryBudget budget;
    arivan_memory_budget_init(&budget, 1);
    arivan_memory_budget_set_phase(&budget, ARIVAN_PHASE_VISION);
    assert(budget.phase == ARIVAN_PHASE_VISION);
    assert(arivan_memory_phase_name(budget.phase)[0] == 'v');
    assert(arivan_memory_kind_name(ARIVAN_MEM_MTP)[0] == 'm');
}

static void test_uniform_cache_clamps_and_reserves(void) {
    ArivanMemoryBudget budget;
    uint32_t slots = 99;
    arivan_memory_budget_init(&budget, 1000);
    assert(arivan_memory_reserve(&budget, ARIVAN_MEM_PERMANENT, 200) == 0);
    assert(arivan_memory_reserve_uniform_cache(
               &budget, ARIVAN_MEM_EXPERT_LRU, 120, 500, 10, 1, &slots) == 0);
    assert(slots == 4);
    assert(budget.by_kind[ARIVAN_MEM_EXPERT_LRU] == 480);
    assert(budget.current_bytes == 680);
}

static void test_uniform_cache_refusal_is_transactional(void) {
    ArivanMemoryBudget budget;
    uint32_t slots = 99;
    arivan_memory_budget_init(&budget, 1000);
    assert(arivan_memory_reserve_uniform_cache(
               &budget, ARIVAN_MEM_EXPERT_LRU, 120, 100, 10, 1, &slots) == 1);
    assert(slots == 0);
    assert(budget.current_bytes == 0);
    assert(arivan_memory_reserve_uniform_cache(
               &budget, ARIVAN_MEM_EXPERT_LRU, 0, 100, 10, 1, &slots) == -1);
    assert(budget.current_bytes == 0);
}

int main(void) {
    test_profiles();
    test_hard_limit_is_transactional();
    test_release_and_peak();
    test_phase_and_names();
    test_uniform_cache_clamps_and_reserves();
    test_uniform_cache_refusal_is_transactional();
    puts("arivan memory budget tests passed");
    return 0;
}
