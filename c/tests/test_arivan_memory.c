#include "../arivan_memory.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

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

static void test_reclassify_preserves_total_and_is_transactional(void) {
    ArivanMemoryBudget budget;
    arivan_memory_budget_init(&budget, 2048);
    assert(arivan_memory_reserve(&budget, ARIVAN_MEM_WORKSPACE, 700) == 0);
    assert(arivan_memory_reclassify(
               &budget, ARIVAN_MEM_WORKSPACE,
               ARIVAN_MEM_RESIDENT_WEIGHTS, 600) == 0);
    assert(budget.current_bytes == 700);
    assert(budget.peak_bytes == 700);
    assert(budget.by_kind[ARIVAN_MEM_WORKSPACE] == 100);
    assert(budget.by_kind[ARIVAN_MEM_RESIDENT_WEIGHTS] == 600);
    assert(arivan_memory_reclassify(
               &budget, ARIVAN_MEM_WORKSPACE,
               ARIVAN_MEM_RESIDENT_WEIGHTS, 101) == -1);
    assert(budget.by_kind[ARIVAN_MEM_WORKSPACE] == 100);
    assert(budget.by_kind[ARIVAN_MEM_RESIDENT_WEIGHTS] == 600);
}

static void test_phase_and_names(void) {
    ArivanMemoryBudget budget;
    arivan_memory_budget_init(&budget, 1000);
    assert(arivan_memory_reserve(&budget, ARIVAN_MEM_PERMANENT, 100) == 0);
    assert(budget.peak_by_phase[ARIVAN_PHASE_LOAD] == 100);
    arivan_memory_budget_set_phase(&budget, ARIVAN_PHASE_VISION);
    assert(budget.phase == ARIVAN_PHASE_VISION);
    assert(budget.peak_by_phase[ARIVAN_PHASE_VISION] == 100);
    assert(arivan_memory_reserve(&budget, ARIVAN_MEM_VISION, 250) == 0);
    assert(budget.peak_by_phase[ARIVAN_PHASE_VISION] == 350);
    assert(arivan_memory_release(&budget, ARIVAN_MEM_VISION, 250) == 0);
    arivan_memory_budget_set_phase(&budget, ARIVAN_PHASE_DECODE);
    assert(budget.peak_by_phase[ARIVAN_PHASE_DECODE] == 100);
    assert(arivan_memory_phase_name(budget.phase)[0] == 'd');
    assert(arivan_memory_kind_name(ARIVAN_MEM_MTP)[0] == 'm');
    assert(strcmp(arivan_memory_kind_name(ARIVAN_MEM_VISION_WINDOW),
                  "vision-window") == 0);
}

static void test_checked_sizes(void) {
    uint64_t value = 0;
    assert(arivan_memory_checked_add(40, 2, &value) == 0 && value == 42);
    assert(arivan_memory_checked_mul(6, 7, &value) == 0 && value == 42);
    assert(arivan_memory_checked_add(UINT64_MAX, 1, &value) != 0);
    assert(arivan_memory_checked_mul(UINT64_MAX, 2, &value) != 0);
    assert(arivan_memory_checked_add(1, 2, NULL) != 0);
    assert(arivan_memory_checked_mul(1, 2, NULL) != 0);
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

static void test_expert_plan_prefers_per_layer_cache(void) {
    ArivanMemoryBudget budget;
    ArivanExpertPlan plan;
    arivan_memory_budget_init(&budget, 2000);
    assert(arivan_memory_reserve(&budget, ARIVAN_MEM_PERMANENT, 200) == 0);
    assert(arivan_memory_plan_experts(
               &budget, 100, 4, 1200, 5, 2, &plan) == 0);
    assert(plan.cache_slots_per_layer == 3);
    assert(plan.staging_slots == 0);
    assert(plan.cache_bytes == 1200);
    assert(budget.by_kind[ARIVAN_MEM_EXPERT_LRU] == 1200);
    assert(budget.by_kind[ARIVAN_MEM_EXPERT_STAGING] == 0);
}

static void test_expert_plan_falls_back_to_bounded_staging(void) {
    ArivanMemoryBudget budget;
    ArivanExpertPlan plan;
    arivan_memory_budget_init(&budget, 1000);
    assert(arivan_memory_reserve(&budget, ARIVAN_MEM_PERMANENT, 750) == 0);
    assert(arivan_memory_plan_experts(
               &budget, 120, 4, 500, 4, 2, &plan) == 0);
    assert(plan.cache_slots_per_layer == 0);
    assert(plan.staging_slots == 2);
    assert(plan.staging_bytes == 240);
    assert(budget.by_kind[ARIVAN_MEM_EXPERT_LRU] == 0);
    assert(budget.by_kind[ARIVAN_MEM_EXPERT_STAGING] == 240);
    assert(budget.current_bytes == 990);
}

static void test_expert_plan_refuses_when_one_staging_slot_does_not_fit(void) {
    ArivanMemoryBudget budget;
    ArivanExpertPlan plan;
    arivan_memory_budget_init(&budget, 1000);
    assert(arivan_memory_reserve(&budget, ARIVAN_MEM_PERMANENT, 900) == 0);
    assert(arivan_memory_plan_experts(
               &budget, 120, 4, 500, 4, 2, &plan) == 1);
    assert(plan.cache_slots_per_layer == 0);
    assert(plan.staging_slots == 0);
    assert(budget.current_bytes == 900);
}

int main(void) {
    test_profiles();
    test_hard_limit_is_transactional();
    test_release_and_peak();
    test_reclassify_preserves_total_and_is_transactional();
    test_phase_and_names();
    test_checked_sizes();
    test_uniform_cache_clamps_and_reserves();
    test_uniform_cache_refusal_is_transactional();
    test_expert_plan_prefers_per_layer_cache();
    test_expert_plan_falls_back_to_bounded_staging();
    test_expert_plan_refuses_when_one_staging_slot_does_not_fit();
    puts("arivan memory budget tests passed");
    return 0;
}
