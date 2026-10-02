"""Declarative memory profiles for Arivan.

Profiles are an admission-control contract, not a promise that every adapter
already satisfies the requested ceiling. Native engines report certification
separately as the tensor pager is integrated family by family.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json


@dataclass(frozen=True)
class MemoryProfile:
    name: str
    installed_gib: float
    engine_budget_gib: float
    context_tokens: int
    kv_slots: int
    vision_tokens: int
    pinned_cache_mib: int
    expert_cache_mib: int
    prefill_chunk: int
    mtp_depth: int
    status: str

    @property
    def os_reserve_gib(self) -> float:
        return round(self.installed_gib - self.engine_budget_gib, 2)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["os_reserve_gib"] = self.os_reserve_gib
        return result

    def engine_environment(self, arch: str) -> dict[str, str]:
        """Return existing engine settings plus contracts for the new pager."""
        env = {
            "ARIVAN_MEMORY_PROFILE": self.name,
            "ARIVAN_MEMORY_LIMIT_BYTES": str(int(self.engine_budget_gib * 1024**3)),
            "ARIVAN_PIN_CACHE_BYTES": str(self.pinned_cache_mib * 1024**2),
            "ARIVAN_EXPERT_CACHE_BYTES": str(self.expert_cache_mib * 1024**2),
            "ARIVAN_MTP_DEPTH": str(self.mtp_depth),
            "COLI_KV_SLOTS": str(self.kv_slots),
        }
        if arch in {"glm", "glm53", "kimi", "olmoe", "deepseek_v4"}:
            env["RAM_GB"] = f"{self.engine_budget_gib:g}"
        if arch == "glm53":
            env.update({
                "GLM53_EXPERT_GB": f"{self.expert_cache_mib / 1024:.3f}",
                "GLM53_MAXT": str(self.context_tokens),
                "GLM53_MAX_IMAGE_TOKENS": str(self.vision_tokens),
                "GLM53_PREFILL_CHUNK": str(self.prefill_chunk),
                "GLM53_BITS": "4",
            })
        elif arch == "glm":
            env.update({
                "CTX_MAX": str(self.context_tokens),
                "PIN_GB": f"{self.pinned_cache_mib / 1024:.3f}",
                "DRAFT": str(self.mtp_depth),
            })
            if self.name == "8gb":
                env["TRUNK_RESIDENT_LAYERS"] = "0"
        return env


PROFILES = {
    "8gb": MemoryProfile(
        name="8gb", installed_gib=8.0, engine_budget_gib=5.25,
        context_tokens=2048, kv_slots=1, vision_tokens=128,
        pinned_cache_mib=384, expert_cache_mib=512,
        prefill_chunk=8, mtp_depth=1, status="experimental",
    ),
    "16gb": MemoryProfile(
        name="16gb", installed_gib=16.0, engine_budget_gib=12.0,
        context_tokens=8192, kv_slots=1, vision_tokens=256,
        pinned_cache_mib=1024, expert_cache_mib=2048,
        prefill_chunk=32, mtp_depth=1, status="experimental",
    ),
    "32gb": MemoryProfile(
        name="32gb", installed_gib=32.0, engine_budget_gib=26.0,
        context_tokens=32768, kv_slots=2, vision_tokens=512,
        pinned_cache_mib=4096, expert_cache_mib=8192,
        prefill_chunk=64, mtp_depth=1, status="baseline",
    ),
}


def get_profile(name: str) -> MemoryProfile:
    try:
        return PROFILES[name.lower()]
    except (AttributeError, KeyError) as error:
        choices = ", ".join(PROFILES)
        raise ValueError(f"unknown memory profile {name!r}; choose {choices}") from error


def render_profile(profile: MemoryProfile, *, json_output: bool = False) -> str:
    if json_output:
        return json.dumps(profile.to_dict(), indent=2, sort_keys=True)
    rows = [
        ("Profile", profile.name),
        ("Status", profile.status),
        ("Installed RAM target", f"{profile.installed_gib:g} GiB"),
        ("Engine hard budget", f"{profile.engine_budget_gib:g} GiB"),
        ("OS/headroom reserve", f"{profile.os_reserve_gib:g} GiB"),
        ("Context", f"{profile.context_tokens:,} tokens"),
        ("KV slots", str(profile.kv_slots)),
        ("Vision", f"{profile.vision_tokens} image tokens"),
        ("Pinned experts", f"{profile.pinned_cache_mib} MiB"),
        ("Ordinary expert cache", f"{profile.expert_cache_mib} MiB"),
        ("Prefill chunk", str(profile.prefill_chunk)),
        ("MTP draft depth", str(profile.mtp_depth)),
    ]
    width = max(len(label) for label, _ in rows)
    return "\n".join(f"{label:<{width}}  {value}" for label, value in rows)
