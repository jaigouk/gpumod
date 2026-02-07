"""VRAM simulation engine for gpumod.

Provides :class:`SimulationEngine` for simulating VRAM usage of modes
and service sets, including alternative suggestions when services
don't fit in GPU memory.  Uses the Strategy pattern for alternative
generation (Open/Closed principle).
"""

from __future__ import annotations

import abc
import asyncio
from typing import TYPE_CHECKING

from gpumod.models import SimulationAlternative, SimulationResult
from gpumod.validation import validate_context_override, validate_mode_id, validate_service_id

if TYPE_CHECKING:
    from gpumod.db import Database
    from gpumod.models import Service
    from gpumod.registry import ModelRegistry
    from gpumod.services.vram import VRAMTracker

_MAX_ALTERNATIVES = 10


class SimulationError(Exception):
    """Raised when a simulation cannot be performed."""


# ---------------------------------------------------------------------------
# Strategy pattern for alternative generation
# ---------------------------------------------------------------------------


class AlternativeStrategy(abc.ABC):
    """Base class for alternative generation strategies."""

    @abc.abstractmethod
    async def generate(
        self,
        services: list[Service],
        total_vram: int,
        gpu_total_mb: int,
        model_registry: ModelRegistry,
    ) -> list[SimulationAlternative]:
        """Generate alternatives for the given services.

        Parameters
        ----------
        services:
            The proposed service list.
        total_vram:
            The total proposed VRAM usage.
        gpu_total_mb:
            Total GPU VRAM capacity.
        model_registry:
            The model registry for VRAM estimates.

        Returns
        -------
        list[SimulationAlternative]
            Zero or more suggested alternatives.
        """


class SleepStrategy(AlternativeStrategy):
    """Suggest sleeping services that support sleep mode."""

    async def generate(
        self,
        services: list[Service],
        total_vram: int,
        gpu_total_mb: int,
        model_registry: ModelRegistry,
    ) -> list[SimulationAlternative]:
        alternatives: list[SimulationAlternative] = []
        for svc in services:
            if svc.sleep_mode != "none":
                # Estimate ~80% VRAM saved when sleeping
                saved = int(svc.vram_mb * 0.8)
                alternatives.append(
                    SimulationAlternative(
                        id=f"sleep-{svc.id}",
                        strategy="sleep",
                        description=f"Sleep {svc.id} ({svc.sleep_mode} mode)",
                        affected_services=[svc.id],
                        vram_saved_mb=saved,
                        projected_total_mb=total_vram - saved,
                        trade_offs=[
                            f"Increased latency on wake for {svc.id}",
                        ],
                    )
                )
        return alternatives


class ContextReductionStrategy(AlternativeStrategy):
    """Suggest reducing context window for services with models."""

    async def generate(
        self,
        services: list[Service],
        total_vram: int,
        gpu_total_mb: int,
        model_registry: ModelRegistry,
    ) -> list[SimulationAlternative]:
        alternatives: list[SimulationAlternative] = []
        for svc in services:
            if svc.model_id is None:
                continue
            # Try halving context iteratively
            try:
                current_vram = svc.vram_mb
                halved_context = 2048  # Start with a reduced context
                estimated = await model_registry.estimate_vram(svc.model_id, halved_context)
                saved = current_vram - estimated
                if saved > 0:
                    alternatives.append(
                        SimulationAlternative(
                            id=f"ctx-reduce-{svc.id}",
                            strategy="context_reduction",
                            description=(
                                f"Reduce context for {svc.id} to {halved_context} tokens"
                            ),
                            affected_services=[svc.id],
                            vram_saved_mb=saved,
                            projected_total_mb=total_vram - saved,
                            trade_offs=[
                                f"Reduced context window for {svc.id}",
                            ],
                        )
                    )
            except (ValueError, RuntimeError):
                # Skip if model not found or estimation fails
                continue
        return alternatives


class ServiceRemovalStrategy(AlternativeStrategy):
    """Suggest removing services, ranked by VRAM usage (largest first)."""

    async def generate(
        self,
        services: list[Service],
        total_vram: int,
        gpu_total_mb: int,
        model_registry: ModelRegistry,
    ) -> list[SimulationAlternative]:
        alternatives: list[SimulationAlternative] = []
        # Sort by VRAM descending — suggest removing largest first
        sorted_services = sorted(services, key=lambda s: s.vram_mb, reverse=True)
        for svc in sorted_services:
            alternatives.append(
                SimulationAlternative(
                    id=f"remove-{svc.id}",
                    strategy="service_removal",
                    description=f"Remove {svc.id} ({svc.vram_mb} MB)",
                    affected_services=[svc.id],
                    vram_saved_mb=svc.vram_mb,
                    projected_total_mb=total_vram - svc.vram_mb,
                    trade_offs=[
                        f"Service {svc.id} will be unavailable",
                    ],
                )
            )
        return alternatives


# ---------------------------------------------------------------------------
# SimulationEngine
# ---------------------------------------------------------------------------


class SimulationEngine:
    """Simulates VRAM usage for modes and service sets.

    Parameters
    ----------
    db:
        The Database instance for looking up modes and services.
    vram:
        The VRAMTracker for querying GPU info.
    model_registry:
        The ModelRegistry for VRAM estimation with context overrides.
    """

    def __init__(
        self,
        db: Database,
        vram: VRAMTracker,
        model_registry: ModelRegistry,
    ) -> None:
        self._db = db
        self._vram = vram
        self._model_registry = model_registry
        self._strategies: list[AlternativeStrategy] = [
            SleepStrategy(),
            ContextReductionStrategy(),
            ServiceRemovalStrategy(),
        ]

    async def _get_gpu_total(self) -> int:
        """Get GPU total VRAM, wrapping errors in SimulationError."""
        try:
            gpu_info = await self._vram.get_gpu_info()
        except Exception as exc:
            msg = f"GPU info unavailable: {exc}"
            raise SimulationError(msg) from exc
        return gpu_info.vram_total_mb

    async def _resolve_vram(
        self,
        service: Service,
        context_overrides: dict[str, int] | None,
    ) -> int:
        """Resolve the VRAM for a service, applying context overrides."""
        if context_overrides and service.id in context_overrides and service.model_id:
            return await self._model_registry.estimate_vram(
                service.model_id, context_overrides[service.id]
            )
        return service.vram_mb

    async def _build_result(
        self,
        services: list[Service],
        gpu_total_mb: int,
        context_overrides: dict[str, int] | None,
    ) -> SimulationResult:
        """Build a SimulationResult from the service list."""
        # Resolve VRAM concurrently
        vram_tasks = [self._resolve_vram(svc, context_overrides) for svc in services]
        vram_values = await asyncio.gather(*vram_tasks)

        # Build services with resolved VRAM
        resolved_services: list[Service] = []
        for svc, vram_mb in zip(services, vram_values, strict=True):
            if vram_mb != svc.vram_mb:
                resolved_services.append(svc.model_copy(update={"vram_mb": vram_mb}))
            else:
                resolved_services.append(svc)

        proposed_usage = sum(vram_values)
        headroom = gpu_total_mb - proposed_usage
        fits = proposed_usage <= gpu_total_mb

        alternatives: list[SimulationAlternative] = []
        if not fits:
            alternatives = await self._generate_alternatives(
                resolved_services, proposed_usage, gpu_total_mb
            )

        return SimulationResult(
            fits=fits,
            gpu_total_mb=gpu_total_mb,
            current_usage_mb=0,
            proposed_usage_mb=proposed_usage,
            headroom_mb=headroom,
            services=resolved_services,
            alternatives=alternatives,
        )

    async def _generate_alternatives(
        self,
        services: list[Service],
        total_vram: int,
        gpu_total_mb: int,
    ) -> list[SimulationAlternative]:
        """Generate alternatives using all strategies, capped at _MAX_ALTERNATIVES."""
        all_alternatives: list[SimulationAlternative] = []
        for strategy in self._strategies:
            alts = await strategy.generate(
                services, total_vram, gpu_total_mb, self._model_registry
            )
            all_alternatives.extend(alts)

        # Sort by vram_saved_mb descending and cap
        all_alternatives.sort(key=lambda a: a.vram_saved_mb, reverse=True)
        return all_alternatives[:_MAX_ALTERNATIVES]

    async def simulate_mode(
        self,
        mode_id: str,
        *,
        add: list[str] | None = None,
        remove: list[str] | None = None,
        context_overrides: dict[str, int] | None = None,
    ) -> SimulationResult:
        """Simulate VRAM usage for a mode with optional modifications.

        Parameters
        ----------
        mode_id:
            The mode to simulate.
        add:
            Additional service IDs to include beyond the mode's services.
        remove:
            Service IDs to exclude from the mode's services.
        context_overrides:
            Map of service_id to context_size (tokens) for VRAM re-estimation.

        Returns
        -------
        SimulationResult
            The simulation result.

        Raises
        ------
        ValueError
            If the mode or any referenced service is not found.
        SimulationError
            If GPU info is unavailable.
        """
        validate_mode_id(mode_id)

        # Validate add/remove IDs
        if add:
            for sid in add:
                validate_service_id(sid)
        if remove:
            for sid in remove:
                validate_service_id(sid)
        if context_overrides:
            for key, val in context_overrides.items():
                validate_context_override(key, val)

        gpu_total_mb = await self._get_gpu_total()

        mode = await self._db.get_mode(mode_id)
        if mode is None:
            msg = f"Mode not found: {mode_id!r}"
            raise ValueError(msg)

        services = await self._db.get_mode_services(mode_id)

        # Apply add
        if add:
            existing_ids = {s.id for s in services}
            for sid in add:
                if sid not in existing_ids:
                    svc = await self._db.get_service(sid)
                    if svc is None:
                        msg = f"Service not found: {sid!r}"
                        raise ValueError(msg)
                    services.append(svc)

        # Apply remove
        if remove:
            remove_set = set(remove)
            services = [s for s in services if s.id not in remove_set]

        return await self._build_result(services, gpu_total_mb, context_overrides)

    async def simulate_services(
        self,
        service_ids: list[str],
        *,
        context_overrides: dict[str, int] | None = None,
    ) -> SimulationResult:
        """Simulate VRAM usage for an explicit list of services.

        Parameters
        ----------
        service_ids:
            The service IDs to simulate.
        context_overrides:
            Map of service_id to context_size (tokens) for VRAM re-estimation.

        Returns
        -------
        SimulationResult
            The simulation result.

        Raises
        ------
        ValueError
            If any service is not found.
        SimulationError
            If GPU info is unavailable.
        """
        for sid in service_ids:
            validate_service_id(sid)
        if context_overrides:
            for key, val in context_overrides.items():
                validate_context_override(key, val)

        gpu_total_mb = await self._get_gpu_total()

        services: list[Service] = []
        for sid in service_ids:
            svc = await self._db.get_service(sid)
            if svc is None:
                msg = f"Service not found: {sid!r}"
                raise ValueError(msg)
            services.append(svc)

        return await self._build_result(services, gpu_total_mb, context_overrides)
