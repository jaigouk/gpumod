"""ServiceManager — top-level orchestrator composing all service-layer components.

Coordinates mode switching, status reporting, and delegates individual
service start/stop to the :class:`LifecycleManager`.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING

from gpumod.models import (
    ModeResult,
    ServiceInfo,
    ServiceState,
    ServiceStatus,
    SleepMode,
    SystemStatus,
)
from gpumod.services.health import HealthMonitor
from gpumod.services.ram import InsufficientRAMError, RAMTracker
from gpumod.services.vram import NvidiaSmiError

if TYPE_CHECKING:
    from gpumod.db import Database
    from gpumod.models import Service
    from gpumod.services.lifecycle import LifecycleManager
    from gpumod.services.registry import ServiceRegistry
    from gpumod.services.sleep import SleepController
    from gpumod.services.vram import VRAMTracker

logger = logging.getLogger(__name__)

# RAM safeguard tunables. A large-VRAM llama.cpp service typically pulls
# RAM proportional to its VRAM footprint (model weights staged + pinned
# transfer buffers + Python heap). The defaults reject the dangerous case
# where MemAvailable looks fine but pages are too fragmented for a
# contiguous CUDA pinned allocation.
RAM_HEADROOM_RATIO = 0.3  # require avail >= sum(vram) * ratio + absolute
RAM_ABSOLUTE_HEADROOM_MB = 5_000
RAM_HARD_FLOOR_MB = 6_000  # never start a service if avail < this
DEFAULT_SETTLE_SECONDS = 15.0  # pause between outgoing-stop and incoming-start
SETTLE_ENV_VAR = "GPUMOD_SETTLE_SECONDS"


class ServiceManager:
    """Top-level orchestrator for GPU service management.

    Composes :class:`Database`, :class:`ServiceRegistry`,
    :class:`LifecycleManager`, :class:`VRAMTracker`, and
    :class:`SleepController` to provide high-level operations such as
    mode switching and system status queries.

    Parameters
    ----------
    db:
        The Database instance for configuration and state persistence.
    registry:
        The ServiceRegistry for discovering services and their drivers.
    lifecycle:
        The LifecycleManager for starting/stopping services.
    vram:
        The VRAMTracker for GPU memory queries and estimation.
    sleep:
        The SleepController for sleep/wake operations.
    health:
        Optional HealthMonitor for continuous health checking.
        If None, a default instance is created.
    """

    def __init__(
        self,
        db: Database,
        registry: ServiceRegistry,
        lifecycle: LifecycleManager,
        vram: VRAMTracker,
        sleep: SleepController,
        health: HealthMonitor | None = None,
        ram: RAMTracker | None = None,
    ) -> None:
        self._db = db
        self._registry = registry
        self._lifecycle = lifecycle
        self._vram = vram
        self._sleep = sleep
        self._ram = ram or RAMTracker()
        self._health = health or HealthMonitor(
            registry=registry,
            on_state_change=self._on_health_change,
        )

    # ------------------------------------------------------------------
    # RAM safeguard helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _required_ram_mb(incoming_vram_mb: int) -> int:
        """Estimate RAM needed to safely start services with given VRAM."""
        return int(incoming_vram_mb * RAM_HEADROOM_RATIO) + RAM_ABSOLUTE_HEADROOM_MB

    @staticmethod
    def _settle_seconds() -> float:
        """Read settle period from env, falling back to default."""
        raw = os.environ.get(SETTLE_ENV_VAR)
        if raw is None or raw == "":
            return DEFAULT_SETTLE_SECONDS
        try:
            return max(0.0, float(raw))
        except ValueError:
            logger.warning(
                "Invalid %s=%r, using default %.1fs",
                SETTLE_ENV_VAR,
                raw,
                DEFAULT_SETTLE_SECONDS,
            )
            return DEFAULT_SETTLE_SECONDS

    async def _check_ram_safeguard(self, incoming_vram_mb: int) -> str | None:
        """Return None if safe to start, else an error message.

        Two-tier check:
          1. Hard floor — refuse if MemAvailable < RAM_HARD_FLOOR_MB
             regardless of service size.
          2. Headroom check — refuse if MemAvailable < required estimate
             for the incoming workload.
        """
        usage = await self._ram.get_usage()
        if usage.available_mb < RAM_HARD_FLOOR_MB:
            return (
                f"RAM safeguard: only {usage.available_mb} MB available, "
                f"hard floor is {RAM_HARD_FLOOR_MB} MB. Refusing to start "
                "to avoid CUDA pinned-memory freeze. Stop other workloads "
                "and retry."
            )

        if incoming_vram_mb > 0:
            required = self._required_ram_mb(incoming_vram_mb)
            if usage.available_mb < required:
                return (
                    f"RAM safeguard: incoming services need ~{required} MB "
                    f"available ({incoming_vram_mb} MB VRAM * "
                    f"{RAM_HEADROOM_RATIO} + {RAM_ABSOLUTE_HEADROOM_MB} MB "
                    f"headroom), but only {usage.available_mb} MB free. "
                    "Refusing to start to avoid CUDA pinned-memory freeze."
                )
        return None

    async def _on_health_change(self, service_id: str, healthy: bool) -> None:
        """React to health state changes reported by HealthMonitor."""
        if healthy:
            logger.info("Service %r recovered", service_id)
        else:
            logger.warning("Service %r is unhealthy", service_id)

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    async def switch_mode(self, target_mode_id: str) -> ModeResult:  # noqa: C901
        """Switch to the target mode, managing service lifecycle and VRAM.

        Steps:
        1. Validate target mode exists.
        2. Get current and target service sets.
        3. Compute diff (to_stop, to_start).
        4. VRAM pre-flight check.
        5. Stop services not in target mode.
        6. Start services not in current mode.
        7. Update current mode in DB.

        Parameters
        ----------
        target_mode_id:
            The ID of the mode to switch to.

        Returns
        -------
        ModeResult
            The result of the switch operation.
        """
        logger.info("Switching to mode %r", target_mode_id)

        # 1. Validate target mode exists
        target_mode = await self._db.get_mode(target_mode_id)
        if target_mode is None:
            logger.warning("Mode not found: %r", target_mode_id)
            return ModeResult(
                success=False,
                mode_id=target_mode_id,
                errors=[f"Mode not found: {target_mode_id}"],
            )

        # 2. Get current and target service sets
        current_mode_id = await self._db.get_current_mode()

        current_service_ids: set[str] = set()
        if current_mode_id is not None:
            current_services = await self._db.get_mode_services(current_mode_id)
            current_service_ids = {s.id for s in current_services}

        target_services = await self._db.get_mode_services(target_mode_id)
        target_service_ids = {s.id for s in target_services}

        # 3. Compute diff (include orphan services from prior modes)
        # Get ALL services currently running or sleeping on the system
        running_services = await self._registry.list_running()
        running_service_ids = {s.id for s in running_services}

        logger.debug(
            "Mode switch state: current_mode=%r, target_mode=%r, "
            "current_mode_services=%s, target_services=%s, running_services=%s",
            current_mode_id,
            target_mode_id,
            sorted(current_service_ids),
            sorted(target_service_ids),
            sorted(running_service_ids),
        )

        # to_stop includes:
        # 1. Services defined in current mode but not in target mode
        # 2. Services actually running/sleeping that aren't in target mode (orphans)
        to_stop = (current_service_ids | running_service_ids) - target_service_ids
        to_start = target_service_ids - current_service_ids

        # Log orphan detection
        orphan_services = running_service_ids - current_service_ids - target_service_ids
        if orphan_services:
            logger.info(
                "Detected orphan services from prior modes: %s",
                sorted(orphan_services),
            )

        # 4. VRAM pre-flight check
        gpu_info = await self._vram.get_gpu_info()
        total_target_vram = 0
        for svc in target_services:
            total_target_vram += await self._vram.estimate_service_vram(svc)

        if total_target_vram > gpu_info.vram_total_mb:
            logger.warning(
                "VRAM exceeded for mode %r: requires %dMB, available %dMB",
                target_mode_id,
                total_target_vram,
                gpu_info.vram_total_mb,
            )
            return ModeResult(
                success=False,
                mode_id=target_mode_id,
                errors=[
                    f"VRAM exceeded: target mode requires {total_target_vram}MB "
                    f"but GPU only has {gpu_info.vram_total_mb}MB"
                ],
            )

        # 5. Handle outgoing services (free VRAM first)
        slept_ids, stopped_ids = await self._handle_outgoing_services(to_stop)

        # 5.5 Wait for VRAM to be released before starting incoming services
        if to_start:
            # Find the largest VRAM requirement among incoming services
            max_incoming_vram = 0
            total_incoming_vram = 0
            for svc_id in to_start:
                svc = await self._registry.get(svc_id)
                max_incoming_vram = max(max_incoming_vram, svc.vram_mb)
                total_incoming_vram += svc.vram_mb

            if max_incoming_vram > 0:
                logger.info(
                    "Waiting for VRAM release before starting services (need %d MB)",
                    max_incoming_vram,
                )
                vram_released = await self._vram.wait_for_vram_release(
                    required_mb=max_incoming_vram,
                    timeout_s=120.0,
                    poll_interval_s=0.5,
                    safety_margin_mb=512,
                )
                if not vram_released:
                    logger.error(
                        "VRAM not released within timeout. "
                        "Previous service may still hold GPU memory."
                    )
                    return ModeResult(
                        success=False,
                        mode_id=target_mode_id,
                        errors=[
                            "VRAM not released within timeout. "
                            "Previous service may still hold GPU memory."
                        ],
                    )

            # 5.6 Settle period: give the kernel time to consolidate freed
            # pages between teardown and CUDA pinned allocations. Skip if
            # nothing was actually stopped/slept.
            if slept_ids or stopped_ids:
                settle = self._settle_seconds()
                if settle > 0:
                    logger.info(
                        "Settle period: waiting %.1fs for kernel to consolidate freed pages",
                        settle,
                    )
                    await asyncio.sleep(settle)

            # 5.7 RAM safeguard — pre-flight check for the CUDA pinned-memory
            # contiguous-allocation freeze pattern.
            error_msg = await self._check_ram_safeguard(total_incoming_vram)
            if error_msg:
                logger.error("Mode switch blocked by RAM safeguard: %s", error_msg)
                return ModeResult(
                    success=False,
                    mode_id=target_mode_id,
                    errors=[error_msg],
                )

        # 6. Handle incoming services
        woken_ids, started_ids = await self._handle_incoming_services(to_start)

        # 7. Update current mode in DB
        await self._db.set_current_mode(target_mode_id)

        logger.info(
            "Mode switch to %r complete: started=%s, woken=%s, slept=%s, stopped=%s",
            target_mode_id,
            started_ids,
            woken_ids,
            slept_ids,
            stopped_ids,
        )

        return ModeResult(
            success=True,
            mode_id=target_mode_id,
            started=started_ids + woken_ids,  # All services now running
            stopped=stopped_ids + slept_ids,  # All services no longer running
        )

    async def _handle_outgoing_services(
        self, service_ids: set[str]
    ) -> tuple[list[str], list[str]]:
        """Handle services leaving the current mode.

        Sleep-capable services are slept; non-sleep services are stopped.

        Returns
        -------
        tuple[list[str], list[str]]
            (slept_ids, stopped_ids)
        """
        slept_ids: list[str] = []
        stopped_ids: list[str] = []

        for service_id in sorted(service_ids):
            service = await self._registry.get(service_id)
            driver = self._registry.get_driver(service.driver)
            status = await driver.status(service)

            is_sleep_capable = driver.supports_sleep and service.sleep_mode != SleepMode.NONE

            if is_sleep_capable and status.state == ServiceState.RUNNING:
                logger.info("Sleeping service %r (not in target mode)", service_id)
                await self._lifecycle.sleep(service_id)
                slept_ids.append(service_id)
            elif status.state == ServiceState.SLEEPING:
                # Orphan sleeping service - must be stopped to free VRAM
                logger.info(
                    "Stopping sleeping service %r (orphan, not in target mode)", service_id
                )
                await self._lifecycle.stop(service_id)
                stopped_ids.append(service_id)
            elif status.state in (ServiceState.RUNNING, ServiceState.UNKNOWN):
                logger.info("Stopping service %r (not in target mode)", service_id)
                await self._lifecycle.stop(service_id)
                stopped_ids.append(service_id)
            # STOPPED services don't need any action

        return slept_ids, stopped_ids

    async def _handle_incoming_services(
        self, service_ids: set[str]
    ) -> tuple[list[str], list[str]]:
        """Handle services entering the target mode.

        Sleeping services are woken; stopped services are started.

        Returns
        -------
        tuple[list[str], list[str]]
            (woken_ids, started_ids)
        """
        woken_ids: list[str] = []
        started_ids: list[str] = []

        for service_id in sorted(service_ids):
            service = await self._registry.get(service_id)
            driver = self._registry.get_driver(service.driver)
            status = await driver.status(service)

            if status.state == ServiceState.SLEEPING:
                logger.info("Waking service %r (required by target mode)", service_id)
                await self._lifecycle.wake(service_id)
                woken_ids.append(service_id)
            else:
                logger.info("Starting service %r (required by target mode)", service_id)
                await self._lifecycle.start(service_id)
                started_ids.append(service_id)

        return woken_ids, started_ids

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    async def get_status(self) -> SystemStatus:
        """Return full system status including GPU, VRAM, and all services.

        GPU information is fetched gracefully — if nvidia-smi fails,
        ``gpu`` and ``vram`` will be ``None``.

        Returns
        -------
        SystemStatus
            The current system status.
        """
        logger.info("Gathering system status")
        current_mode = await self._db.get_current_mode()

        # Gather service statuses concurrently
        all_services = await self._registry.list_all()

        async def _get_service_info(svc: Service) -> ServiceInfo:
            driver = self._registry.get_driver(svc.driver)
            try:
                status = await driver.status(svc)
            except Exception:
                logger.exception("Failed to get status for %s", svc.id)
                status = ServiceStatus(state=ServiceState.UNKNOWN)
            return ServiceInfo(service=svc, status=status)

        service_infos = list(
            await asyncio.gather(*[_get_service_info(svc) for svc in all_services])
        )

        # GPU info (graceful on failure)
        gpu_info = None
        vram_usage = None
        try:
            gpu_info = await self._vram.get_gpu_info()
            vram_usage = await self._vram.get_usage()
        except NvidiaSmiError:
            logger.warning("nvidia-smi unavailable; GPU info will be omitted")

        return SystemStatus(
            gpu=gpu_info,
            vram=vram_usage,
            current_mode=current_mode,
            services=service_infos,
        )

    # ------------------------------------------------------------------
    # Convenience delegates
    # ------------------------------------------------------------------

    async def start_service(self, service_id: str) -> None:
        """Start a service via the lifecycle manager.

        Pre-flight RAM safeguard runs before any allocation to avoid the
        CUDA pinned-memory contiguous-allocation freeze pattern.

        Parameters
        ----------
        service_id:
            The ID of the service to start.

        Raises
        ------
        InsufficientRAMError
            If MemAvailable is below the hard floor or the headroom
            estimate for this service.
        """
        service = await self._registry.get(service_id)
        error_msg = await self._check_ram_safeguard(service.vram_mb)
        if error_msg:
            usage = await self._ram.get_usage()
            required = self._required_ram_mb(service.vram_mb)
            logger.error("start_service %r blocked: %s", service_id, error_msg)
            raise InsufficientRAMError(
                required_mb=required,
                available_mb=usage.available_mb,
                message=error_msg,
            )

        logger.info("Starting service %r", service_id)
        await self._lifecycle.start(service_id)

    async def stop_service(self, service_id: str) -> None:
        """Stop a service via the lifecycle manager.

        Parameters
        ----------
        service_id:
            The ID of the service to stop.
        """
        logger.info("Stopping service %r", service_id)
        await self._lifecycle.stop(service_id)
