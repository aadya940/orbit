"""World: per-session container replacing all globals.

Holds drivers, journal, policy, probe cache and the step budget. Two
Worlds share nothing, which is what makes everything testable."""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from .drivers.base import CapabilityScore, Driver, run_ladder
from .journal import Journal
from .policy import Policy
from .types import (
    Action,
    ActionKind,
    ActionResult,
    BudgetExhausted,
    Observation,
    OrbitError,
    SurfaceUnreadable,
    TargetUnresolvable,
)


class World:
    """Per-session container for drivers, journal, policy and step budget.

    A World owns everything a single agent session needs: the driver set, the
    routing decision about which driver currently perceives the focused
    surface, the fallback ladder used for actions, and the step budget. Two
    Worlds share no state, which is what makes the system testable.

    Attributes
    ----------
    drivers : Dict[str, Driver]
        Driver instances by name, copied from the constructor argument.
    auto_route : bool
        True when no explicit primary was given, meaning perception is routed
        automatically to whichever driver can see the focused surface.
    primary : str
        Name of the driver currently used for perception.
    policy : Policy
        Policy object governing action behaviour.
    journal : Journal
        Structured event log for the session.
    probe_cache : Dict[str, List[CapabilityScore]]
        Capability scores per probed surface name.
    ladder_order : List[str]
        Driver names in fallback order for action execution.
    max_steps : int
        Hard ceiling on steps that may be spent.
    used : int
        Steps spent so far.
    surface_owner : str
        Name of the driver that opened the surface currently in focus.

    Notes
    -----
    The focused surface is SESSION state, not per-verb state. A later verb
    must keep looking at the app the previous verb switched to; it must not
    re-pick a background browser that still "sees" its own page.
    """

    def __init__(
        self,
        drivers: Dict[str, Driver],
        primary: Optional[str] = None,
        policy: Optional[Policy] = None,
        journal: Optional[Journal] = None,
        max_steps: int = 40,
        runtime: Optional[dict] = None,
    ) -> None:
        """Build a World from a driver set and optional session state.

        Parameters
        ----------
        drivers : Dict[str, Driver]
            Driver instances keyed by name. Must be non-empty.
        primary : str, optional
            Name of the driver to use for perception. When given, automatic
            routing is disabled and this driver is trusted. Default is None,
            which enables automatic routing.
        policy : Policy, optional
            Policy governing action behaviour. Default is None, which builds
            a default Policy.
        journal : Journal, optional
            Journal to append events to. Default is None, which builds a
            fresh Journal.
        max_steps : int, optional
            Maximum number of steps that may be spent. Default is 40.
        runtime : dict, optional
            Session-shared driver runtime (started backends, remembered
            focus owner). Default is None, which starts a fresh session.

        Raises
        ------
        ValueError
            If `drivers` is empty.

        Notes
        -----
        The remembered `primary` in `runtime` is honoured because the focused
        surface is SESSION state, not per-verb state: a later verb must not
        re-pick a background browser that still "sees" its page. The runtime
        dict is shared across a session's verbs so a native task does not
        relaunch a browser on its second verb; a fresh dict means a fresh
        session.
        """
        if not drivers:
            raise ValueError("World needs at least one driver")
        self.drivers: Dict[str, Driver] = dict(drivers)
        # When primary is given, respect it and don't auto-route. When it
        # isn't, route perception to whichever driver can actually see the
        # focused surface (browser -> dom, native window -> tree), re-picked
        # whenever the current one goes blind.
        self.auto_route: bool = primary is None
        # Focused surface is SESSION state, not per-verb: a later verb must
        # keep looking at the app the previous verb switched to, not re-pick
        # a still-"visible" background browser.
        remembered = (runtime or {}).get("primary")
        self.primary: str = primary or remembered or next(iter(self.drivers))
        self.policy: Policy = policy or Policy()
        self.journal: Journal = journal or Journal()
        self.probe_cache: Dict[str, List[CapabilityScore]] = {}
        self.ladder_order: List[str] = list(self.drivers)
        self.max_steps: int = max_steps
        self.used: int = 0
        # Driver runtime (which backends are started, last surface hint)
        # is shared across the session's verbs so a native task doesn't
        # relaunch a browser on its second verb. Fresh dict = fresh session.
        self._runtime: dict = runtime if runtime is not None else {}
        self._runtime.setdefault("started", set())
        self._routed: bool = False

    # Session-shared driver runtime, exposed as a plain attribute.
    @property
    def _started(self) -> Set[str]:
        """Names of drivers already started in this session.

        Returns
        -------
        Set[str]
            The live set of started driver names, shared across the session.

        Notes
        -----
        Backed by the session-shared runtime dict rather than an instance
        attribute, so a second verb in the same session does not restart
        backends that are already running.
        """
        return self._runtime["started"]

    # -- budget ------------------------------------------------------------
    @property
    def remaining(self) -> int:
        """Number of steps still available in the budget.

        Returns
        -------
        int
            `max_steps` minus `used`, clamped at zero.
        """
        return max(0, self.max_steps - self.used)

    def spend(self, n: int = 1) -> None:
        """Consume steps from the budget.

        Parameters
        ----------
        n : int, optional
            Number of steps to spend. Default is 1.

        Returns
        -------
        None
            This method mutates `used` in place.

        Raises
        ------
        BudgetExhausted
            If spending `n` steps would exceed `max_steps`. The budget is
            left unchanged and the exhaustion is journalled first.
        """
        if self.used + n > self.max_steps:
            self.journal.append("budget_exhausted", max_steps=self.max_steps, used=self.used)
            raise BudgetExhausted(
                f"budget of {self.max_steps} steps exhausted", max_steps=self.max_steps
            )
        self.used += n

    # -- perception / action ----------------------------------------------
    async def observe(self) -> Observation:
        """Perceive the focused surface through the primary driver.

        Routes perception first when automatic routing is enabled and no
        routing decision has been made yet, then delegates to the primary
        driver. If that driver has gone blind, re-routes once and retries.

        Returns
        -------
        Observation
            The current observation. When nothing has been opened yet, an
            empty observation with surface "none" and kind "none".

        Raises
        ------
        SurfaceUnreadable
            If the primary driver cannot read the surface and automatic
            routing is disabled, or if the re-routed driver also fails.

        Notes
        -----
        With nothing started, we deliberately do not probe-launch every
        backend just to look at an empty desktop; the first navigate opens a
        surface and the next observe routes to it. Blind is not the same as
        wrong surface: a driver that owns the surface escalates to vision
        rather than losing perception to a driver on a different surface.
        """
        if self.auto_route and not self._routed:
            # Nothing opened yet: don't probe-launch every backend to look at
            # an empty desktop. Return an empty observation; the first
            # navigate opens a surface and the next observe routes to it.
            if not self._started:
                return Observation(surface="none", kind="none")
            await self._route()
        try:
            return await self.drivers[self.primary].observe()
        except SurfaceUnreadable:
            if not self.auto_route:
                raise
            # Current perception driver went blind (surface changed under
            # us, e.g. focus moved browser -> native app). Re-route once.
            await self._route()
            return await self.drivers[self.primary].observe()

    async def _ensure_started(self, name: str) -> None:
        """Start a driver once, if it exposes a start hook.

        Parameters
        ----------
        name : str
            Name of the driver to start.

        Returns
        -------
        None
            This method records the driver in the started set.

        Notes
        -----
        The start hook may be synchronous or asynchronous, so its result is
        awaited only when awaitable. Membership is tracked in the
        session-shared runtime, so rungs are started lazily and at most once
        per session rather than once per verb.
        """
        if name in self._started:
            return
        driver = self.drivers[name]
        start = getattr(driver, "start", None)
        if start is not None:
            res = start()
            if hasattr(res, "__await__"):
                await res
        self._started.add(name)

    def _route_order(self) -> List[str]:
        """Build the probe order, most authoritative driver first.

        The order is: the surface owner (whoever opened what is in focus),
        then surface-agnostic rungs (vision), then other running drivers,
        then cold ones.

        Returns
        -------
        List[str]
            Driver names in probe order, each appearing exactly once.

        Notes
        -----
        Surface-agnostic rungs ground on the OWNER's pixels, so a blind but
        alive surface escalates within itself. Other running drivers and
        cold drivers are relevant only when the owned surface is genuinely
        gone. Blind is not the same as wrong surface: a canvas page the DOM
        cannot read must fall to vision, never to the accessibility driver
        that happens to "see" the browser's own chrome.
        """
        owner = self.surface_owner
        order = [owner] if owner in self.drivers else []
        order += [
            n for n, d in self.drivers.items()
            if getattr(d, "surface", "x") is None and n not in order
        ]
        order += [n for n in self.drivers if n in self._started and n not in order]
        order += [n for n in self.drivers if n not in order]
        return order

    def _set_primary(self, name: str) -> None:
        """Set the perception driver and remember it for later verbs.

        Parameters
        ----------
        name : str
            Name of the driver that should perceive from now on.

        Returns
        -------
        None
            This method mutates `primary` and the session runtime.

        Notes
        -----
        The choice is written into the session-shared runtime because the
        focused surface is SESSION state, not per-verb state: a later verb
        must not re-pick a background browser that still "sees" its page.
        """
        self.primary = name
        self._runtime["primary"] = name  # carry focus across verbs

    @property
    def surface_owner(self) -> str:
        """Driver that owns the surface currently in focus.

        Returns
        -------
        str
            Name of the driver that opened the focused surface, falling back
            to the current primary when no owner has been recorded.

        Notes
        -----
        Perception may fall back to a surface-agnostic rung, but the owner
        defines WHICH surface we are looking at. Ownership is session state,
        so a later verb keeps the surface the previous verb switched to.
        """
        return self._runtime.get("owner") or self.primary

    def _set_owner(self, name: str) -> None:
        """Record which driver owns the surface now in focus.

        Parameters
        ----------
        name : str
            Name of the driver that just opened the focused surface.

        Returns
        -------
        None
            This method mutates the session-shared runtime.

        Notes
        -----
        Ownership is stored in the session runtime rather than on the verb,
        because a later verb must not re-pick a background browser that still
        "sees" its page. The owner also anchors the ladder, which excludes
        drivers bound to a different surface.
        """
        self._runtime["owner"] = name

    async def _route(self) -> None:
        """Pick the perception driver that actually sees the focused surface.

        Probes drivers in `_route_order`, commits to the first usable one,
        rebuilds the ladder, and journals the decision.

        Returns
        -------
        None
            This method mutates `primary`, `ladder_order` and the routed flag.

        Notes
        -----
        Surface-agnostic rungs are bound to the owner BEFORE probing, or
        vision would grab the whole screen instead of the owned surface, and
        would then act outside the owner's coordinate space. Probing failures
        are swallowed into an unusable score so one broken backend cannot
        abort routing. Blind is not the same as wrong surface: a driver that
        owns the surface escalates to vision rather than losing perception to
        a different-surface driver.
        """
        scores: List[CapabilityScore] = []
        chosen: Optional[str] = None
        # Bind surface-agnostic rungs to the owner BEFORE probing them, or
        # vision would grab the whole screen instead of the owned surface.
        owner_name = self.surface_owner
        if owner_name in self.drivers:
            await self._ensure_started(owner_name)
            self._bind_surface_agnostic(self.drivers[owner_name])
        for name in self._route_order():
            try:
                await self._ensure_started(name)
                obs = await self.drivers[name].observe()
                cs = CapabilityScore.from_observation(name, obs)
            except Exception:
                cs = CapabilityScore(driver=name, usable=False)
            scores.append(cs)
            if cs.usable:
                chosen = name
                break  # first driver that sees the surface wins
        if chosen:
            self._set_primary(chosen)
        self._rebuild_ladder()
        self._routed = True
        self.journal.append(
            "route", primary=self.primary, ladder=self.ladder_order,
            scores={s.driver: round(s.score, 3) for s in scores},
        )

    def _rebuild_ladder(self) -> None:
        """Rebuild the action ladder around the currently focused surface.

        The ladder is the primary driver first, then fallback rungs that can
        act on the SAME surface, then surface-agnostic rungs.

        Returns
        -------
        None
            This method mutates `ladder_order` and rebinds agnostic rungs.

        Notes
        -----
        The ladder excludes drivers bound to a different surface (dom versus
        tree), since such a driver would act on a background window rather
        than the focused one. Surface compatibility is defined by the OWNER,
        not by whichever driver happens to be perceiving: vision is
        surface-agnostic and would otherwise widen the ladder to unrelated
        surfaces. Surface-agnostic rungs (keyboard, vision, surface None)
        always ride along as last-resort rungs, bound to the owner's pixels
        and coordinate space.
        """
        # Surface compatibility is defined by the OWNER, not by whichever
        # driver happens to be perceiving (vision is surface-agnostic and
        # would otherwise widen the ladder to unrelated surfaces).
        owner_name = self.surface_owner if self.surface_owner in self.drivers else self.primary
        owner = self.drivers[owner_name]
        prim = getattr(owner, "surface", None)
        rest = [
            n for n in self.drivers if n != self.primary
            and getattr(self.drivers[n], "surface", None) in (None, prim)
        ]
        self.ladder_order = [self.primary] + rest
        self._bind_surface_agnostic(owner)

    def _bind_surface_agnostic(self, primary_driver: Driver) -> None:
        """Point surface-agnostic rungs at the focused surface.

        Parameters
        ----------
        primary_driver : Driver
            The driver that owns the focused surface, whose screenshot and
            pointer factories the agnostic rungs should borrow.

        Returns
        -------
        None
            This method mutates the agnostic rungs in `ladder_order`.

        Notes
        -----
        Surface-agnostic rungs are bound to the owner's pixels and coordinate
        space, so vision grounds on the focused surface instead of assuming
        an OS-global screen. An agnostic driver is never bound to itself or to
        another agnostic driver, since that would give it nothing concrete to
        aim at.
        """
        for name in self.ladder_order:
            driver = self.drivers[name]
            if getattr(driver, "surface", "x") is not None or driver is primary_driver:
                continue
            if getattr(primary_driver, "surface", "x") is None:
                continue  # never bind an agnostic rung to itself
            capture = getattr(primary_driver, "screenshot", None)
            if hasattr(driver, "set_capture") and capture is not None:
                driver.set_capture(capture)
            pointer_factory = getattr(primary_driver, "pointer", None)
            if hasattr(driver, "set_pointer"):
                driver.set_pointer(pointer_factory() if pointer_factory else None)

    def reroute(self) -> None:
        """Force re-selection of the perception driver on the next observe.

        Returns
        -------
        None
            This method clears the routed flag.

        Notes
        -----
        Only clears the flag; the surface owner is left intact, so the next
        route still starts from the driver that owns the focused surface.
        """
        self._routed = False

    async def _navigate(self, action: Action) -> ActionResult:
        """Open a URL or application and take ownership of the new surface.

        Parameters
        ----------
        action : Action
            The navigate action whose `value` names the URL or application
            to open.

        Returns
        -------
        ActionResult
            Result carrying whether the navigation landed, the driver used as
            strategy, and any error message as note.

        Notes
        -----
        The driver that CAN open this target decides the surface (dom opens
        URLs, tree launches apps), so each driver is asked rather than
        pattern-matching the target string here. Navigate cannot be routed by
        observation because the surface does not exist yet. Once it lands,
        that driver owns the surface and becomes primary: capability probing
        alone cannot tell focus, since after switching web to desktop the
        browser driver still "sees" its now-background page and would wrongly
        win. Only a genuine blind (SurfaceUnreadable) triggers re-routing.
        """
        target_name = self._pick_navigator(action.value)
        await self._ensure_started(target_name)
        try:
            await self.drivers[target_name].act(action)
            landed, note = True, ""
        except OrbitError as exc:
            landed, note = False, exc.message
        # The driver that just opened the surface is the one now in focus, so
        # trust it as primary. (Capability probing alone can't tell focus:
        # after switching web -> desktop, the browser driver still "sees"
        # its now-background page and would wrongly win.) Only a genuine
        # blind (SurfaceUnreadable) triggers re-routing.
        if landed:
            self._set_owner(target_name)   # this driver owns the new surface
            self._set_primary(target_name)
            self._rebuild_ladder()  # ladder now excludes the other surface
            self._routed = True
        else:
            self._set_primary(target_name)
            self._routed = False
        self.journal.append(
            "action", kind_="navigate", target=action.value, value=action.value,
            landed=landed, strategy=target_name, attempts=1, diff="navigated",
            errors=[note] if note else [],
        )
        return ActionResult(landed=landed, action=action, strategy=target_name, note=note)

    def _pick_navigator(self, target: Optional[str]) -> str:
        """Ask each driver whether it can open a target; the first yes wins.

        Parameters
        ----------
        target : str, optional
            The URL or application name to open. May be None, in which case
            drivers are still consulted and will normally decline.

        Returns
        -------
        str
            Name of the driver that claimed the target, or the current
            primary when no driver claimed it.

        Notes
        -----
        Drivers own the definition ("is this a URL?" lives in the dom driver,
        not in a hardcoded classifier here), which keeps target syntax
        knowledge next to the code that acts on it.
        """
        for name, driver in self.drivers.items():
            can = getattr(driver, "can_navigate", None)
            if can is not None and can(target):
                return name
        # No driver claimed it: fall back to the current primary.
        return self.primary

    async def act(self, action: Action) -> ActionResult:
        """Execute an action, falling back down the ladder as needed.

        Navigate actions are handled specially; everything else is routed (if
        needed) and then run through the fallback ladder, with the outcome
        journalled either way.

        Parameters
        ----------
        action : Action
            The action to perform on the focused surface.

        Returns
        -------
        ActionResult
            Result of the rung that handled the action, including whether it
            landed, the strategy used, attempt count and the observed diff.

        Raises
        ------
        TargetUnresolvable
            If no rung on the ladder could resolve the action's target. The
            failure is journalled with its code, message and context first.

        Notes
        -----
        Navigate opens a surface that does not exist yet, so it cannot be
        routed by observation: the driver that CAN open the target handles it
        and the next observe routes perception normally. The ladder passed to
        `run_ladder` excludes drivers bound to a different surface, and each
        rung is started only when actually reached. A navigation may switch
        the focused surface (browser to native app and back), so perception is
        forced to re-route on the next observe.
        """
        # Navigate opens a surface that doesn't exist yet, so it can't be
        # routed by observation. The driver that *can* open the target
        # handles it; the next observe then routes perception normally.
        if action.kind is ActionKind.NAVIGATE and self.auto_route:
            return await self._navigate(action)

        if self.auto_route and not self._routed:
            await self._route()
        await self._ensure_started(self.primary)  # observe_via / first rung
        ordered = [self.drivers[n] for n in self.ladder_order if n in self.drivers]
        try:
            outcome = await run_ladder(
                ordered, action,
                observe_via=self.drivers[self.primary],
                ensure=self._ensure_started,  # start each rung only when reached
            )
        except TargetUnresolvable as exc:
            self.journal.append(
                "action_failed",
                kind_=action.kind.value, target=action.target,
                error=exc.code, message=exc.message, context=exc.context,
            )
            raise
        result = outcome.result
        self.journal.append(
            "action",
            kind_=action.kind.value, target=action.target, value=action.value,
            landed=result.landed, strategy=result.strategy,
            attempts=result.attempts, diff=result.diff.summary(),
            errors=[str(e) for e in outcome.errors],
        )
        # A navigation may switch the focused surface (browser <-> native
        # app); force perception to re-route on the next observe.
        if self.auto_route and action.kind is ActionKind.NAVIGATE:
            self._routed = False
        return result

    async def probe(self, surface: str) -> List[str]:
        """Score every driver against a surface and reorder the ladder.

        Parameters
        ----------
        surface : str
            Name under which the resulting scores are cached.

        Returns
        -------
        List[str]
            Driver names sorted best first, which is also the new
            `ladder_order`.

        Notes
        -----
        Drivers that raise while observing are scored as unusable rather than
        aborting the probe, so one broken backend cannot hide the rest. This
        is a whole-set scoring pass, distinct from `_route`, which commits to
        the first usable driver in owner-first order.
        """
        scores: List[CapabilityScore] = []
        for name, driver in self.drivers.items():
            try:
                obs = await driver.observe()
                scores.append(CapabilityScore.from_observation(name, obs))
            except Exception:
                scores.append(CapabilityScore(driver=name, usable=False))
        scores.sort(key=lambda s: s.score, reverse=True)
        self.probe_cache[surface] = scores
        self.ladder_order = [s.driver for s in scores]
        self.journal.append(
            "probe", surface=surface,
            order=self.ladder_order,
            scores={s.driver: round(s.score, 3) for s in scores},
        )
        return self.ladder_order
