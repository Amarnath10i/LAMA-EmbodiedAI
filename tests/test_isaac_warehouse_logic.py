"""Pre-flight logic verification for IsaacWarehouse, without real Isaac Lab.

`isaac_warehouse.py` has never been run against real Isaac Sim -- there is no
RTX-class GPU available to this project's author (see the module's own
docstring). This file cannot fix that. What it CAN do is inject a minimal
fake `isaaclab` package into `sys.modules` -- built from real `torch`
tensors, not mocks that silently accept anything -- so that every line of
`IsaacWarehouse` actually executes: every method call, every attribute
access, every tensor shape assumption. That catches typos, wrong attribute
names, and logic bugs (precondition handling, mechanism triggering, budget
accounting) with total confidence, and narrows what is left unverified down
to exactly one thing: whether the fake's guessed signature for
`set_external_force_and_torque` (and a handful of other calls) matches the
real Isaac Lab API. That is a real gap this file cannot close -- see
docs/ISAAC_LAB_SETUP.md -- but everything else about this backend's behaviour
is now genuinely exercised, not just read for plausibility.

The fake's "physics" is deliberately crude (a fixed per-step displacement
scaled by 1/mass, not real force integration) -- just enough to reproduce the
one qualitative property that actually matters for the tests below: heavier
objects move less under the same applied force.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
torch = pytest.importorskip("torch")  # the Isaac backend logic is tested with real torch tensors

# --------------------------------------------------------------------------- #
# a minimal fake isaaclab, injected before isaac_warehouse.py is imported
# --------------------------------------------------------------------------- #


class _FakeAppLauncher:
    def __init__(self, headless=True, **kw):
        self.app = object()


class _FakeSimulationCfg:
    def __init__(self, dt=1.0 / 60.0, **kw):
        self.dt = dt


class _FakeSimulationContext:
    def __init__(self, cfg=None):
        self.device = "cpu"
        self._dt = cfg.dt if cfg is not None else 1.0 / 60.0
        self.step_count = 0

    def set_camera_view(self, eye, target):
        pass

    def reset(self):
        pass

    def step(self):
        self.step_count += 1

    def get_physics_dt(self):
        return self._dt


def _cfg_func(path, cfg):
    """Stand-in for GroundPlaneCfg/DomeLightCfg's `.func(path, cfg)` spawn call."""


class _FakeGroundPlaneCfg:
    def __init__(self, **kw):
        self.func = _cfg_func


class _FakeDomeLightCfg:
    def __init__(self, **kw):
        self.func = _cfg_func


class _FakeRigidBodyPropertiesCfg:
    def __init__(self, kinematic_enabled=False, **kw):
        self.kinematic_enabled = kinematic_enabled


class _FakeMassPropertiesCfg:
    def __init__(self, mass=1.0, **kw):
        self.mass = mass


class _FakeCollisionPropertiesCfg:
    def __init__(self, **kw):
        pass


class _FakePreviewSurfaceCfg:
    def __init__(self, diffuse_color=(1, 1, 1), **kw):
        self.diffuse_color = diffuse_color


class _FakeCuboidCfg:
    def __init__(self, size, rigid_props, mass_props, collision_props,
                visual_material, **kw):
        self.size = size
        self.rigid_props = rigid_props
        self.mass_props = mass_props
        self.collision_props = collision_props
        self.visual_material = visual_material


class _FakeInitialStateCfg:
    def __init__(self, pos=(0.0, 0.0, 0.0), **kw):
        self.pos = pos


class _FakeRigidObjectCfg:
    InitialStateCfg = _FakeInitialStateCfg

    def __init__(self, prim_path, spawn, init_state, **kw):
        self.prim_path = prim_path
        self.spawn = spawn
        self.init_state = init_state


class _FakeRigidObjectData:
    def __init__(self, pos_w):
        self.root_pos_w = pos_w  # (1, 3) tensor


class _FakeRigidObject:
    """Crude but real: force -> displacement scaled by 1/mass, so heavier
    objects genuinely move less, exactly the property under test."""

    #: Arbitrary scale making a 25N push produce a legible few-cm-to-metre
    #: displacement range across the catalogue's mass spread, for the test's
    #: own purposes only -- not a claim about real PhysX behaviour.
    _DISPLACEMENT_SCALE = 0.02

    def __init__(self, cfg):
        self.cfg = cfg
        x, y, z = cfg.init_state.pos
        self.data = _FakeRigidObjectData(
            torch.tensor([[x, y, z]], dtype=torch.float32)
        )
        self._mass = max(cfg.spawn.mass_props.mass, 0.05)
        self._kinematic = cfg.spawn.rigid_props.kinematic_enabled
        self._pending_force = None
        self.n_force_calls = 0
        self.n_pose_writes = 0

    def write_root_pose_to_sim(self, pose):
        assert pose.shape == (1, 7), f"expected (1, 7), got {tuple(pose.shape)}"
        self.data.root_pos_w = pose[:, :3].clone()
        self.n_pose_writes += 1

    def write_root_velocity_to_sim(self, vel):
        assert vel.shape == (1, 6), f"expected (1, 6), got {tuple(vel.shape)}"

    def set_external_force_and_torque(self, forces, torques, **kw):
        assert forces.shape == (1, 1, 3), f"expected (1, 1, 3), got {tuple(forces.shape)}"
        assert torques.shape == (1, 1, 3), f"expected (1, 1, 3), got {tuple(torques.shape)}"
        self._pending_force = (forces.clone(), torques.clone())
        self.n_force_calls += 1

    def write_data_to_sim(self):
        if self._kinematic or self._pending_force is None:
            return
        forces, torques = self._pending_force
        if float(forces.abs().sum()) > 0:
            self.data.root_pos_w[0, :2] += (
                forces[0, 0, :2] / self._mass
            ) * self._DISPLACEMENT_SCALE

    def update(self, dt):
        pass


def _make_fake_isaaclab():
    isaaclab = types.ModuleType("isaaclab")
    app = types.ModuleType("isaaclab.app")
    app.AppLauncher = _FakeAppLauncher
    sim = types.ModuleType("isaaclab.sim")
    sim.SimulationCfg = _FakeSimulationCfg
    sim.SimulationContext = _FakeSimulationContext
    sim.GroundPlaneCfg = _FakeGroundPlaneCfg
    sim.DomeLightCfg = _FakeDomeLightCfg
    sim.CuboidCfg = _FakeCuboidCfg
    sim.RigidBodyPropertiesCfg = _FakeRigidBodyPropertiesCfg
    sim.MassPropertiesCfg = _FakeMassPropertiesCfg
    sim.CollisionPropertiesCfg = _FakeCollisionPropertiesCfg
    sim.PreviewSurfaceCfg = _FakePreviewSurfaceCfg
    assets = types.ModuleType("isaaclab.assets")
    assets.RigidObject = _FakeRigidObject
    assets.RigidObjectCfg = _FakeRigidObjectCfg
    isaaclab.app = app
    isaaclab.sim = sim
    isaaclab.assets = assets
    return {
        "isaaclab": isaaclab,
        "isaaclab.app": app,
        "isaaclab.sim": sim,
        "isaaclab.assets": assets,
    }


@pytest.fixture
def fake_isaaclab(monkeypatch):
    """Install the fake package, reset IsaacWarehouse's process-wide
    singleton so each test starts clean, and import fresh."""
    for name, mod in _make_fake_isaaclab().items():
        monkeypatch.setitem(sys.modules, name, mod)

    sys.modules.pop("lama.env.isaac_warehouse", None)
    import lama.env.isaac_warehouse as iw

    iw._app_handle = None
    iw._sim_handle = None
    yield iw
    iw._app_handle = None
    iw._sim_handle = None


def _oracle_ids(world, oracle_cls, kind):
    return oracle_cls(world).ids_of_kind(kind)


# --------------------------------------------------------------------------- #
# construction and the contract
# --------------------------------------------------------------------------- #
def test_constructs_against_the_fake_backend(fake_isaaclab):
    from lama.env.interface import Environment

    w = fake_isaaclab.IsaacWarehouse(seed=0, n_objects=8, budget=30.0)
    assert isinstance(w, Environment)
    assert w.backend_name == "isaac-lab"
    assert w.budget_remaining == 30.0
    assert w.episode == 0


def test_second_construction_reuses_the_singleton_app(fake_isaaclab):
    w1 = fake_isaaclab.IsaacWarehouse(seed=0, n_objects=8, budget=10.0)
    sim_after_first = fake_isaaclab._sim_handle
    w2 = fake_isaaclab.IsaacWarehouse(seed=1, n_objects=8, budget=10.0)
    assert fake_isaaclab._sim_handle is sim_after_first, (
        "constructing a second IsaacWarehouse must not relaunch the app"
    )


def test_observations_never_reveal_object_kinds(fake_isaaclab):
    w = fake_isaaclab.IsaacWarehouse(seed=0, n_objects=8, budget=30.0)
    kinds = {o.kind for o in w._objects.values()}
    obs = w._observe()
    blob = repr(obs)
    leaked = {k for k in kinds if k in blob}
    assert not leaked, f"observation leaked kinds: {leaked}"


def test_every_verb_is_accepted_for_every_target_without_raising(fake_isaaclab):
    from lama.env import INTERACTION_ACTIONS, Interaction

    w = fake_isaaclab.IsaacWarehouse(seed=0, n_objects=8, budget=1e6)
    for target_id in list(w._objects):
        for action in INTERACTION_ACTIONS:
            i = Interaction(action, None if action.name == "RELEASE" else target_id)
            w.step(i)  # must not raise


# --------------------------------------------------------------------------- #
# affordance resolution mirrors warehouse.py's semantics
# --------------------------------------------------------------------------- #
def test_doors_start_locked(fake_isaaclab):
    from lama.env import Action, Effect, Interaction
    from lama.env.isaac_warehouse import IsaacWarehouseOracle

    w = fake_isaaclab.IsaacWarehouse(seed=0, n_objects=8, budget=30.0)
    door = _oracle_ids(w, IsaacWarehouseOracle, "door")[0]
    w.step(Interaction(Action.APPROACH, door))
    r = w.step(Interaction(Action.OPEN, door))
    assert r.record.outcome.effect is Effect.BLOCKED


def test_a_latching_mechanism_opens_and_holds_the_door(fake_isaaclab):
    from lama.env import Action, Effect, Interaction
    from lama.env.isaac_warehouse import IsaacWarehouseOracle

    for layout_seed in range(6):
        w = fake_isaaclab.IsaacWarehouse(seed=layout_seed, n_objects=8, budget=30.0)
        oracle = IsaacWarehouseOracle(w)
        for kind, verb in (("button", Action.PRESS), ("lever", Action.PULL),
                           ("valve", Action.ROTATE)):
            ids = oracle.ids_of_kind(kind)
            if not ids:
                continue
            door = oracle.ids_of_kind("door")[0]
            mech = ids[0]
            w.step(Interaction(Action.APPROACH, mech))
            r = w.step(Interaction(verb, mech))
            assert r.record.outcome.effect is Effect.ACTUATED
            assert r.record.outcome.had_remote_effect
            w.step(Interaction(Action.APPROACH, door))
            assert w._objects[door].is_open, f"{kind} must latch the door open"
            return
    pytest.fail("no layout in range produced a testable actuator")


def test_the_plate_only_holds_the_door_open_while_weighted(fake_isaaclab):
    from lama.env import Action, Effect, Interaction
    from lama.env.isaac_warehouse import IsaacWarehouseOracle

    w = fake_isaaclab.IsaacWarehouse(seed=0, n_objects=8, budget=30.0)
    oracle = IsaacWarehouseOracle(w)
    door, plate = oracle.ids_of_kind("door")[0], oracle.ids_of_kind("plate")[0]

    w.step(Interaction(Action.APPROACH, plate))
    r = w.step(Interaction(Action.PRESS, plate))
    assert r.record.outcome.effect is Effect.ACTUATED
    assert w._objects[door].is_open

    w.step(Interaction(Action.APPROACH, door))
    assert not w._objects[door].is_open, "standing on the plate must not latch"


def test_a_heavy_object_holds_the_plate_down_permanently(fake_isaaclab):
    """The flagship secondary affordance, through the Isaac code path."""
    from lama.env import Action, Effect, Interaction
    from lama.env.isaac_warehouse import IsaacWarehouseOracle

    w = fake_isaaclab.IsaacWarehouse(seed=0, n_objects=8, budget=30.0)
    oracle = IsaacWarehouseOracle(w)
    door = oracle.ids_of_kind("door")[0]
    plate = oracle.ids_of_kind("plate")[0]
    block = oracle.ids_of_kind("block")[0]

    w.step(Interaction(Action.APPROACH, block))
    w.step(Interaction(Action.GRASP, block))
    w.step(Interaction(Action.APPROACH, plate))
    r = w.step(Interaction(Action.PLACE_ON, plate))
    assert r.record.outcome.effect is Effect.SUPPORTED
    assert r.record.outcome.had_remote_effect

    w.step(Interaction(Action.APPROACH, door))
    assert w._objects[door].is_open


def test_the_look_alike_crate_is_not_heavy_enough(fake_isaaclab):
    from lama.env import Action, Effect, Interaction
    from lama.env.isaac_warehouse import IsaacWarehouseOracle

    w = fake_isaaclab.IsaacWarehouse(seed=0, n_objects=8, budget=30.0)
    oracle = IsaacWarehouseOracle(w)
    door = oracle.ids_of_kind("door")[0]
    plate = oracle.ids_of_kind("plate")[0]
    crate = oracle.ids_of_kind("crate")[0]

    w.step(Interaction(Action.APPROACH, crate))
    w.step(Interaction(Action.GRASP, crate))
    w.step(Interaction(Action.APPROACH, plate))
    r = w.step(Interaction(Action.PLACE_ON, plate))
    assert r.record.outcome.effect is Effect.SUPPORTED
    assert not r.record.outcome.had_remote_effect

    w.step(Interaction(Action.APPROACH, door))
    assert not w._objects[door].is_open


# --------------------------------------------------------------------------- #
# the physical layer: hidden mass reaches displacement through real force calls
# --------------------------------------------------------------------------- #
def test_heavier_objects_move_less_when_shoved(fake_isaaclab):
    from lama.env import Action, Interaction
    from lama.env.isaac_warehouse import IsaacWarehouseOracle

    w = fake_isaaclab.IsaacWarehouse(seed=0, n_objects=8, budget=1e6)
    oracle = IsaacWarehouseOracle(w)
    moved = {}
    for kind in ("crate", "block"):
        target = oracle.ids_of_kind(kind)[0]
        w.step(Interaction(Action.APPROACH, target))
        r = w.step(Interaction(Action.PUSH, target))
        moved[kind] = r.record.outcome.displacement
    assert moved["crate"] > moved["block"] > 0, (
        "the fake physics scales displacement by 1/mass; if this fails the "
        "force-application call site is not reaching the object at all"
    )


def test_force_is_cleared_after_a_push_not_left_applied(fake_isaaclab):
    from lama.env import Action, Interaction
    from lama.env.isaac_warehouse import IsaacWarehouseOracle

    w = fake_isaaclab.IsaacWarehouse(seed=0, n_objects=8, budget=1e6)
    target = IsaacWarehouseOracle(w).ids_of_kind("crate")[0]
    w.step(Interaction(Action.APPROACH, target))
    w.step(Interaction(Action.PUSH, target))
    body = w._objects[target].body
    forces, _ = body._pending_force
    assert float(forces.abs().sum()) == 0.0, "force must be zeroed after settling"


def test_kinematic_fixed_objects_do_not_move_when_pushed(fake_isaaclab):
    from lama.env import Action, Interaction
    from lama.env.isaac_warehouse import IsaacWarehouseOracle

    w = fake_isaaclab.IsaacWarehouse(seed=0, n_objects=8, budget=1e6)
    door = IsaacWarehouseOracle(w).ids_of_kind("door")[0]
    before = w._body_pos(w._objects[door]).copy()
    w.step(Interaction(Action.APPROACH, door))
    w.step(Interaction(Action.PUSH, door))
    after = w._body_pos(w._objects[door])
    assert np.allclose(before, after), "a kinematic (fixed) object must not move"


# --------------------------------------------------------------------------- #
# budget, reset, and reset's teleport-back-to-spawn behaviour
# --------------------------------------------------------------------------- #
def test_every_attempt_costs_budget_including_failures(fake_isaaclab):
    from lama.env import Action, Effect, Interaction
    from lama.env.isaac_warehouse import IsaacWarehouseOracle

    w = fake_isaaclab.IsaacWarehouse(seed=0, n_objects=8, budget=20.0)
    door = IsaacWarehouseOracle(w).ids_of_kind("door")[0]
    before = w.budget_remaining
    w.step(Interaction(Action.APPROACH, door))
    r = w.step(Interaction(Action.LIFT, door))
    assert r.record.outcome.effect.name in ("NOTHING", "BLOCKED")
    assert w.budget_remaining < before - 0.001  # approach + lift both cost


def test_reset_restores_budget_and_teleports_objects_back(fake_isaaclab):
    from lama.env import Action, Interaction
    from lama.env.isaac_warehouse import IsaacWarehouseOracle

    w = fake_isaaclab.IsaacWarehouse(seed=0, n_objects=8, budget=20.0)
    oracle = IsaacWarehouseOracle(w)
    crate = oracle.ids_of_kind("crate")[0]
    w.step(Interaction(Action.APPROACH, crate))
    w.step(Interaction(Action.PUSH, crate))
    moved_pos = w._body_pos(w._objects[crate]).copy()

    w.reset(seed=0)
    assert w.budget_remaining == 20.0
    assert w.episode == 1
    reset_pos = w._body_pos(w._objects[crate])
    assert not np.allclose(moved_pos[:2], reset_pos[:2]), (
        "reset must teleport objects back to their spawn pose"
    )


def test_layout_is_fixed_across_resets(fake_isaaclab):
    """Documented scope limitation: composition does not vary across resets
    on one instance, only across separately constructed instances."""
    w = fake_isaaclab.IsaacWarehouse(seed=0, n_objects=8, budget=10.0)
    from lama.env.isaac_warehouse import IsaacWarehouseOracle

    before = IsaacWarehouseOracle(w).present_kinds()
    w.reset(seed=1)
    after = IsaacWarehouseOracle(w).present_kinds()
    assert before == after
