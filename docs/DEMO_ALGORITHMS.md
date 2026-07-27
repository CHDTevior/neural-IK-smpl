# Demo algorithms: constraint mapping, no-jump interaction, and verification

This documents *how* the interactive demo (`demo/server.py` +
`demo/frontend.html`) works, at the level a developer needs to extend it or
port it: how browser constraints map onto the model's native tensor
interface, the FK/mesh decode path, the multi-aspect constraint model, the
no-jump interaction system, the network scheduling, the rotation-encoding
convention catch, and the numerical gates the whole path was verified
against. Running instructions and the HTTP API contract live in
[`../demo/README.md`](../demo/README.md).

Throughout, "upstream" refers to the pinned smpl-ik checkout;
`SmplModel.get_data_from_batch`, `pack_data`, `make_translation_invariant`,
`forward`, `shared_step`, and `apply_smpl_quat` are in
`smplik/models/smpl_model.py` of that checkout.

## 1. From UI constraints to the model's native interface

The network is trained on `input_data` dictionaries produced by
`SmplModel.get_data_from_batch`. The server does not adapt the model to the
demo; it reconstructs that exact dictionary from the JSON request
(`build_input_data` in `demo/server.py`), including details that are easy to
get subtly wrong:

**Per-type tensors.** Effectors are grouped by type into three parallel
blocks, each `[batch=1, N_type, ...]`:

| Key | Shape | Content |
|---|---|---|
| `position_data` | `[1, Np, 3]` | target point, meters, world frame |
| `rotation_data` | `[1, Nr, 6]` | ortho6d of the joint's **world** rotation (section 6/7) |
| `lookat_data` | `[1, Nl, 6]` | `[target_world(3), local_dir(3)]`, `local_dir` normalized with the repo's `normalize_vector` (eps 1e-5) — same normalization as training |
| `*_weight` | `[1, N]` | blending weight in `[0, 1]` |
| `*_tolerance` | `[1, N]` | tolerance, meters |
| `*_id` | `[1, N]` int64 | joint index `0..23` |
| `betas` | `[1, 10]` float32 | SMPL shape |
| `gender` | `[1, 1]` int64 | `0` male / `1` female / `2` neutral (the dataset's encoding; upstream `apply_smpl` treats "anything other than 0/1" as neutral, the server accepts exactly `{0, 1, 2}`) |

**Empty-shape conventions.** Unused categories are *not omitted* — upstream
`pack_data` unconditionally concatenates all three blocks (padding position
data to 6 columns, appending tolerance as a 7th), so an absent category must
be present as a zero-row tensor of the right rank: `[1, 0, 3]` / `[1, 0, 6]`
for data, `[1, 0]` for weight/tolerance, `[1, 0]` int64 for ids. The server
constructs these exactly as `get_data_from_batch` does.

**Translation invariance, and why >= 1 weighted position effector is
required.** Inside `model.forward`, `make_translation_invariant`
re-references the input around the **weighted centroid of the position
effectors**:

```
reference = sum_i(w_i * pos_i) / sum_i(w_i)
position_data -= reference ;  lookat targets -= reference
```

and the predicted joint positions are shifted back by `+ reference` on the
way out. The network therefore never sees absolute world coordinates — a
position-effector set defines the world frame. With zero position effectors
(or all weights zero) the centroid is `0/0`: there is no frame, and the
upstream training code itself guarantees at least one positional effector
"for translation invariance" (comment in `get_data_from_batch`). The server
mirrors that invariant as a fail-loud `400` rather than guessing a frame:

> `at least one position effector with weight > 0 is required (the model is
> translation-invariant around the weighted centroid of position effectors)`

Note `make_translation_invariant` runs *inside* `forward`; the server
deliberately does not duplicate it (a first-draft mistake worth warning
about: re-referencing on the server *and* in `forward` would shift the scene
twice).

## 2. Decoding a solve: FK and the SMPL LBS mesh

`model.forward` returns ortho6d **local** joint rotations and a root
position. The server decodes them to renderable geometry exactly as upstream
`shared_step` does for its FK metrics:

```
predicted["joint_rotations"]  [1, 24, 6]
  -> rotation_6d_to_matrix    (pytorch3d Gram-Schmidt)
  -> matrix_to_quaternion
  -> model.apply_smpl_quat(betas, quats, root_position=predicted root,
                           gender, predict_verts=True)
  -> fk_joint_positions [24, 3], fk_world_rotations [24, 3, 3], verts [6890, 3]
```

`apply_smpl_quat` selects the gendered SMPL body (male/female/neutral) per
the batch's `gender`, runs the kinematic chain, and with
`predict_verts=True` also skins the 6890-vertex SMPL mesh (linear blend
skinning) — that mesh is what the frontend renders, not an approximation.
The reply additionally includes
`matrix_to_rotation_6d(fk_world_rotations)`, i.e. the solved pose's world
rotations *in the same encoding the rotation-effector input uses*. That
self-consistency is load-bearing for the no-jump system (section 4, layer 1).

## 3. The multi-aspect constraint model

The frontend's unit of constraint is a **joint row**, not an effector: each
row (one per SMPL joint, at most one row per joint) carries up to three
**independent, simultaneously active aspects** — position, rotation, lookat
— each with its own target, weight, and tolerance. On the wire this changes
nothing: `buildSolvePayload` flattens every enabled aspect to one plain
effector object, so the server sees the same `{type, joint, ...}` list as
ever.

Why not the simpler "one effector row with a type dropdown"? Because
type-*switching* is semantically wrong, and visibly so. The concrete case
that killed it: pin the pelvis position and drag it down into a squat, then
try to *also* rotate the pelvis. With a type dropdown, switching the pelvis
effector from `position` to `rotation` **removes the position pin from the
payload** — the next solve is free to stand the body back up, and does. The
user's intent ("keep the squat, and now turn the hips") is two coexisting
constraints on one joint. The multi-aspect row expresses exactly that: the
position pin stays in the payload while the rotation aspect is manipulated.

UI mechanics (all in `demo/frontend.html`): each row shows three toggle
chips; a chip click enables a disabled aspect, makes an enabled aspect the
gizmo's manipulation target, or disables it if it already is the target.
Every enabled aspect gets its own marker in the 3D view (pin sphere /
orientation cube / look-at target sphere with a direction arrow), so a joint
with position + rotation shows both at once; clicking a marker selects that
row *and* that aspect.

## 4. The no-jump system

"No jump" = enabling, switching, or editing constraints must never visibly
teleport the body. Three cooperating layers:

**Layer 1 — seed new constraints from the current solved state.** When an
aspect is enabled it is initialized *already satisfied*:

- *position*: seeded at the joint's current solved position
  (`state.joints[joint]`; GT fallback before the first solve);
- *rotation*: seeded from the **`/solve`-returned world `rot6d`** of that
  joint (`state.solvedRot6d`, kept from the last reply; falls back to the
  sample's GT `gt_rot6d` from `/meta`, then identity). This is why the
  server echoes `rot6d` in every reply and why `/meta` carries `gt_rot6d`;
- *lookat*: the target is placed 0.6 m from the joint along the local
  direction axis rotated into world space by the joint's current world
  rotation — i.e. where the joint is *already* looking.

The same re-seeding runs when a row is switched to a different joint and
when the lookat local-axis preset changes.

**Layer 2 — lazy re-solve.** Because a freshly seeded constraint is
satisfied by construction, enabling it fires **no solve at all** — the pose
cannot move, so nothing is sent. The first actual edit (gizmo drag, numeric
input, weight/tolerance slider) fires the solve. Disabling an aspect *does*
re-solve (a constraint disappeared; the result is blended by layer 3), and
if the last weighted position aspect is disabled the server's `400` is shown
verbatim in the error banner — the client does not silently prevent or
reorder anything.

**Layer 3 — blended application.** Non-drag solve replies (enable/disable,
slider release, sample tweaks) are applied as a **180 ms cubic ease-out
blend** (`1 - (1-u)^3`) interpolating all 6890 mesh vertices and 24 joints
from the currently displayed pose to the reply. During an active gizmo drag
replies apply instantly — at an ~80 ms solve cadence the pose deltas are
small and responsiveness wins. Under `prefers-reduced-motion` the blend is
disabled entirely (instant application), honoring the OS accessibility
setting.

**The honest limit.** The model is **not idempotent**: feeding back
constraints extracted from the current pose does not reproduce that pose
exactly. The measured floor is the all-24-effector self-transfer — all 24 GT
joints given as position effectors still returns an 11.5 mm MPJPE pose
(section 8), and sparse constraint sets sit further from fixed-point.
Consequently, after enabling a new constraint the *next* solve (whatever
triggers it) lands on a slightly different body than the one on screen, even
if the user barely moved anything. Layers 2 + 3 turn that residual
adjustment into a deferred, 180 ms-smoothed settle instead of an
enable-time pop — it is *hidden well*, not eliminated. Claiming zero
adjustment would require an idempotent solver, which this model is not.

## 5. Network scheduling

**Trailing-edge 80 ms throttle.** Continuous interactions (gizmo drag, beta
sliders) request solves via `scheduleSolve(false)`:

```js
if (solveTimer) return;                                  // keep the pending timer
const wait = Math.max(0, 80 - (performance.now() - lastSolveAt));
solveTimer = setTimeout(() => { solveTimer = null; doSolve(); }, wait);
```

The load-bearing line is `if (solveTimer) return`. **The version we shipped
first got this wrong**: it *reset* the timer on every input event (classic
debounce — `clearTimeout` + new `setTimeout`). Pointer-move events arrive
every ~8-16 ms, far inside the 80 ms window, so during a continuous drag the
timer never expired: no solves while dragging, one solve when the pointer
stopped. It presented exactly as "the pose only updates when you release the
gizmo", which looks like a backend or gizmo bug and is purely a scheduling
bug. The fix is the trailing-edge *throttle* above: an armed timer is left
alone, so solves fire every ~80 ms *during* the drag, and the trailing edge
still catches the final position. Discrete actions (gizmo release, sample
load, retry button) bypass the throttle with `scheduleSolve(true)`.

**Stale-reply dropping.** Solves are async and the server serializes them,
so replies can return out of order relative to user intent. Every request
captures a monotonically increasing sequence number; a reply is applied only
if its number exceeds `lastAppliedSeq` (which it then becomes). A slow old
solve that lands after a newer one is dropped instead of snapping the body
back to a stale pose. Error replies do not consume sequence numbers, and an
in-flight counter drives the "busy" latency chip.

## 6. Gizmo math

The shipped frontend delegates manipulation to three.js
`TransformControls` (vendored, r160): each enabled aspect's marker mesh is a
proxy object; the gizmo attaches to the selected marker in `translate` mode
(position target, lookat target) or `rotate` mode (rotation aspect), and
`objectChange` events copy the proxy's transform back into the constraint
state. Picking is a `Raycaster` over the marker group using pointer NDC
coordinates, registered *after* the controls so gizmo handles win; the
raycaster's line-pick threshold is dropped from its 1 m default to 0.02 m so
look-at arrow lines cannot grab distant clicks.

Two pieces of math are the demo's own:

**quaternion -> rot6d rows from column-major elements.** three.js
`Matrix4.elements` is **column-major**; pytorch3d's rot6d is the first two
**rows**. So the extraction is

```js
const e = new THREE.Matrix4().makeRotationFromQuaternion(q).elements;
return [e[0], e[4], e[8],  e[1], e[5], e[9]];   // row 0, row 1
```

(`e[0], e[4], e[8]` is the first *row* precisely because the array is
column-major.) The inverse, `rot6dToQuat`, re-implements pytorch3d's
`rotation_6d_to_matrix` Gram-Schmidt — `b1 = norm(a1)`,
`b2 = norm(a2 - (b1.a2) b1)`, `b3 = b1 x b2` — and writes `b1/b2/b3` as the
**rows** of a `Matrix4` (whose `set()` takes row-major arguments) before
converting to a quaternion.

**Camera-parallel-plane dragging (first-generation renderer).** The demo's
first frontend was a raw-WebGL renderer with hand-rolled effector dragging;
the three.js rewrite replaced it with `TransformControls`, but the technique
is worth recording because it is the correct free-drag primitive: on grab,
project the effector to clip space and keep its NDC depth `z` and clip `w`;
on every move, rebuild the clip-space point from the new pointer NDC `x, y`
and the *stored* `z, w`, and unproject through the inverse
projection-view matrix:

```
clip = [ndcX * w, ndcY * w, ndcZ_grab * w, w]
world = PV^-1 * clip  (then divide by the resulting w)
```

Holding NDC depth fixed confines the motion to the plane through the grab
point parallel to the camera plane — the drag follows the cursor exactly,
with no axis gymnastics, at any camera angle.

## 7. The rot6d rows-vs-columns catch

The single most dangerous convention in this stack. "6D rotation
representation" (Zhou et al., CVPR 2019) is described in the literature as
"the first two *columns* of the rotation matrix", and widely implemented
that way. **pytorch3d — which is what this model trained on — uses the
first two ROWS**: `matrix_to_rotation_6d` returns
`matrix[..., :2, :]` flattened, and `rotation_6d_to_matrix` stacks the
Gram-Schmidt frame as rows (`torch.stack((b1, b2, b3), dim=-2)`).

During development the two rot6d builders — the server-side `gt_rot6d` meta
builder and the frontend's quaternion converter — were written independently
and **disagreed**: one produced rows, the other columns. The disagreement
was arbitrated by reading the pytorch3d source (not either builder's
docstring): rows. Both were standardized on rows and the convention is now
pinned in three places: the server docstring, the `/solve` API contract, and
the frontend converter comments.

The bug this prevents is nasty because it half-works: columns-instead-of-rows
delivers `R^T = R^-1` — every rotation constraint applies the *inverse* world
rotation. Identity (`[1,0,0,0,1,0]`) is self-inverse, and so are all 180-degree
flips, so casual testing passes; generic targets come out mirrored/reversed
in a way that reads as "the model is bad at rotations" rather than "the
client is transposing". If you write a new client, send
`matrix_to_rotation_6d`-of-world-rotation semantics (rows), and round-trip a
`/solve` reply's `rot6d` back as a rotation effector as your first test — a
transposed client visibly twists the body on that test.

## 8. Verification gates

The demo path is gated on reproducing the offline reference, not on looking
plausible. `run_demo_inference.py` produces `demo_data.json` with the
reference outputs; a test client that drives **only the public HTTP API** (no
repo imports) then checks the server against it. Gates and measured results
for the released checkpoint (canonical epoch 13; server CPU, 4 torch
threads):

1. **Offline reference reproduction (8 samples, fixed 6-effector scheme).**
   Server `/solve` joints vs the offline pipeline's stored predictions,
   tolerance 1e-3 m per joint. Measured: worst per-joint difference across
   all 8 samples x 24 joints = **0.159 mm** (consistent with the reference
   being stored at 4-decimal rounding, i.e. 0.1 mm quantization, plus
   float32 non-associativity between the two code paths). Per-sample
   pelvis-aligned MPJPE 14.25-45.34 mm, mean 25.11 mm on these samples;
   the checkpoint's val-random MPJPE is 58.39 mm (paper: 59.3).
2. **All-24-effector self-transfer (sample 211).** Every GT joint given as a
   position effector. Measured: pelvis-aligned MPJPE **11.502 mm — exactly
   matching the offline reference value** (gate tolerance 0.1 mm; per-joint
   position match vs the stored prediction < 1 mm). This number is also the
   measured non-idempotency floor cited in section 4.
3. **Rotation-effector path.** 3 position effectors (pelvis + ankles) + GT
   world `rot6d` on pelvis and both wrists: output must be finite and sane
   (all |coordinates| < 10 m); wrist position error vs GT is reported (no
   wrist position effector is given, so it is not gated to zero).
4. **Lookat path.** 3 position effectors + a head lookat with the default
   `[0,0,1]` local axis: finite/sane output, head moves toward the target
   while the pinned pelvis stays put.
5. **Malformed requests fail loudly.** A rotation-only request must return
   exactly `400` with the position-effector error JSON — verified, along
   with the server's startup gate: it refuses to boot if the live test-set
   GT deviates from `demo_data.json` by > 1e-3 m (measured deviation
   5.0e-5 m, i.e. the reference's own 4-decimal rounding).

Beyond the numeric gates, the interaction on top was validated visually
(drag sweeps, squat-then-rotate, enable/disable cycling under throttle) —
per this project's standing rule that for CV systems an accurate visual demo
outranks any metric.
