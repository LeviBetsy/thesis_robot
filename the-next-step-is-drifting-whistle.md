# MCL with a blind-spot-aware sensor model

## Context

The next milestone is Monte Carlo Localization with a particle filter. The blocker is that
the camera (mounted tilted down, in front) has a minimum range: anything closer than
`max_calibrated_rel` in [scale_calibration.py](app/perception/scale_calibration.py) is not
detected, and [mde_depth.py](app/perception/mde_depth.py) `pcd_to_ray_casting` then reports
`max_range`. Standing right in front of a wall looks byte-identical to an empty room. The
question was whether this can be fixed in software rather than by adding an ultrasonic sensor.

**It can, mostly.** Investigation found the near-field failure is only partly geometric. Three
separate defects make it catastrophic instead of graceful, and all three are software:

**1. DAV2's output is sigmoid-squashed and nothing undoes it.** `ai_models/DAV2/metric_depth/depth_anything_v2/dpt.py:113`
is `nn.Sigmoid()` and line 183 multiplies by `max_depth=20.0`, but `scripts/mde/DAV2_pth.py`
loads the *relative* checkpoint into that *metric* head. So `d_rel = 20·sigmoid(y)`. Measured on
`data/test/rel_depth_test.npz` at the 96 calibration pixels: across the **bottom 80 image rows
`d_rel` spans 0.0021 units** out of a 0–20 range. There is essentially no numerical signal in
the near field. Applying `y = logit(d_rel/20)`:

| fit | max depth error | rms |
|---|---|---|
| raw `d_rel`, single global line | 242 mm | — |
| `logit(d_rel/20)`, single global line | **12.3 mm** | **2.1 mm** |

A single global line in logit space beats the current 15-segment piecewise fit. This is the
highest-leverage change in the plan and it is about four lines of code.

**2. The extrinsics in [robot.py:22-28](app/robot_module/robot.py#L22-L28) are wrong.** For a
ground plane `1/Z` is exactly affine in image row `v`. Least-squares on the 96 samples in
`config/z_real_16group.npz` gives `1/Z = 0.0160985·v + 0.2326199` with **1.1 mm rms** depth
residual — the npz is trustworthy — and implies **tilt 26.504°, height 0.1214 m**, not 30°/0.11 m.
Pushing the calibration points through the current `cam_R`/`cam_t` puts the floor at
`z_r ∈ [-0.042, -0.016]` instead of 0; with the fitted values, `z_r ∈ [-0.0008, +0.0016]`.
Every range out of `pcd_to_ray_casting` is currently ~3–4% short, which MCL would read as a
pose bias.

**3. The per-frame calibration is self-defeating.** When a wall covers the floor sample points,
the monotonicity filter kills all groupings, `min=100 / max=0`, and the **entire** depth map
becomes `-1` — not just the wall. That is the observed failure.

Corrected geometry (all from the fitted plane):

| | value |
|---|---|
| Bottom image row (v=479) hits floor | **0.150 m** from robot center — true geometric near limit |
| Nearest calibrated sample | 0.156 m from robot center |
| Farthest calibrated sample | 0.587 m (≈ `max_range=0.6`) |
| Per-ray near limit `R_MIN` | 0.1501 (center) … 0.1575 m (edge rays) |

**Honest scope:** below ~0.15 m the blind spot is geometric and no software recovers a range —
the correct output there is a `TOO_CLOSE` flag, which is still highly informative. Between
0.15 m and 0.25 m, where behaviour is currently catastrophic, software fully recovers it.

## Approach

Two independent range channels, fused, plus an MCL measurement model that treats "no return"
as evidence rather than missing data.

- **MDE metric channel** (existing, repaired): logit-space calibration with graceful degradation.
- **Floor-boundary channel** (new): scale-free. On the floor, the disparity logit `Y` follows a
  known template `T(v)` fixed by the ground-plane fit. Subtract it; obstacles appear as a
  positive residual jump. The contact row maps to metric range through inverse perspective
  mapping using only `(m, b0)` and the intrinsics — **no MDE scale enters**, so this channel
  survives exactly the frames where the metric calibration collapses.

Verified on `data/test/rel_depth_test.npz` (excluding the 149 obstacle columns):

| | value |
|---|---|
| open-floor residual, rows 200–479 | rms **0.029**, max 0.110 |
| obstacle peak residual | **2.764** |
| false breaks at τ=0.20, k=4, v_top=100 | 5 / 481 clean columns (~1%) |
| false breaks at τ=0.20, k=4, v_top=200 | 0 / 481 |

## Phase 0 — Foundation

**`scripts/camera/calibrate_ground_plane.py`** (new, offline). Fit `1/Z = m·v + b0` over the
96 floor samples, and `1/Z = a·logit(d_rel/20) + b` on a clean reference frame. Write
`config/ground_plane.npz` with `m, b0, tilt_rad, cam_height, a_seed, b_seed, max_depth`.
Reference values: `m=0.0160985, b0=0.2326199, tilt=0.46259 rad, h=0.12137, a_seed=0.8984, b_seed=0.4260`.
Only `(m, b0)` is observable — `tilt` and `cy` are exactly degenerate for a plane — so derive
tilt/height from it purely to populate `cam_R`/`cam_t`.

**`app/perception/ground_plane.py`** (new, numpy-only, shared Pi/laptop). `class GroundPlaneModel`:
- `pixel_to_ground(u, v)` → floor point `(X_right, Y_fwd)` from the camera:
  `s=(v-cy)/fy; den=sin(t)+s·cos(t); Y=h(cos t − s·sin t)/den; X=(u−cx)/fx·h/den`
- `ground_to_bearing_range(X, Y)` — must match `pcd_to_ray_casting` exactly:
  `bearing=arctan2(X,Y)`, `d_cam=hypot(X,Y)`, `r=sqrt(d_cam²+ct²+2·d_cam·ct·cos(bearing))`
  (law of cosines, confirmed correct in the existing code)
- `ray_min_ranges(n_rays=16)` → the `R_MIN` array above.

**[robot.py](app/robot_module/robot.py)** — default `camera_tilt=-26.504`, add `cam_height=0.1214`,
`cam_forward=0.07`, `self.radius=0.095`. Keep them keyword args so the old behaviour is one
argument away. Also raise `MDE_Depth.ground_Z` from 0.01 → **0.02**: with corrected extrinsics
the floor sits at ~0 ± 2 mm, leaving the old threshold only 8 mm of margin against MDE noise.

**[map.py](app/localization/map.py)** — fix the confirmed `add_wall` off-by-one (it uses
`int(x1 // cell_size)` for the constant axis while `world_to_grid` adds the perimeter offset;
for x=0.895 that is col 35 vs 36). Change both instantiations to
`OccupancyGrid(1.075, 1.775, 0.025)` → 45×73 (the current 1.07/1.78 rounds the arena 2 cm long).
Add:
- `distance_field()` — brute force over ~3.3k cells × ~300 obstacle cells, cached, no scipy
- `blocked_mask(radius)` — `distance_field() < radius`, for the collision prior
- `ray_cast_batch(px, py, ptheta, bearings, max_range=0.6, step=0.0125)` — vectorized
  fixed-step cast for N particles × K bearings. Cast from the **camera**
  `(px + cf·cos θ, py + cf·sin θ)` along world heading `θ − bearing`, then convert back to a
  robot-center range with the same law of cosines the observation uses. Use float32 + flat
  indexing (`idx = gy·cols + gx`, `np.argmax` over the step axis). Benchmarked 2.38 ms at
  N=500, K=16 → ~15 ms on a Pi 4. **A precomputed likelihood field is not needed** — `z_exp`
  is required for the no-return term anyway.

## Phase 1 — Logit-space calibration ([scale_calibration.py](app/perception/scale_calibration.py))

Add `to_logit(d_rel, max_depth=20.0)` / `from_logit`, and run the *entire* calibration on `y`.
Replace the 15-segment piecewise fit with **one global line** via `np.polyfit` — this also drops
the `sklearn` dependency, making the module importable on the Pi. Keep `self.fits` shaped `(n,2)`
with `n=1` so `relative_to_metric` needs no structural change; keep the old attribute names as
aliases.

Graceful degradation, driven by `n_blocks` = surviving blocks of 16:

| `n_blocks` | action | `conf` |
|---|---|---|
| ≥ 8 | fresh fit, update EMA | 1.0 |
| 3–7 | blend `λ·fresh + (1−λ)·EMA`, `λ = n_blocks/8` | `n_blocks/8` |
| 0–2 | EMA only, `stale += 1` | `0.5 · 0.8^stale` |
| `stale > 5` | emit no metric depth; all MDE rays `NO_RETURN` | 0.0 |

EMA the fit's **values at two fixed anchor logits** (`y=2.0` and `y=7.0`, ≈0.52 m and ≈0.16 m),
not `(a, b)` directly — slope and intercept are strongly anticorrelated and an EMA on them
wanders along the degenerate direction. Sanity-gate a fresh fit: reject if `a ≤ 0`, if implied
depth at either anchor is outside [0.05, 3.0] m, or if `|a − a_ema|/a_ema > 0.5`.

Fix `relative_to_metric`: the current `extrapolate=True` drops the *lower* bound, so it
extrapolates the **far** side (opposite of the near-field intent), and then `searchsorted(...)−1
== −1` applies `fits[-1]`, the *nearest* segment, to the *farthest* pixels. Change the parameter
to `extrapolate = False | 'far' | 'near' | 'both'` and correct the segment selection.

## Phase 2 — Floor-boundary ranging (`app/perception/floor_boundary.py`, new)

```
T(v)   = (m·v + b0 − b_seed) / a_seed        # expected floor logit, precomputed (480,)
Y(u,v) = logit(d_rel / 20)
R(u,v) = Y − g·T(v) − c_u                    # ≈0 on floor, jumps POSITIVE on obstacles
```

`class FloorBoundaryRanger`:
- `estimate_frame_affine(Y)` — robust `(g, c)` per frame: ~5 iterations of trimmed LSQ of `Y`
  on `T` over rows 320–479, keeping `|resid| < 3·MAD`. Measured: `g=0.989, c=0.066`,
  inlier fraction 0.977.
- `column_offsets(Y, g)` — `c_u` = median over seed rows (430–480) of `Y − g·T`. Per-column
  offsets are what take the open-floor residual down to rms 0.029.
- `find_breaks(Y, g, c_u)` — scan each column bottom-up, declare a break at the first row where
  `R > τ` for `k_consec` consecutive rows. Seeds: `τ=0.20`, `k_consec=4`, 5×5 median prefilter.
- `too_close_columns(c_u, c_global)` — if the **seed band itself** reads nearer than the floor
  (`c_u − c_global > τ_seed`), the obstacle is inside the blind cone → `TOO_CLOSE`. The opposite
  sign → `INVALID` (vignette / undistort border).
- `breaks_to_scan(...)` — map each break to a ground point, then bin **by bearing, not by column**.

Three design points that differ from the obvious implementation:

1. **Bin by bearing, not by column.** `pcd_to_ray_casting` bins the *ground* bearing. At the
   bottom row the ground bearing at u=0 is **−48.9°**, outside `±fov_x/2 = ±35.0°` — so the
   bottom image corners are already silently dropped by the existing binner. Column-to-ray
   mapping is not one-to-one and the two channels would not be comparable.
2. **Do not take a strict `min` per bin.** False breaks bias *short*, and at `v_top=100` about
   1% of clean floor columns produce one (measured: 5/481). With ~40 columns per bin, a strict
   min lets a single bad column corrupt a bin. Use a low percentile (start at the 20th) over the
   contributing columns, or require ≥3 agreeing columns. **Tune this against `v_top`** — the
   tradeoff is measured: `v_top=200` gives zero false breaks but caps range at 0.33 m, while
   `v_top=100` reaches 0.62 m at ~1% false-break rate.
3. **Template subtraction, not a per-column line fit.** A per-column fit re-estimates 2
   parameters from data that may be mostly obstacle; the template fixes the slope from geometry
   and estimates one offset from the seed band, which is the part most likely to be floor.

Sigma: `σ_f(r) = sqrt((3·|dr/dv|(v*))² + (0.02r)² + 0.008²)` ≈ 0.009 m at 0.15 m, 0.021 m at 0.6 m.
Restrict the scan to `v ≥ v_top` — row-to-range sensitivity is 0.29 mm/px at v=479 but 5.3 mm/px
at v=100, so above that a 1-px error costs centimetres.

## Phase 3 — Protocol

**`app/perception/range_scan.py`** (new). Per-ray `(range float32, sigma float32, status uint8,
source uint8)` = 12 B, plus a 40 B header with a magic word, `seq`, `t_capture`, `calib_conf`,
`fov_x`. Total 232 B. Statuses: `VALID / NO_RETURN / TOO_CLOSE / INVALID`.
`RangeScan.pack()/unpack()`, plus `from_legacy(arr)` so a bare float32 array still decodes.

**[zmq_stream.py](app/stream/zmq_stream.py)** — add an optional 16 B header to `VideoStreamer`
(`magic, seq, t_capture`) and `send_scan`/scan-decoding to the range pair. Sniff the magic word
on both channels so old-Pi/new-laptop and new-Pi/old-laptop combinations still run.

**Use `seq`, not clocks, for correspondence.** The Pi and laptop share no clock. The Pi assigns
a monotonic `seq` per captured frame and records `seq → cumulative encoder totals` in a ring
buffer; the laptop echoes `seq` back untouched. `t_capture` rides along for latency telemetry only.

**[mde_depth.py](app/perception/mde_depth.py)** — `frame_to_scan(frame, seq, t_capture)`.
The MDE channel must report `NO_RETURN` for an empty bin instead of pre-filling `max_range`
([mde_depth.py:61](app/perception/mde_depth.py#L61)) — that pre-fill is precisely what makes
"against a wall" and "empty room" identical. `fuse_channels`: floor channel is authoritative on
`TOO_CLOSE`; both valid and agreeing within 3σ → inverse-variance fuse; both valid but
disagreeing → take the smaller with inflated σ; one valid → take it; neither → `NO_RETURN`.
No hard range switch is needed — `σ_f` and `σ_m` cross over naturally.

**[laptop/mde_projection/main.py](laptop/mde_projection/main.py)** — ~4 lines. Note line 27
`if frame:` raises `ValueError: truth value of an array...` on the first frame; `real_time_stream()`
cannot currently run. Fix to `if item is not None:`.

## Phase 4 — MCL on the Pi

**`app/localization/motion_model.py`** — `class OdometryMotionModel`. Sampled differential-drive,
driven by raw `(dl, dr)` in metres. Noise on the wheel increments where it physically lives:
`σ_l = α₁|dl| + α₂|dr−dl| + ε` (α₁=0.10, α₂=0.05, ε=0.5 mm ≈ half an encoder count; the wheel
is 0.6109 mm/count). Plus a **static per-particle effective wheelbase** `w_i = w·(1+N(0,0.02²))`,
drawn once at init and carried through resampling — wheelbase error is the dominant systematic
in differential drive and produces heading drift proportional to total turning, which per-step
iid noise cannot represent. Same midpoint integrator as
[odometry.py:39-41](app/localization/odometry.py#L39-L41), so with noise zeroed the particle
mean reproduces raw odometry exactly.

**`app/localization/sensor_model.py`** — `class BlindSpotBeamModel`. The core of the fix.

- **VALID**: standard hit/short/rand mixture (`w_hit=0.80, w_short=0.10, w_rand=0.10`), with the
  max-range delta *deleted* — a max-range reading is now reported as `NO_RETURN`. Use the
  per-ray `sigma` from the wire, not a global `σ_hit`; that is the point of shipping it.
- **NO_RETURN** — the key term. With `S` the logistic:
  ```
  q(z_exp) = 1 − S((z_exp − r_min[k])/w_lo) · S((r_far − z_exp)/w_hi)
  p_k      = p_nr_floor + (1 − p_nr_floor)·q(z_exp)
  ```
  `q → 1` when `z_exp < r_min` (surface inside the blind cone) **or** `z_exp > r_far`, and
  `q → 0` in between. So a no-return is *informative*: it rules out the mid-band, and it no
  longer penalizes a particle correctly pressed against a wall. `p_nr_floor=0.15` gives a
  ~6.7:1 likelihood ratio per ray. Edges are deliberately asymmetric: `w_lo=0.02 m` (sharp — the
  near limit is a hard geometric fact known to ±7 mm) vs `w_hi=0.05 m` (soft — the far limit is
  a detectability limit that degrades gradually as `dr/dv` blows up).
- **TOO_CLOSE**: `p_k = p_tc_lo + p_tc_hi·S((r_min[k] + 0.02 − z_exp)/w_lo)` — ~17:1, the
  sharpest observation the sensor produces, which is right for a near-binary geometric fact.
- **INVALID**: `p_k = 1`.
- Combine in log space with a floor at `1e-3`, then **`logw = α·Σ log p` with `α = 0.5`**. This
  is not optional: 16 rays from one depth map at adjacent bearings are strongly correlated, and
  treating them as independent collapses `N_eff` to 1–2 within a few frames. Tune `α ∈ [0.3, 0.7]`
  targeting steady-state `N_eff` of 0.3–0.6·N.

**`app/localization/particle_filter.py`** — `class ParticleFilter`. N=500. Gaussian init around
the known start pose (tracking only, no global seeding). Collision prior: `-inf` weight where
`blocked_mask(0.095)` is true — `robot_radius=0.095 m` is an **assumption** (wheelbase 0.122 →
wheel centers at ±0.061, plus overhang and the camera mast 0.07 m forward); measure the widest
and longest half-extents and set it to the larger. Systematic (low-variance) resampler, only
when `N_eff < N/2`, carrying the per-particle wheelbase; roughening scaled by `N^(-1/3)`.
Injection: **do not inject uniform particles** — in a 1.9 m² arena they land near
plausible-but-wrong poses and steal weight. Inject 1% as Gaussians around the current mean with
inflated spread (0.05 m, 8°). `estimate()` uses a circular mean for θ.

**`app/localization/mcl_localization.py`** — the glue. `mark_frame(seq)` snapshots odometry
totals at capture; `on_scan(scan)` looks up `seq`, predicts on the delta since the last
*processed* capture, updates, resamples, then applies the capture→now motion **noiselessly to
the mean estimate only** (the particle set must stay at capture time). Publishes into `Robot`
under `mutex_lock` and via `PoseStreamer`.

**[odometry.py](app/localization/odometry.py)** — additive: cumulative `dl_total`/`dr_total`
accumulators and a `totals()` accessor. Move the `dl`/`dr` computation **inside** the mutex
(currently at lines 33-34, outside) — once MCL writes the pose back, an encoder packet landing
between the snapshot and the write would be silently lost.

**[test/mde_localization/mde_localization.py](test/mde_localization/mde_localization.py)** —
instantiate the grid + `MCLLocalization`, replace the empty `callback_new_pcd` stub (line 44-46)
with `mcl.on_scan(scan)`, and add `seq` / `mark_frame` to the capture loop.

Budget at N=500, 3 Hz on a Pi 4: ray cast ~15 ms, sensor model ~2 ms, prior <0.5 ms, resample
~1 ms → **~19 ms, ~6% of one core**.

## Phase 5 — Visualization and verification

**Visualization.** `MapVisualizer.update(data)` already takes an override array, so no changes
to [visualize_map.py](app/localization/visualize_map.py) — build the overlay on the laptop from
a `PoseReceiver` feed: grid as base, ≤200 subsampled particles at 0.35 grey, estimate cell at
0.85, a 3-cell heading stub, and the 16 rays drawn from the estimate (that is how you eyeball
whether the observed scan and the map ray cast agree). **Use port 8081** — `MapVisualizer` binds
`0.0.0.0:8080` and `RobotController` binds `127.0.0.1:8080`, which do collide. `PoseStreamer`
binds `127.0.0.1`, so either bind `0.0.0.0` or run `ssh -L 5000:localhost:5000`.

**`test/mde_localization/replay_mcl.py`** (new, no hardware). `grid.ray_cast_batch` is both the
simulator and the model's expectation function, which is what makes this testable:
- `test_pressed_against_wall()` — **the regression test for the whole problem.** Truth pose
  0.12 m from a wall so every forward ray is `TOO_CLOSE`/`NO_RETURN`; assert
  `L(truth) >> L(open-room pose)`. Under the current code both poses produce an identical
  all-0.6 scan and the likelihoods are equal — the bug, stated as an executable test.
- `test_no_return_is_informative()` — assert `L(NR | z_exp=0.10) ≈ L(NR | z_exp=0.90) >> L(NR | z_exp=0.35)`.
- `test_sensor_model_shapes()` — sweep a candidate pose over ±0.15 m / ±20° and assert the
  argmax is within one step of truth.
- `run_closed_loop()` — full PF on a synthetic trajectory. Targets: RMS position < 0.03 m,
  heading < 5°, mean `N_eff` > 0.3N. Compare against dead-reckoning-only to prove MCL helps.
- `test_protocol_roundtrip()` — `unpack(pack())` field-by-field, plus `from_legacy`.
- `test_floor_channel_on_stills()` — run the ranger on `data/test/rel_depth_test.npz` (golden:
  obstacle at 0.386–0.390 m spanning bearings −36° to −24°), `data/test/cube_60cm.jpg` (known
  0.60 m ground truth), `data/floor_verification/`, `data/leg_problem/ref{16,18,20}.jpg` (thin
  table legs — the hardest case). Write `debug_overlay()` images for eyeballing contact rows.

**Hardware bring-up order.** Each step gates the next:
1. Extrinsics fix → `average_floor_z(pcd_rc)` ≈ 0 ± 3 mm on a clean-floor frame.
2. Logit calibration → metric depth of the 96 samples within 10 mm.
3. Floor channel alone, stationary at a surveyed pose → 16 ranges vs `ray_cast_batch` at that
   pose, sub-2 cm agreement.
4. Drive toward a wall until `TOO_CLOSE` fires; confirm it fires at ~0.15 m and not before.
5. MCL with `α` tuned so `N_eff` sits in 0.3–0.6·N.
6. Push the robot off its odometry track by hand; watch MCL pull it back on the visualizer.

**One capture to go take:** a deliberate "robot pressed against a wall" frame. It is the only
image needed to tune `τ_seed`, and there is nothing equivalent in `data/` today.

## Parameters needing empirical tuning

| knob | seed | tune on |
|---|---|---|
| `τ` (break threshold) | 0.20 | `data/leg_problem/ref{16,18,20}.jpg` — thin legs span few columns |
| `k_consec` | 4 | same; raise to 6 if speckle survives the median |
| `v_top` / bin percentile | 100 / 20th | measured tradeoff above; joint choice |
| `τ_seed` | 0.35 | the near-wall capture above |
| `α` (ray correlation) | 0.5 | `N_eff` in the replay harness |
| `p_nr_floor` | 0.15 | raise if logs show spurious `NO_RETURN`s |
| `robot_radius` | 0.095 | **measure the chassis** |
