# Scale Fit Formula Fix Report

## Problem

When using log-MSE loss: `loss = mean((log(m) - log(s*f))^2)`

The optimal scale should be the **geomean**, not sum/sum:

```
d/ds [mean((log(m) - log(s*f))^2)] = 0
→ mean(log(m) - log(s*f)) = 0
→ s = 10^mean(log(m) - log(f))  (geomean)
```

Using sum/sum scale in log-MSE loss causes:
- Scale dominated by high-fluence vertices
- Log residual not minimized
- Position gradient diluted by constant bias

## Files Fixed (log-MSE loss with sum/sum scale)

### 1. ef_multi_source.py
- **Line 92**: `scale = np.sum(measurement[valid]) / np.sum(forward[valid])`
- **Loss**: `return float(np.mean((log_meas - log_fwd) ** 2))` (log-MSE)
- **Fix**: Use `scale_factor_logmse(measurement[valid], forward[valid])`

### 2. m5_prime_joint.py
- **Line 96**: `scale = np.sum(measurement[valid]) / np.sum(forward[valid])`
- **Loss**: log-MSE
- **Fix**: Use `scale_factor_logmse`

### 3. eg_optical_prior.py
- **Line 91**: `scale = np.sum(measurement[valid]) / np.sum(forward[valid])`
- **Loss**: log-MSE
- **Fix**: Use `scale_factor_logmse`

### 4. m4_prime_surface.py
- **Line 99**: `scale = float(np.sum(phi_mcx[valid]) / np.sum(forward[valid]))`
- **Loss**: log-MSE
- **Fix**: Use `scale_factor_logmse`

### 5. m4_prime_multiview_fixed.py
- **Line 69**: `scale_i = float(np.sum(measurement[valid]) / np.sum(forward[valid]))`
- **Loss**: log-MSE (per-view)
- **Fix**: Use `scale_factor_logmse`

## Files NOT Modified (diagnostic only, will redo in Step 3)

- `diag_p01_abc.py`, `diag_p01_abc_fast.py`: Optimizer diagnostics (paused)
- `diag_ef_vs_m4.py`, `verify_init_effect.py`: Diagnostic scripts
- `d2_surface_space_ncc.py`, `d2_1_direct_path_vertex_ncc.py`: Will redo with unified metrics
- `d2c_superficial_regime.py`, `d2b_analyze_distance.py`: Will redo

## Files NOT Modified (different context)

- `m4_prime_inversion.py`: Scale used for reporting, not in loss
- `m4_prime_multiview.py`: Original version, may be deprecated
- `m2_prime_validation.py`: Reporting only
- `plot_comparison.py`: Already fixed in Step 1

## Summary

| File | Changed | Reason |
|------|---------|--------|
| ef_multi_source.py | ✅ | log-MSE loss |
| m5_prime_joint.py | ✅ | log-MSE loss |
| eg_optical_prior.py | ✅ | log-MSE loss |
| m4_prime_surface.py | ✅ | log-MSE loss |
| m4_prime_multiview_fixed.py | ✅ | log-MSE loss |
