# NCC Unification Report

## Changes to shared/metrics.py

Added two new functions:
- `ncc_log(a, b, eps=1e-20)`: Log-space NCC (auxiliary only)
- `scale_factor_logmse(meas, forward, eps=1e-20)`: Optimal scale for log-MSE loss

Existing functions unchanged:
- `ncc(a, b)`: Linear-space Pearson correlation (PRIMARY metric)
- `scale_factor_k(a, b)`: Linear scale factor (sum/sum)

## Files Modified

### 1. cubature_conv/run_ball_sweep.py
- **Deleted**: Local `compute_ncc()` function (lines 17-20)
- **Changed**: Import `from shared.metrics import ncc` and use `ncc(a, b)` instead

### 2. mvp_pipeline/m3_prime_multi_view.py
- **Deleted**: Local `compute_ncc()` function (lines 141-148) - was log-space NCC
- **Changed**: Import `from shared.metrics import ncc, ncc_log` and use appropriate function

### 3. mvp_pipeline/d2_surface_space_ncc.py
- **Changed**: Replace inline `np.corrcoef(log_mcx, log_closed)[0, 1]` with `ncc_log(mcx_vals, closed_vals)`

### 4. mvp_pipeline/plot_comparison.py
- **Changed**: Replace inline log-space correlation with `ncc_log(my_green, archived_mcx)`

## Files NOT Modified (diagnostic only)

These files use NCC for diagnostics and will be updated in Step 3:
- `diagnostics/forward_audit.py`
- `diagnostics/ncc_investigation.py`

## Summary

| File | Local Function | Replacement |
|------|---------------|-------------|
| cubature_conv/run_ball_sweep.py | `compute_ncc()` (linear) | `shared.metrics.ncc` |
| mvp_pipeline/m3_prime_multi_view.py | `compute_ncc()` (log) | `shared.metrics.ncc_log` |
| mvp_pipeline/d2_surface_space_ncc.py | inline log-NCC | `shared.metrics.ncc_log` |
| mvp_pipeline/plot_comparison.py | inline log-NCC | `shared.metrics.ncc_log` |

## Note on Log-space NCC

The historical §4.C NCC values (0.9365 / 0.9498 / 0.9429 / 0.9834 / 0.9578) were computed using log-space correlation.
For paper, linear NCC will be the primary metric, log-NCC only auxiliary.
