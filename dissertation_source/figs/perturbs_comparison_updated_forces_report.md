# Perturbation Acceleration Comparison: Force Model Validation Report

**Generated**: 2025-12-14  
**Figure**: `perturbs_comparison_updated_forces_20251214.pdf`  
**Data**: `perturbs_comparison_updated_forces_20251214.npz`

---

## 1. System Information

| Property | Value |
|----------|-------|
| Git commit | `90369f4` |
| Python version | 3.13.9 (Anaconda, Clang 20.1.8) |
| Platform | macOS-26.2-arm64-arm-64bit-Mach-O |

---

## 2. Environment Variables

| Variable | Value | Notes |
|----------|-------|-------|
| `SATPY_USE_CPP_DRAG` | (not set) | C++ drag uses Python fallback by default |
| `SATPY_SPHERICAL_CONSTANTS` | (not set) | Uses default path |

---

## 3. C++ Backend Status

| Backend | Module Available | Auto-Enabled |
|---------|-----------------|--------------|
| `perturbs_forces_cpp` | ✅ Yes | - |
| Drag (`compute_drag`) | ✅ Yes | ❌ Requires `SATPY_USE_CPP_DRAG=1` |
| SRP (`compute_srp`) | ✅ Yes | ✅ Auto-enabled |
| Third-body gravity | ✅ Yes | ✅ Auto-enabled |

---

## 4. Physics Constants Verification

### Recent Fixes Validated

| Constant | Value | Expected | Status |
|----------|-------|----------|--------|
| `flux_constant` | 4.56e-6 N/m² | 4.56e-6 (solar pressure at 1 AU) | ✅ Correct |
| `omegaE` | 7.2921158579e-5 rad/s | 2π/86164.0905 | ✅ Correct |
| `omegaE` period | 86164.0905 s | Sidereal day | ✅ Correct |

### Notes on Recent Physics Updates
- **SRP**: Now uses constant solar radiation pressure at 1 AU ≈ 4.56e-6 N/m² (= 1361 W/m² / c), NOT F10.7 solar radio flux
- **Drag**: Earth rotation rate (`omegaE`) uses sidereal day (86164.0905 s), NOT solar day (86400 s)

---

## 5. Numerical Comparison: Python vs C++

### 5.1 Drag

| Metric | Value |
|--------|-------|
| Test state | LEO (400 km altitude), circular orbit |
| Max absolute difference | 0.00e+00 |
| Max relative difference | 0.00e+00 |

**Python output**: `[-0.0, -5.66403751e-08, -0.0]` km/s²  
**C++ output**: `[-0.0, -5.66403751e-08, -0.0]` km/s²

✅ **Exact match**

### 5.2 Solar Radiation Pressure (SRP)

| Metric | Value |
|--------|-------|
| Test state | LEO, Sun at 1 AU along +X |
| Max absolute difference | 0.00e+00 |
| Max relative difference | 0.00e+00 |

**Python output**: `[-6.84e-09, 0.0, 0.0]` km/s²  
**C++ output**: `[-6.84e-09, 0.0, 0.0]` km/s²

✅ **Exact match**

### 5.3 Sun Third-Body Gravity

| Metric | Value |
|--------|-------|
| Test state | LEO, Sun at 1 AU |
| Max absolute difference | 1.49e-22 |
| Max relative difference | 2.76e-13 |

**Python output**: `[5.37375583e-10, 0.0, 0.0]` km/s²  
**C++ output**: `[5.37375583e-10, 0.0, 0.0]` km/s²

✅ **Match within machine precision** (float64 relative tolerance ~1e-15)

### 5.4 Moon Third-Body Gravity

| Metric | Value |
|--------|-------|
| Test state | LEO, Moon at 384,400 km |
| Max absolute difference | 2.90e-24 |
| Max relative difference | 2.41e-15 |

**Python output**: `[1.20179922e-09, 0.0, 0.0]` km/s²  
**C++ output**: `[1.20179922e-09, 0.0, 0.0]` km/s²

✅ **Match within machine precision**

---

## 6. Performance Comparison

Benchmarked with 1000 iterations per test.

| Force | Python (µs/call) | C++ (µs/call) | Speedup |
|-------|-----------------|---------------|---------|
| Drag | 3.27 | 0.47 | **6.9x** |
| SRP | 4.73 | 0.63 | **7.5x** |
| Third-body gravity | 4.34 | 0.68 | **6.4x** |

### Notes
- C++ implementations provide **6-8x speedup** over Python
- Speedup is consistent across all ported forces
- C++ uses PGO-optimized build (+10-20% additional speedup)

---

## 7. Magnitude Sanity Checks

### Expected Values (from `satpy_tools/Propagators.md`)

| Perturbation | Expected Range | Notes |
|--------------|---------------|-------|
| Spherical harmonics (LEO) | ~1e-2 km/s² | Includes J2-J7 terms |
| Drag | ~1e-7 to 1e-5 km/s² | Varies with altitude, solar activity |
| SRP (CR=1, A/m=1 m²/kg) | ~4.56e-9 km/s² | At 1 AU |
| Moon third-body | ~1e-9 km/s² | Tidal acceleration |
| Sun third-body | ~1e-10 km/s² | Smaller than Moon for LEO |

### Actual Measurements (A/m=1.0 m²/kg, CR=1.0, LEO 400 km)

| Force | Magnitude (km/s²) | Status |
|-------|-------------------|--------|
| Drag | 2.42e-07 | ✅ Within expected range |
| SRP | 4.56e-09 | ✅ Matches expected (P x A/m = 4.56e-6 x 1.0 / 1000) |
| Sun | 3.78e-10 | ✅ Within expected range |
| Moon | 1.20e-09 | ✅ Within expected range |

---

## 8. Plot Data Summary

### Satellite Configuration
- A/m ratio: 0.01 m²/kg
- Object radius: 0.5 m
- q/m ratio: 0.0 C/kg (uncharged)
- CD: 2.2, CR: 1.3

### Dust Grain Configuration  
- A/m ratio: 1.948 m²/kg (derived from 20 µm tungsten sphere)
- Object radius: 2e-05 m (20 µm)
- q/m ratio: 4.96e-09 C/kg (20 elementary charges)
- CD: 1.3, CR: 1.35

### Perturbations Plotted
1. Drag
2. Solar radiation pressure
3. Earth IR (albedo + thermal)
4. Lorentz force
5. Coulomb drag
6. Moon gravity
7. Sun gravity
8. Venus gravity
9. Jupiter gravity
10. Relativity

### Altitude Grid
- Range: 100 km to 20,000 km
- Points: 48 (log-spaced)
- Epoch: JD 2459713.28 (JD_STANDARD + 0.2 days)

---

## 9. Files Generated

| File | Size | Description |
|------|------|-------------|
| `perturbs_comparison_updated_forces_20251214.pdf` | 70 KB | Regenerated figure |
| `perturbs_comparison_updated_forces_20251214.npz` | 11 KB | Profile data |
| `perturbs_comparison_updated_forces_report.md` | - | This report |

### NPZ Contents
```
altitudes: (48,) float64
sat_profile: (10, 48) float64
dust_profile: (10, 48) float64
perturbation_labels: (10,) <U15
perturbation_keys: (10,) <U12
metadata: (1,) object
```

---

## 10. Conclusions

1. **Force implementations are correct**: Python and C++ implementations match to machine precision (relative differences < 1e-12)

2. **Physics constants are correct**:
   - Solar radiation pressure uses 4.56e-6 N/m² (correct value at 1 AU)
   - Earth rotation rate uses sidereal day (86164.0905 s)

3. **C++ performance gains are significant**: 6-8x speedup across all ported forces

4. **Force magnitudes are physically reasonable**: All perturbations produce accelerations within expected ranges documented in `Propagators.md`

5. **Original figure preserved**: `perturbs_comparison_cropped_dissertation.pdf` is unchanged

---

*Report generated automatically by force validation script*
