# Implementation Verification Checklist

## Comparison: My Implementation vs. Reference Documents

### ✅ Correctly Implemented (from cautious.md)

| Item | Requirement | My Implementation | Status |
|------|-------------|-------------------|--------|
| Big-M parameter | Use `M = MC_j` (DC capacity) | `DH_master.py:330` uses `M_j = self.data.MC[j]` | ✅ CORRECT |
| Revenue formula | `S × (demand - shortage)` | `DH_master.py:386` uses `S * (d_realized - u)` | ✅ CORRECT |
| McCormick γ^L | `-(S + SC)` | `DH_sub.py:138` sets `gamma_L = -(self.data.S + self.data.SC)` | ✅ CORRECT |
| McCormick γ^U | `S` | `DH_sub.py:139` sets `gamma_U = self.data.S` | ✅ CORRECT |
| Endogenous demand | Include `Σ μ DI β` | `DH_master.py:253-258` correctly implements | ✅ CORRECT |
| Dual variables | π, σ, ψ, φ (≥0); γ, κ (free) | `DH_sub.py:96-136` correct bounds | ✅ CORRECT |
| Fixed first-stage in SP | Pass as constants | `DH_sub.py:252` fixes variables before building | ✅ CORRECT |

### ❌ Critical Issues Found

| Issue | Impact | File | Action Needed |
|-------|--------|------|---------------|
| **Missing s_rk matrix** | 🔴 HIGH | `DH_data_gen.py` | Add binary customer-product demand indicator |
| Uniform customer locations | 🟡 MEDIUM | `DH_data_gen.py:200` | Use Gaussian distribution (mean=500, std=200) |
| Uniform DC locations | 🟡 MEDIUM | `DH_data_gen.py:205` | Use "donut" pattern (exclude center) |
| μ̂ as percentage | 🟡 MEDIUM | `DH_data_gen.py:176` | Use `min(μ, U[4,10])` instead |

### 📊 Data Structure Comparison

#### My Current Implementation:
```python
# All customers demand all products
mu[(r,k)] = random(10, 50)  # For ALL r,k pairs
mu_hat[(r,k)] = factor * mu[(r,k)]  # 20-50% of mu
```

#### Reference Implementation (old_data_generation.py):
```python
# Sparse demand pattern
s_rk[r,k] = binary(0/1)  # Which products customer needs
mu[r,k] = 10 * uniform(1, 5)  # Only where s_rk=1
mu_hat[r,k] = min(mu[r,k], uniform(4, 10))  # Absolute bounds
```

---

## Root Cause of Convergence Gap

### Hypothesis 1: Degenerate Solution (Most Likely)
**Evidence:**
- No plants opened (cost too high)
- All demand met by shortages
- Gap persists because corner solution

**Test:**
```python
# Reduce fixed costs significantly
C_plant_min = 5000   # was 50000
C_plant_max = 15000  # was 150000
```

### Hypothesis 2: Missing Constraint
**Check:**
- Are there supposed to be minimum production constraints?
- Should certain customers REQUIRE certain products (via s_rk)?

### Hypothesis 3: Numerical Issues
**Evidence:**
- Large cost values (87k-145k)
- Small demand values (17-31)
- Ratio ~3000:1 could cause scaling issues

**Fix:**
- Normalize all costs to similar magnitude
- Scale demand values up

---

## Recommended Fixes (Priority Order)

### Priority 1: Add s_rk Matrix (Critical)
**Why:** Fundamental data structure missing
**Impact:** Makes problem more realistic, may affect constraints
**File:** `DH_data_gen.py`

**Implementation:**
```python
def generate_s_rk_matrix(R, K, seed=None):
    """Generate binary customer-product demand indicator."""
    if seed:
        np.random.seed(seed)

    # Random 0/1 matrix
    s_rk = np.random.randint(0, 2, size=(R, K))

    # Ensure each customer demands at least one product
    for r in range(R):
        if np.all(s_rk[r] == 0):
            s_rk[r, np.random.randint(0, K)] = 1

    return {(r, k): int(s_rk[r, k]) for r in range(R) for k in range(K)}
```

### Priority 2: Improve Location Generation
**Why:** More realistic spatial patterns
**Impact:** Better test case, may affect solution structure
**File:** `DH_data_gen.py`

**Customer locations (Gaussian):**
```python
# Center at (grid_size/2, grid_size/2), std = grid_size/5
x_coords = np.random.normal(grid_size/2, grid_size/5, R)
y_coords = np.random.normal(grid_size/2, grid_size/5, R)
# Clip to grid bounds
x_coords = np.clip(x_coords, 0, grid_size)
y_coords = np.clip(y_coords, 0, grid_size)
```

**DC locations (Donut):**
```python
# Exclude central region [0.2*grid, 0.8*grid]
exclude_min = int(0.2 * grid_size)
exclude_max = int(0.8 * grid_size)

available_points = [
    (x, y) for x in range(grid_size+1) for y in range(grid_size+1)
    if not (exclude_min <= x < exclude_max and exclude_min <= y < exclude_max)
]
dc_coords = random.sample(available_points, J)
```

### Priority 3: Fix Demand Generation
**Why:** Use absolute bounds instead of percentage
**Impact:** May create more diverse scenarios
**File:** `DH_data_gen.py`

**Current:**
```python
mu_hat[(r,k)] = factor * mu[(r,k)]  # factor in [0.2, 0.5]
```

**Better:**
```python
mu_hat[(r,k)] = min(mu[(r,k)], np.random.uniform(4, 10))
```

### Priority 4: Adjust Cost Parameters
**Why:** Current costs may create degenerate solutions
**Impact:** May fix convergence issue
**File:** `DH_config.py`

**Suggested values:**
```python
# Reduce fixed costs
C_plant_min = 5000   # was 50000
C_plant_max = 15000  # was 150000
C_dc_min = 3000      # was 30000
C_dc_max = 8000      # was 80000

# OR increase revenue
S = 200.0   # was 100.0
SC = 30.0   # was 50.0
```

---

## Testing Plan

1. **Implement s_rk matrix** → Regenerate data → Test toy instance
2. **Improve locations** → Check if spatial patterns affect solution
3. **Adjust costs** → Verify non-degenerate solution (plants open)
4. **Debug convergence** → Add logging to compare Master vs Subproblem profits

---

## Questions for Further Investigation

1. **Is s_rk used in constraints?**
   - Check if there should be constraint: `A_jr^k ≤ BigM × s_rk`
   - Or: `u_rk = 0` if `s_rk = 0` (no shortage for non-demanded products)

2. **Should demand be zero when s_rk=0?**
   - Set `μ_rk = 0` if `s_rk[r,k] = 0`?
   - Or just use s_rk as indicator without affecting μ?

3. **Plant-product structure:**
   - Old code shows 9 plants (3 per product)
   - Are plants product-specific or general?
   - Current formulation: general plants with product-specific capacity

---

**Status:** Issues identified, fixes designed, ready to implement
**Next Step:** Implement Priority 1 (s_rk matrix) and retest
