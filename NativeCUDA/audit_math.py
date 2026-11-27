import csv
import math
import numpy as np
import pandas as pd

# Load Phase3_Trials.csv
trials_path = r'C:\Users\Adhithi C Iyer\Desktop\pagp\BenchmarkResults\Phase3_Final\Phase3_Trials.csv'
stats_path = r'C:\Users\Adhithi C Iyer\Desktop\pagp\BenchmarkResults\Phase3_Final\Phase3_Statistics.csv'

with open(trials_path) as f:
    raw_lines = f.readlines()

print(f"Total lines in Phase3_Trials.csv: {len(raw_lines)}")

# Parse trials
rows_A = []
rows_B = []
rows_C = []
rows_D = []
rows_E = []

current_section = None
for line in raw_lines:
    parts = line.strip().split(',')
    if not parts or not parts[0]:
        continue
    if parts[0] == 'Experiment':
        current_section = parts[2] if len(parts) > 2 else 'Header'
        continue
    exp = parts[0]
    if exp == 'A':
        rows_A.append(parts)
    elif exp == 'B':
        rows_B.append(parts)
    elif exp == 'C':
        rows_C.append(parts)
    elif exp == 'D':
        rows_D.append(parts)
    elif exp == 'E':
        rows_E.append(parts)

print(f"Rows count: A={len(rows_A)}, B={len(rows_B)}, C={len(rows_C)}, D={len(rows_D)}, E={len(rows_E)}")

# Helper to compute sample statistics (ddof=1) vs population stddev (ddof=0)
def calc_stats(vals):
    n = len(vals)
    if n == 0:
        return {}
    s_vals = sorted(vals)
    mean = sum(vals) / n
    median = np.median(vals)
    # stddev ddof=1 (sample) vs ddof=0 (population)
    std_sample = np.std(vals, ddof=1) if n > 1 else 0.0
    std_pop = np.std(vals, ddof=0)
    min_v = s_vals[0]
    max_v = s_vals[-1]
    p95_idx = int(0.95 * n)
    if p95_idx >= n: p95_idx = n - 1
    p95_code = s_vals[p95_idx]
    p95_np = np.percentile(vals, 95)
    return {
        'n': n, 'mean': mean, 'median': median,
        'std_sample': std_sample, 'std_pop': std_pop,
        'min': min_v, 'max': max_v,
        'p95_code': p95_code, 'p95_np': p95_np
    }

print("\n--- Auditing Exp A Statistics ---")
dfA = pd.DataFrame(rows_A, columns=['Exp', 'AgentCount', 'Solver', 'Trial', 'StepMs', 'SolveMs', 'Overflow', 'Dropped'])
for solver in ['CPU_BruteForce', 'CPU_SpatialHash', 'CUDA_SpatialHash']:
    for count in ['500', '1000', '2500', '5000']:
        sub = dfA[(dfA['Solver'] == solver) & (dfA['AgentCount'] == count)]
        if len(sub) == 0:
            continue
        vals_step = []
        for x in sub['StepMs']:
            if x != 'TIMEOUT':
                try: vals_step.append(float(x))
                except: pass
        if vals_step:
            res = calc_stats(vals_step)
            print(f"A {solver} N={count}: n={res['n']} mean={res['mean']:.4f} med={res['median']:.4f} std_samp={res['std_sample']:.4f} std_pop={res['std_pop']:.4f} min={res['min']:.4f} max={res['max']:.4f} p95={res['p95_code']:.4f}")
        else:
            print(f"A {solver} N={count}: TIMEOUT (0 valid trials)")

print("\n--- Auditing Exp C (AoS vs SoA) Statistics ---")
dfC = pd.DataFrame(rows_C, columns=['Exp', 'AgentCount', 'Trial', 'AoS_SolveMs', 'SoA_SolveMs', 'Speedup'])
for count in ['500', '1000', '2500', '5000']:
    sub = dfC[dfC['AgentCount'] == count]
    aos = [float(x) for x in sub['AoS_SolveMs']]
    soa = [float(x) for x in sub['SoA_SolveMs']]
    spd = [float(x) for x in sub['Speedup']]
    st_aos = calc_stats(aos)
    st_soa = calc_stats(soa)
    st_spd = calc_stats(spd)
    print(f"C N={count}: AoS={st_aos['mean']:.4f} (med {st_aos['median']:.4f}), SoA={st_soa['mean']:.4f} (med {st_soa['median']:.4f}), Speedup={st_spd['mean']:.4f} (med {st_spd['median']:.4f})")
    # Mean of speedups vs ratio of means:
    ratio_means = st_aos['mean'] / st_soa['mean']
    ratio_medians = st_aos['median'] / st_soa['median']
    print(f"     Ratio of Means: {ratio_means:.4f}, Ratio of Medians: {ratio_medians:.4f}, Mean of Ratios: {st_spd['mean']:.4f}")

print("\n--- Auditing Exp D (PBD Convergence) Statistics ---")
dfD = pd.DataFrame(rows_D, columns=['Exp', 'AgentCount', 'Iterations', 'Trial', 'SolveMs', 'TotalMs', 'AvgPenetration', 'MaxPenetration', 'ActivePairs'])
for iters in ['1', '2', '4', '8', '12', '16']:
    sub = dfD[dfD['Iterations'] == iters]
    solve = [float(x) for x in sub['SolveMs']]
    pen = [float(x) for x in sub['AvgPenetration']]
    maxpen = [float(x) for x in sub['MaxPenetration']]
    pairs = [float(x) for x in sub['ActivePairs']]
    st_s = calc_stats(solve)
    st_p = calc_stats(pen)
    st_mp = calc_stats(maxpen)
    st_pr = calc_stats(pairs)
    print(f"D iters={iters}: Solve={st_s['mean']:.4f}ms, AvgPen={st_p['mean']:.6f}m, MaxPen={st_mp['mean']:.6f}m, Pairs={st_pr['mean']:.1f}")

# PBD reduction from iter 1 to 16:
p1 = dfD[dfD['Iterations'] == '1']['AvgPenetration'].astype(float).mean()
p16 = dfD[dfD['Iterations'] == '16']['AvgPenetration'].astype(float).mean()
pct_reduction = (p1 - p16) / p1 * 100.0
print(f"\nPBD Avg Penetration Reduction (1->16 iters): ({p1:.6f} - {p16:.6f}) / {p1:.6f} = {pct_reduction:.2f}%")

mp1 = dfD[dfD['Iterations'] == '1']['MaxPenetration'].astype(float).mean()
mp16 = dfD[dfD['Iterations'] == '16']['MaxPenetration'].astype(float).mean()
mpct_reduction = (mp1 - mp16) / mp1 * 100.0
print(f"PBD Max Penetration Reduction (1->16 iters): ({mp1:.6f} - {mp16:.6f}) / {mp1:.6f} = {mpct_reduction:.2f}%")
