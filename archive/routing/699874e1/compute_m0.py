"""
Compute M0 2x2 cosine matrix for Namath/Marchand same-template routing test.
Run after extracting all 4 vectors from the Lazarus MCP.
"""
import numpy as np
import math
import json

def prep(lst):
    a = np.array(lst, dtype=np.float32)
    idx = int(np.argmax(np.abs(a)))
    if abs(float(a[idx])) > 10 * float(np.mean(np.abs(a))):
        a[idx] = 0.0
    return a

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Load vectors from npy files
outdir = "/Users/christopherhay/chris-source/lazarus-play/experiments/699874e1"
Q11 = np.load(f"{outdir}/Q11.npy")
Q12 = np.load(f"{outdir}/Q12.npy")
F11 = np.load(f"{outdir}/F11.npy")
F12 = np.load(f"{outdir}/F12.npy")

# Verify spikes already zeroed
for name, v in [("Q11", Q11), ("Q12", Q12), ("F11", F11), ("F12", F12)]:
    idx = int(np.argmax(np.abs(v)))
    print(f"{name}: max_dim={idx}, max_val={v[idx]:.1f}")

print()
# Compute 2x2 matrix
pairs = [
    ("Q11", "F11", Q11, F11),
    ("Q11", "F12", Q11, F12),
    ("Q12", "F11", Q12, F11),
    ("Q12", "F12", Q12, F12),
]
results = {}
for n1, n2, a, b in pairs:
    c = cosine(a, b)
    angle = math.degrees(math.acos(min(1.0, max(-1.0, c))))
    results[f"{n1}x{n2}"] = {"cosine": c, "angle_deg": angle}
    print(f"{n1} vs {n2}: cosine={c:.6f}  angle={angle:.2f}°")

print()
# Key question: does Q11 route to F11 (correct) or F12 (wrong)?
c_q11_f11 = results["Q11xF11"]["cosine"]
c_q11_f12 = results["Q11xF12"]["cosine"]
ratio_q11 = c_q11_f11 / c_q11_f12
print(f"Q11 routing: F11={c_q11_f11:.6f} vs F12={c_q11_f12:.6f} ratio={ratio_q11:.4f}x → {'CORRECT' if ratio_q11 > 1.0 else 'FAILURE'}")

c_q12_f12 = results["Q12xF12"]["cosine"]
c_q12_f11 = results["Q12xF11"]["cosine"]
ratio_q12 = c_q12_f12 / c_q12_f11
print(f"Q12 routing: F12={c_q12_f12:.6f} vs F11={c_q12_f11:.6f} ratio={ratio_q12:.4f}x → {'CORRECT' if ratio_q12 > 1.0 else 'FAILURE'}")

json.dump(results, open(f"{outdir}/m0_results.json", "w"), indent=2)
print(f"\nSaved to {outdir}/m0_results.json")
