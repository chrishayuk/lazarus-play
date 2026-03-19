1. **Surprise as event predictor** — anti-correlated with actual events. Events use predictable operational language.

2. **Speaker-change frequency as event signal** — anti-correlated. High speaker changes indicate routine callouts, not significant events.

3. **Angular velocity phase boundaries** — too smooth. No sharp transitions to detect in the residual stream trajectory.

4. **Tension probe as event detector** — domain context inflation. Space mission content saturates the tension range (3-5 out of 5). No floor anchor. Val accuracy below chance at all layers.

5. **Two-layer cascade L26→L29** — redundant. L26 dark space and L29 attention carry overlapping signal. One layer is sufficient per query type.

6. **Expression-mode tonal steering at scale** — W170 rank 308. Too many dimensions to steer reliably at corpus scale.

7. **4D evaluative manifold** — only 2-3D effective. The fourth dimension doesn't carry discriminative signal.

8. **Inject-all multi-fact** — catastrophic. L31-L33 amplifies the largest coefficient regardless of entity identity. No address bus in the amplification layers. Wrong routing → wrong answer at 99.9% confidence.

9. **Compressed page replay** — worse than plain replay. The compression artifacts degrade answer quality more than the latency savings justify.

10. **Operational language prompt warnings** — marginal improvement on one failure mode, causes new errors elsewhere. Net negative.

11. **Fixed 15% confidence threshold at scale** — becomes useless beyond N≈50. At N=3,625, max Q·K scores drop below 1%. Zero injection rate with fixed threshold. Must use adaptive threshold (mean × 2.0).

12. **L14 K-vectors for routing novel entities** — the entity compass at L14 fires only for training-data entities (Namath: sep_score 39.26, Zarkov: 0.41). Novel facts get near-random discrimination. L29 is correct because novel entities build signal through template processing.

13. **Multi-layer K concatenation (L23+L29)** — L23 ≈ L29 in PCA structure. 2-4× storage cost, minimal discrimination gain. Not worth it.

14. **Multi-head K concatenation** — same finding. Different KV heads at L29 don't carry enough additional entity-discriminative information to justify 4× storage (2 KB/fact vs 512 bytes/fact).

15. **Entity-enhanced K-vectors (amplifying entity signal in hidden space)** — the entity signal IS large and well-separated in 2560D (73.87° for Namath/Marchand). But W_K collapses it during projection to 256D. Amplifying in hidden space doesn't help because you can't control which directions W_K preserves. The problem is the projection matrix, not the representation.

16. **Contrastive K-vectors (subtracting corpus mean)** — same root cause as 15. Removing shared structure in 2560D doesn't guarantee the entity-discriminative residual survives W_K projection to 256D. Not testable without direct K-space manipulation.

17. **Fixing W_K at inference time** — the projection matrix is frozen. Without retraining (LoRA on KV projections with a routing-aware objective), you cannot make W_K preserve entity-discriminative directions. Engineering around it is cheaper and more reliable.

18. **Raw H-space routing without centering** — dim 443 spike (magnitude 42K-65K) shared across all vectors compresses every pairwise cosine to 0.97-0.99. Routing becomes noise. Discrimination ratio 1.011×. Worse than K-space.

19. **H-space PCA-16 routing for V-injection** — the geometry works beautifully (16 dimensions, 32 bytes, resolves Namath at 1.15×, beats W_K-256 at 1/16th storage). But format matching creates a circular dependency: extracting a format-matched fact vector requires the document context as input, and if you have the document context loaded you don't need V-injection. The routing mechanism can't be used for the thing it was designed to route.

---

Nineteen dead ends. Every one looked promising before the experiment. The ones that hurt most:

- **12 (L14)** — peak entity separation, fires perfectly for known entities, completely useless for novel ones. The most seductive trap.
- **15 (entity-enhanced K)** — the elegant geometric fix. 73.87° of separation, 43% variance. All of it lost in a weight matrix you can't change.
- **19 (PCA-16)** — the best routing geometry we found. 32 bytes. 99% entity variance. Killed by a dependency loop, not by the maths.

The surviving architecture: adaptive Q·K at ~40% injection rate, replay fallback for the rest. Pure geometry. No hacks. The W_K projection problem is real and open.