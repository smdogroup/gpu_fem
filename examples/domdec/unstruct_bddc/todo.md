# TODO

## Main paper direction

Current plan is to focus on **comparing partitioning methods** rather than developing new algorithms.

The main hypothesis is:

- Structured partitioning + wing wraparound provides the best BDDC performance for thin shells.
- METIS-based unstructured partitioning can be significantly improved through:
  - Corner optimization.
  - Improved BDDC interface control near wing junctions.
- Despite these improvements, structured partitioning is still expected to outperform unstructured partitioning.
- Support these conclusions through both numerical comparisons and theoretical discussion.

---

## Implemented improvements

- [x] Standard greedy partition improvement
- [x] Edge expansion
- [x] Corner separation
- [x] Macro-elements / supernodes
- [x] METIS corner optimization (structured cylinders)
- [x] METIS corner optimization (unstructured cylinders)
- [x] Optional METIS corner optimization for comparisons
- [x] Junction-aware METIS edge weighting
- [x] Prevent BDDC vertices on wing junctions

---

## Performance studies

- [ ] Compare against partitioning methods from the literature.
- [ ] Generate heatmaps over:
  - Shell thickness = `{1e-1, 1e-2, 1e-3}`
  - Target subdomain size = `{4, 16, 64}`
- [ ] Compare:
  - Structured cylinders
  - Unstructured cylinders
  - Structured wings
  - Unstructured wings
- [ ] Produce scatter plots relating runtime/iterations to:
  - Corner violations
  - Total BDDC vertices
  - Wing-junction BDDC vertices
  - Other interface quality metrics

---

## Optional future work

Only if further improvement of unstructured partitioning is desired:

- [ ] Identify additional partition/interface quality metrics governing BDDC performance for unstructured cylinders and wings.
- [ ] Determine whether additional interface constraints beyond corner violations and wing-junction BDDC vertices are required.
- [ ] Develop new partition optimization strategies targeting these additional metrics.