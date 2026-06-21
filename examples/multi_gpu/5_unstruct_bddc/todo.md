## Unstruct splittings to try

- [x] standard greedy
- [x] edge expansion
- [x] corner separation (reduce corner violations = element cannot have two corner / vertex nodes on it)
- [x] macro-elements / supernodes from elem-conn
# - [ ] make subdomains of different sizes in some sort of way..  
#     - [ ] doesn't really seem to work yet, needs to do some type of global scans rn..
# - [ ] do some reading on global-local discrete optimizations maybe.. and other stuff like HJB
# - [ ] need better global-local corner sep optimizations.. it's stalling a lot.. draw stuff by hand? and see how I might do that? HJB?
# - [ ] discrete optimizations
#     - [ ] BFS vs DFS
#     - [ ] HJB (hamilton Jacobi-bellman) discrete form
#     - [ ] simulated annealing
#     - [ ] how does METIS do it?
# - [ ] how to do global-local optimizations with discrete problem?

* METIS optimizer works great with larger subdomain sizes like target_sd_size=64 from subdomain=8 input!! For unstructured even too! 
    * it just doesn't work well with smaller subdomain inputs (which can be slightly faster).. so a bit lost performance but good thin shell still subdomain_size = 8 is still good enough for large mesh sizes!

- [ ] make a METIS corner sep optimizer for the wing case
    - [ ] add METIS weights so initial mesh edges don't want to go on these junction boundaries?
    - [ ] ensure no vertex nodes on the wing junction nodes.. corrections for this
    - [ ] test this out..

- [ ] compare subdomain splitting methods from literature, see METIS ref https://www.cs.utexas.edu/~pingali/CS395T/2009fa/papers/metis.pdf?utm_source=chatgpt.com
    - [ ] a bunch of heatmap tables of thickness = {1e-1, 1e-2, 1e-3} x target_sd_size = {2^2, 4^2, 8^2} or something like that? Also do non-divisible values?
    - [ ] do comparisons for:
        - [ ] struct cylinders (diff nxe values)
        - [ ] unstruct cylinders
        - [ ] struct wing
        - [ ] unstruct wing