## Unstruct splittings to try

- [x] standard greedy
- [x] edge expansion
- [x] corner separation (reduce corner violations = element cannot have two corner / vertex nodes on it)
- [x] macro-elements / supernodes from elem-conn

* METIS optimizer works great with larger subdomain sizes like target_sd_size=64 from subdomain=8 input!! For unstructured even too! 
    * it just doesn't work well with smaller subdomain inputs (which can be slightly faster).. so a bit lost performance but good thin shell still subdomain_size = 8 is still good enough for large mesh sizes!
- [x] METIS corner sep opt for struct cylinder case
- [x] make METIS split option that doesn't do corner-opt (so you can see impact), using template on METIS corner opt thing
- [x] METIS corner sep opt for unstruct cylinder case

- [ ] seems to only give small speedups (METIS => METIS-OPT) for the unstructured case.. some vertex nodes / corner violations not counted properly? Leading to lots more iterations?
    * still lots of corner violations remain.. why?
    * mesh regularity seems to have high impact (low # points+low size is more regular, high # embedded points + high size is more irregular)
    * with more regular mesh (but unstruct) metis + metis-opt very similar results.. but irregular mesh also similar runtimes but very bad # iterations.. some mesh regularity factor?
    * can I improve the irregular case more? What is limiting convergence still?

- [ ] make a METIS corner sep optimizer for the wing case
    - [ ] add METIS weights so initial mesh edges don't want to go on these junction boundaries?
    - [ ] ensure no vertex nodes on the wing junction nodes.. corrections for this
    - [ ] test this out..

- [ ] compare subdomain splitting methods from literature, see METIS ref https://www.cs.utexas.edu/~pingali/CS395T/2009fa/papers/metis.pdf?utm_source=chatgpt.com
    - [ ] a bunch of heatmap tables of thickness = {1e-1, 1e-2, 1e-3} x target_sd_size = {4, 16, 64} or something like that? Also do non-divisible values?
    - [ ] do comparisons for:
        - [ ] struct cylinders (diff nxe values)
        - [ ] unstruct cylinders
        - [ ] struct wing
        - [ ] unstruct wing

- [ ] make a scatter plot with 4 dots (unstruct-cyl, struct-cyl, unstruct-wing, struct-wing) for thin shells and show how corner violations / whatever metrics influence the runtime
      - [ ] it would show evidence / correlation of a pattern here..
      - [ ] maybe also show diff thicks across diff cases affect sensitivity?
      - [ ] keep wing vertex on patch boundary metric too and show how that influences runtime (again show a bunch of wing cases maybe in scatter plot to show evidence instead off proof!), like Aaron's paper with many optim cases back then




## MAYBE

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