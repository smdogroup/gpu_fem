
# 2/3/25 NASA structures meeting
Reordering -> Nested dissection –> METIS will be much more efficient
76% compute throughput because you are memory-bound
Speed of light not as important
Prioritize nonlinear solve over MDAO stuff before May
Linear solver is the bottle neck
Q-ordering on ILU -> GMRES -> Read the paper by Kevin
RCM Ordering, Cuthill Mckee ordering in TACS utilities cpp
Q ordering + ILU -> speed up? -> May help with nonlinear solves
Direct factorization acts as a fallback or basline
Look in TACS for METIX nested dissections TACS Assembler
