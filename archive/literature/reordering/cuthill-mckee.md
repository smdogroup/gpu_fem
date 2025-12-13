# Cuthill-Mckee Algorithm


* use TACS METIS code for this one.. instead of AMD
- [ ] [Cuthill-Mckee wiki](https://en.wikipedia.org/wiki/Cuthill%E2%80%93McKee_algorithm)
- [ ] [Cuthill and Mckee, Reducing the bandwidth of sparse symmetric matrices](https://dl.acm.org/doi/10.1145/800195.805928)
- [ ] [Matrix bandwidth, Wolfram mathworld](https://mathworld.wolfram.com/Bandwidth.html#:~:text=The%20bandwidth%20of%20a%20matrix%20M%3D%20%28m_%20%28ij%29%29,of%20%7Ci-j%7C%20such%20that%20m_%20%28ij%29%20is%20nonzero.)
    * matrix bandwidth of matrix $M = m_{ij}$ is defined as $bandwidth = max_{i,j} |i-j| \land m_{ij} \neq 0$
* reverse Cuthill-Mckee is usually better than regular one
- [ ] [Tutorial: Bandwidth reduction - The CutHill-McKee Algorithm](https://ciprian-zavoianu.blogspot.com/2009/01/project-bandwidth-reduction.html)
    * the Cuthill-McKee algorithm minimizes the bandwidth of the matrix by reordering the nodes or the rows of the matrix with some permutation $M' = PMP^T$
    * the most widely used bandwidth minimizing algorithm is Reverse Cuthill-Mckee (RCM), although a new 1976 GPS algorithm has equivalent quality but is up to 10x as fast
    * degree of a node in a graph is the number of nodes adjacent to it.
* algorithm for reverse-cuthill McKee (from above reference)
    * Prepare an empty queue $Q$ and an empty result array $R$. 
    * Then select the node $P$ which has the lowest degree in $G(n)$ graph, call it the parent
    * add to the queue all nodes adjacent with $P$ in increasing order of their degree.
    * extract the first node from the queue and examine it, call it the child $C$.
    * if $C$ hasn't previously been inserted in the result array $R$ then add it to the first free position in $R$, repeated by adding to the queue all nodes adjacent to $C$ in increasing order of degree.
    * if $Q$ queue is not empty repeat
    * if there are unexplored nodes, repeat from the first step
    * reverse the order of the elements in $R$, element $R[i]$ is swapped with element $R[n+1-i]$. 
    * the result array indices that $R[L] = i$ means the new label of node $i$ is $L$.
    * I believe in Dr. K implementation, there is a level set strategy for the reordering? See Dr. K's code in TACSUtilities.cpp in TACS.

## Faster GPS ALgorithm for minimizing bandwidth
- [ ] [GPS or Gibbs, Poole, and Stockmeyer algorithm: "An Algorithm for Reducing the Bandwidth and Profile of a Sparse Matrix](https://www.jstor.org/stable/2156090?seq=1)
    * this GPS algorithm has similar effectiveness to Reverse Cuthill-Mckee and is up to 10x as fast as the RCM algorithm.
    * Let $Ax = b$ be a sparse $n \times n$ linear system of equations.
    * Let the bandwidth $\beta = \max_{a_{ij} \neq 0} |i - j|$
    * Let the profile of $A$ be $\sum_{i=1}^n \delta_i$ such that $\delta_i = i - f_i$ and $f_i = \min_{j} \{j : a_{ij} \neq 0 \}$, so that the profile is the overall max distances from the diagonal among all the rows
    * The GPS algorithm minimizes the bandwidth by finding a permutation of nodes so that $P A P^T$ matrix has smaller bandwidth and profile than original $A$.
    * requires symmetric, non-zeor structure.
    * the paper provides a simple description of the reverse Cuthill-McKee algorithm which describes it in terms of level-sets.
    * let's come back to this later.. and try and implement it because it's much faster than RCM.




<!-- ## Nested Dissection Method
* why is this here?

- [ ] [Nested dissection wiki](https://en.wikipedia.org/wiki/Nested_dissection)
- [ ] [Nested Dissection of a Regular Finite Element Mesh](https://epubs.siam.org/doi/10.1137/0710032) -->
