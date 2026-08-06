// the following are some reordering utils

#pragma once
#include <cstring>
#include "mesh/legacy/TACSUtilities.h"

// q-ordering (writing it myself),
// https://arc.aiaa.org/doi/epdf/10.2514/6.2020-3022 how to apply ILU(k) from
// below RCM reorderings (reverrse cuthill-mckee)? here's how to implement
// q-ordering 1) reorder the CSR data structure with bandwidth minimizing step
// like RCM 2) compute the bandwidth of the new reordered structure 3) random
// reodering reorder every groups of rows with #rows=1/p * bandwidth where p =
// prune width
//      normal prune widths are 2,1,1/2,1/4 (lower prune width results in more
//      nnz but more stable convergence of iterative solvers like GMRES)

// int TacsIntegerComparator(const void *a, const void *b) { return (*(int *)a - *(int *)b); }

/*!
  Merge two sorted arrays into a single sorted array, in place.
  This relies on both arrays having unique elements independently (ie
  a cannot contain duplicates and b cannot contain duplicates, but a
  can contain some of the same elements as b).

  Two part algorithm:
  1. Find the number of duplicates between a and b
  2. Run through the list backwards placing elements into a[]
  when appropriate.
*/
// int TacsMergeSortedArrays(int na, int *a, int nb, const int *b) {
//     int ndup = 0;

//     int j = 0, i = 0;
//     for (; i < na; i++) {
//         while ((j < nb) && b[j] < a[i]) {
//             j++;
//         }
//         if (j >= nb) {
//             break;
//         }
//         if (a[i] == b[j]) {
//             ndup++;
//         }
//     }

//     int len = na + nb - ndup;  // End of the array
//     int end = len - 1;

//     j = nb - 1;
//     i = na - 1;
//     while (i >= 0 && j >= 0) {
//         if (a[i] > b[j]) {
//             a[end] = a[i];
//             end--, i--;
//         } else if (b[j] > a[i]) {
//             a[end] = b[j];
//             end--, j--;
//         } else {  // b[j] == a[i]
//             a[end] = a[i];
//             end--, j--, i--;
//         }
//     }

//     // Only need to copy over remaining elements from b - if any
//     while (j >= 0) {
//         a[j] = b[j];
//         j--;
//     }

//     return len;
// }

/*!
  Extend the length of an integer array to a new length of array
*/
// void TacsExtendArray(int **_array, int oldlen, int newlen) {
//     int *oldarray = *_array;
//     int *newarray = new int[newlen];
//     memcpy(newarray, oldarray, oldlen * sizeof(int));
//     delete[] * _array;
//     *_array = newarray;
// }

/*
  Extend the length of a TacsScalar array to a new length
*/
// void TacsExtendArray(double **_array, int oldlen, int newlen) {
//     double *oldarray = *_array;
//     double *newarray = new double[newlen];
//     memcpy(newarray, oldarray, oldlen * sizeof(double));
//     delete[] * _array;
//     *_array = newarray;
// }

/*
  Sort an array and remove duplicate entries from the array. Negative
  values are removed from the array.

  This function is useful when trying to determine the number of
  unique design variable from a set of unsorted variable numbers.

  The algorithm proceeds by first sorting the array, then scaning from
  the beginning to the end of the array, copying values back only once
  duplicate entries are discovered.

  input:
  len:     the length of the array
  array:   the array of values to be sorted

  returns:
  the size of the unique list <= len
*/
// int TacsUniqueSort(int len, int *array) {
//     // Sort the array
//     qsort(array, len, sizeof(int), TacsIntegerComparator);

//     int i = 0;  // The location from which to take the entires
//     int j = 0;  // The location to place the entries

//     // Remove the negative entries
//     while (i < len && array[i] < 0) i++;

//     for (; i < len; i++, j++) {
//         while ((i < len - 1) && (array[i] == array[i + 1])) {
//             i++;
//         }

//         if (i != j) {
//             array[j] = array[i];
//         }
//     }

//     return j;
// }

// cuthill-mckee and reverse cuthill-mckee are shown here

/*!
  Given an unsorted CSR data structure, with duplicate entries,
  sort/uniquify each array, and recompute the rowp values on the
  fly.

  For each row, sort the array and remove duplicates.  Copy the
  values if required and skip the diagonal.  Note that the copy may
  be overlapping so memcpy cannot be used.
*/
void TacsSortAndUniquifyCSR(int nvars, int *rowp, int *cols, int remove_diag) {
    // Uniquify each column of the array
    int old_start = 0;
    int new_start = 0;
    for (int i = 0; i < nvars; i++) {
        // sort cols[start:rowp[i]]
        int rsize = TacsUniqueSort(rowp[i + 1] - old_start, &cols[old_start]);

        if (remove_diag) {
            int end = old_start + rsize;
            for (int j = old_start, k = new_start; j < end; j++, k++) {
                if (cols[j] == i) {
                    rsize--;
                    k--;
                } else if (j != k) {
                    cols[k] = cols[j];
                }
            }
        } else if (old_start != new_start) {
            int end = old_start + rsize;
            for (int j = old_start, k = new_start; j < end; j++, k++) {
                cols[k] = cols[j];
            }
        }

        old_start = rowp[i + 1];
        rowp[i] = new_start;
        new_start += rsize;
    }

    rowp[nvars] = new_start;
}

/*
  Compute the RCM level sets andreordering of the graph given by the
  symmetric CSR data structure rowp/cols.

  rowp/cols represents the non-zero structure of the matrix to be
  re-ordered

  Here levset is a unique, 0 to nvars array containing the level
  sets
*/
static int TacsComputeRCMLevSetOrder(const int nvars, const int *rowp, const int *cols,
                                     int *rcm_vars, int *levset, int root) {
    int start = 0;  // The start of the current level
    int end = 0;    // The end of the current level

    // Set all the new variable numbers to -1
    for (int k = 0; k < nvars; k++) {
        rcm_vars[k] = -1;
    }

    int var_num = 0;
    while (var_num < nvars) {
        // If the current level set is empty, find any un-ordered variables
        if (end - start == 0) {
            if (rcm_vars[root] >= 0) {
                // Find an appropriate root
                int i = 0;
                for (; i < nvars; i++) {
                    if (rcm_vars[i] < 0) {
                        root = i;
                        break;
                    }
                }
                if (i >= nvars) {
                    return var_num;
                }
            }

            levset[end] = root;
            rcm_vars[root] = var_num;
            var_num++;
            end++;
        }

        while (start < end) {
            int next = end;
            // Iterate over the nodes added to the previous level set
            for (int current = start; current < end; current++) {
                int node = levset[current];

                // Add all the nodes in the next level set
                for (int j = rowp[node]; j < rowp[node + 1]; j++) {
                    int next_node = cols[j];

                    if (rcm_vars[next_node] < 0) {
                        rcm_vars[next_node] = var_num;
                        levset[next] = next_node;
                        var_num++;
                        next++;
                    }
                }
            }

            start = end;
            end = next;
        }
    }

    // Go through and reverse the ordering of all the variables
    for (int i = 0; i < var_num; i++) {
        int node = levset[i];
        rcm_vars[node] = var_num - 1 - i;
    }

    return var_num;
}

/*!
  Perform Reverse Cuthill-McKee ordering.

  Input:
  ------
  nvars, rowp, cols == The CSR data structure containing the
  graph representation of the matrix

  root: The root node to perform the reordering from
  Returns:
  --------
  rcm_vars == The new variable ordering

  The number of variables ordered in this pass of the RCM reordering
*/
int TacsComputeRCMOrder(const int nvars, const int *rowp, const int *cols, int *rcm_vars, int root,
                        int n_rcm_iters) {
    if (n_rcm_iters < 1) {
        n_rcm_iters = 1;
    }

    int *levset = new int[nvars];
    int rvars = 0;
    for (int k = 0; k < n_rcm_iters; k++) {
        rvars = TacsComputeRCMLevSetOrder(nvars, rowp, cols, rcm_vars, levset, root);
        if (nvars != rvars) {
            return rvars;
        }
        root = rcm_vars[0];
    }

    delete[] levset;
    return rvars;
}

__HOST__ void computeILUk(int nnodes, int nnzb, int *&rowPtr, int *&colPtr, int levFill,
                          double fill, int **_levs, bool print = true) {
    int nrows = nnodes;  // Record the number of rows/columns
    int ncols = nnodes;

    // Number of non-zeros in the original matrix
    int mat_size = nnzb;
    int size = 0;
    int max_size = (int)(fill * mat_size);  // The maximum size - for now

    int *cols = new int[max_size];
    int *levs = new int[max_size];  // The level of fill of an entry
    int *rowp = new int[nrows + 1];
    int *diag = new int[nrows];

    // Fill in the first entries
    rowp[0] = 0;

    // Allocate space for the temporary row info
    int *rlevs = new int[ncols];
    int *rcols = new int[ncols];

    for (int i = 0; i < nrows; i++) {
        int nr = 0;  // Number of entries in the current row

        // Add the matrix elements to the current row of the matrix.
        // These new elements are sorted.
        int diag_flag = 0;
        for (int j = rowPtr[i]; j < rowPtr[i + 1]; j++) {
            if (colPtr[j] == i) {
                diag_flag = 1;
            }
            rcols[nr] = colPtr[j];
            rlevs[nr] = 0;
            nr++;
        }

        // No diagonal element associated with row i, add one!
        if (!diag_flag) {
            nr = TacsMergeSortedArrays(nr, rcols, 1, &i);
        }

        // Now, perform the symbolic factorization -- this generates new entries
        int j = 0;
        for (; rcols[j] < i; j++) {  // For entries in this row, before the diagonal
            int clev = rlevs[j];     // the level of fill for this entry

            int p = j + 1;                   // The index into rcols
            int k_end = rowp[rcols[j] + 1];  // the end of row number cols[j]

            // Start with the first entry after the diagonal in row, cols[j]
            // k is the index into cols for row cols[j]
            for (int k = diag[rcols[j]] + 1; k < k_end; k++) {
                // Increment p to an entry where we may have cols[k] == rcols[p]
                while (p < nr && rcols[p] < cols[k]) {
                    p++;
                }

                // The element already exists, check if it has a lower level of
                // fill and update the fill level if necessary
                if (p < nr && rcols[p] == cols[k]) {
                    if (rlevs[p] > (clev + levs[k] + 1)) {
                        rlevs[p] = clev + levs[k] + 1;
                    }
                } else if ((clev + levs[k] + 1) <= levFill) {
                    // The element does not exist but should since the level of
                    // fill is low enough. Insert the new entry into the list,
                    // but keep the list sorted
                    for (int n = nr; n > p; n--) {
                        rlevs[n] = rlevs[n - 1];
                        rcols[n] = rcols[n - 1];
                    }

                    rlevs[p] = clev + levs[k] + 1;
                    rcols[p] = cols[k];
                    nr++;
                }
            }
        }

        // Check if the size will be exceeded by adding the new elements
        if (size + nr > max_size) {
            int mat_ext = (int)((fill - 1.0) * mat_size);
            if (nr > mat_ext) {
                mat_ext = nr;
            }
            max_size = max_size + mat_ext;
            TacsExtendArray(&cols, size, max_size);
            TacsExtendArray(&levs, size, max_size);
        }

        // Now, put the new entries into the cols/levs arrays
        for (int k = 0; k < nr; k++) {
            cols[size] = rcols[k];
            levs[size] = rlevs[k];
            size++;
        }

        rowp[i + 1] = size;
        diag[i] = j + rowp[i];
    }

    // Clip the cols array to the correct size
    if (max_size > size) {
        TacsExtendArray(&cols, size, size);
    }

    if (rowPtr[nrows] > 0) {
        int rank = 0;
        // MPI_Comm_rank(comm, &rank);
        if (print) {
            printf(
                "[%d] BCSRMat: ILU(%d) Input fill ratio %4.2f, actual "
                "fill ratio: %4.2f, nnz(ILU) = %d\n",
                rank, levFill, fill, (1.0 * rowp[nrows]) / rowPtr[nrows], rowp[nrows]);
        }
    }

    delete[] rcols;
    delete[] rlevs;

    int *old_rowp = rowPtr;
    int *old_cols = colPtr;

    // Store the rowp/cols and diag arrays
    // data->rowp = rowp;
    // data->cols = cols;
    // data->diag = diag;
    rowPtr = rowp;
    colPtr = cols;
    // need to store separate diag too? it's only integers, I don't think I need
    // it here..

    delete[] old_rowp;
    delete[] old_cols;

    *_levs = levs;
}