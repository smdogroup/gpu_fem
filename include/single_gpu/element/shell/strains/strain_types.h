#pragma once

// enum used for specifying the strain types for separating + reducing registers on GPU
// aka only computes certain strain contributions to Kelem, or Relem, etc.
enum STRAIN : short {
    ALL,
    BENDING,
    TYING,
    DRILL,
};