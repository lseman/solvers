#ifndef SOLVERS_KLU_H
#define SOLVERS_KLU_H

#include <vector>
#include <utility>
#include <stdexcept>
#include <cstdint>

namespace klu {

// CSC matrix format
template <typename IntT = int32_t>
struct SparseCSC {
    IntT n;
    std::vector<IntT> Ap;
    std::vector<IntT> Ai;
    std::vector<double> Ax;
};

// BTF decomposition result
struct BTFResult {
    int nblocks;
    std::vector<int> P;       // Row permutation
    std::vector<int> Q;       // Column permutation
    std::vector<int> R;       // Block boundaries: R[k] to R[k+1]-1 is block k
    std::vector<int> block_of_row; // block_of_row[i] = block index for row i
};

// Symbolic analysis result
struct SymbolicResult {
    int n;
    int nblocks;
    std::vector<int> P;       // Final row permutation (AMD + BTF)
    std::vector<int> Q;       // Final column permutation (AMD + BTF)
    std::vector<int> R;       // Block boundaries
    std::vector<int> Pinv;    // Inverse row permutation
    std::vector<int> Qinv;    // Inverse column permutation
    std::vector<int> etree;   // Elimination tree for each block
    std::vector<int> Lp;      // L column pointers
    std::vector<int> Li;      // L row indices
    std::vector<int> Lnz;     // Nonzeros per L column
};

// Numeric factorization result
struct NumericResult {
    int n;
    int nblocks;
    int lnz;                  // Nonzeros in L (including diagonal)
    int unz;                  // Nonzeros in U (including diagonal)
    std::vector<int> Pnum;    // Final pivot permutation
    std::vector<int> Pinv;    // Inverse pivot permutation
    std::vector<int> Lip;     // L column pointers per block
    std::vector<int> Uip;     // U column pointers per block
    std::vector<int> Lp;      // L column pointers (full)
    std::vector<int> Up;      // U column pointers (full)
    std::vector<int> Llen;    // L column lengths
    std::vector<int> Ulen;    // U column lengths
    std::vector<int> Li;      // L row indices (concatenated)
    std::vector<double> Lx;   // L numerical values
    std::vector<int> Ui;      // U row indices (concatenated)
    std::vector<double> Ux;   // U numerical values
    std::vector<double> Udiag;// U diagonal elements
};

// KLU solver class
class Solver {
public:
    Solver();

    // Set parameters
    void setPivotTolerance(double tol) { tol_ = tol; }
    void setRowScaling(int scale) { scale_ = scale; } // 0: none, 1: sum, 2: max

    // Symbolic analysis + BTF decomposition
    SymbolicResult analyze(const SparseCSC<int>& A);

    // Numeric factorization
    NumericResult factorize(const SparseCSC<int>& A, const SymbolicResult& sym);

    // Solve Ax = b
    std::vector<double> solve(const NumericResult& num, const std::vector<double>& b) const;

    // Solve A'x = b (transpose solve)
    std::vector<double> tsolve(const NumericResult& num, const std::vector<double>& b) const;

private:
    // BTF decomposition using graph reachability
    BTFResult decompose_btf(const SparseCSC<int>& A) const;

    // AMD ordering for a submatrix
    std::vector<int> amd_block(const SparseCSC<int>& A, int block_start, int block_end) const;

    // Sparse LU factorization for a single block
    NumericResult factorize_block(const SparseCSC<int>& A, int block_start, int block_end,
                                  const std::vector<int>& block_perm);

    // Backward substitution
    void solve_lower(const std::vector<int>& Li, const std::vector<double>& Lx,
                     const std::vector<int>& Lp, std::vector<double>& x) const;

    // Forward substitution
    void solve_upper(const std::vector<int>& Ui, const std::vector<double>& Ux,
                     const std::vector<int>& Up, std::vector<double>& x) const;

    double tol_;
    int scale_;
};

} // namespace klu

#endif // SOLVERS_KLU_H
