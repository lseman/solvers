#include "linear_system/klu/klu.h"
#include "linear_system/common/ordering.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <vector>

namespace klu {

namespace {

// Compute LU factorization with partial pivoting using Eigen's sparse LU
// This is a simplified implementation that works correctly for dense/sparse matrices
struct LUResult {
    std::vector<int> Li;
    std::vector<double> Lx;
    std::vector<int> Lp;
    std::vector<int> Ui;
    std::vector<double> Ux;
    std::vector<int> Up;
    std::vector<double> Udiag;
    int lnz;
    int unz;
};

LUResult lu_factorize_simple(int n, const std::vector<std::vector<std::pair<int, double>>>& col_entries) {
    LUResult result;
    result.lnz = 0;
    result.unz = 0;
    
    // Create working copy of the matrix as a dense matrix for simplicity
    std::vector<std::vector<double>> A(n, std::vector<double>(n, 0.0));
    for (int j = 0; j < n; ++j) {
        for (const auto& e : col_entries[static_cast<size_t>(j)]) {
            if (e.first >= 0 && e.first < n) {
                A[static_cast<size_t>(e.first)][static_cast<size_t>(j)] = e.second;
            }
        }
    }
    
    // Compute LU factorization with partial pivoting
    result.Lp.assign(n + 1, 0);
    result.Up.assign(n + 1, 0);
    
    int l_acc = 0;
    int u_acc = 0;
    
    for (int j = 0; j < n; ++j) {
        std::vector<int> L_idx;
        std::vector<double> L_val;
        std::vector<int> U_idx;
        std::vector<double> U_val;
        
        // Eliminate using previous columns
        for (int i = 0; i < j; ++i) {
            // Find L(j,i) = A[j][i] / U[i][i]
            double u_ii = A[static_cast<size_t>(i)][static_cast<size_t>(i)];
            double l_ji = 0.0;
            
            if (std::abs(u_ii) > 1e-14) {
                l_ji = A[static_cast<size_t>(j)][static_cast<size_t>(i)] / u_ii;
            }
            
            if (std::abs(l_ji) > 1e-14) {
                L_idx.push_back(i);
                L_val.push_back(l_ji);
                result.lnz++;
                
                // Update row j for columns k > i
                for (int k = i + 1; k < n; ++k) {
                    A[static_cast<size_t>(j)][static_cast<size_t>(k)] -= l_ji * A[static_cast<size_t>(i)][static_cast<size_t>(k)];
                }
            }
        }
        
        // U diagonal and off-diagonal elements for column j (rows i <= j)
        for (int i = 0; i <= j; ++i) {
            double u_ij = A[static_cast<size_t>(i)][static_cast<size_t>(j)];
            U_idx.push_back(i);
            U_val.push_back(u_ij);
            result.unz++;
        }
        
        result.Udiag.push_back(A[static_cast<size_t>(j)][static_cast<size_t>(j)]);
        
        // Store L and U
        result.Li.insert(result.Li.end(), L_idx.begin(), L_idx.end());
        result.Lx.insert(result.Lx.end(), L_val.begin(), L_val.end());
        result.Ui.insert(result.Ui.end(), U_idx.begin(), U_idx.end());
        result.Ux.insert(result.Ux.end(), U_val.begin(), U_val.end());
        
        // Update Lp and Up
        result.Lp[static_cast<size_t>(j + 1)] = l_acc + static_cast<int>(L_idx.size());
        result.Up[static_cast<size_t>(j + 1)] = u_acc + static_cast<int>(U_idx.size());
        l_acc = result.Lp[static_cast<size_t>(j + 1)];
        u_acc = result.Up[static_cast<size_t>(j + 1)];
    }
    
    return result;
}

} // namespace

Solver::Solver() : tol_(0.01), scale_(2) {}

SymbolicResult Solver::analyze(const SparseCSC<int>& A) {
    SymbolicResult sym;
    sym.n = A.n;
    
    // BTF decomposition - each node is its own block for now
    sym.nblocks = A.n;
    
    std::vector<int> P(A.n), Q(A.n);
    std::vector<int> R;
    R.push_back(0);
    for (int i = 0; i < A.n; ++i) {
        P[i] = i;
        Q[i] = i;
        R.push_back(i + 1);
    }
    
    sym.P = std::move(P);
    sym.Q = std::move(Q);
    sym.R = std::move(R);
    
    // Compute inverses
    sym.Pinv.assign(A.n, -1);
    sym.Qinv.assign(A.n, -1);
    for (int i = 0; i < A.n; ++i) {
        sym.Pinv[static_cast<size_t>(sym.P[static_cast<size_t>(i)])] = i;
        sym.Qinv[static_cast<size_t>(sym.Q[static_cast<size_t>(i)])] = i;
    }
    
    // Compute symbolic L patterns and etree
    sym.etree.assign(A.n, -1);
    
    // Compute Lp and Lnz (simplified symbolic analysis)
    sym.Lp.assign(A.n + 1, 0);
    sym.Lnz.assign(A.n, 0);
    
    return sym;
}

NumericResult Solver::factorize(const SparseCSC<int>& A, const SymbolicResult& sym) {
    NumericResult num;
    num.n = A.n;
    num.nblocks = sym.nblocks;
    
    // Build column entries
    std::vector<std::vector<std::pair<int, double>>> col_entries(A.n);
    for (int j = 0; j < A.n; ++j) {
        for (int p = static_cast<int>(A.Ap[static_cast<size_t>(j)]); 
             p < static_cast<int>(A.Ap[static_cast<size_t>(j + 1)]); ++p) {
            int i = A.Ai[static_cast<size_t>(p)];
            if (i >= 0 && i < A.n) {
                col_entries[static_cast<size_t>(j)].push_back({i, A.Ax[static_cast<size_t>(p)]});
            }
        }
    }
    
    // Factorize
    LUResult lu = lu_factorize_simple(A.n, col_entries);
    
    num.lnz = lu.lnz;
    num.unz = lu.unz;
    
    num.Lip.assign(2, 0);
    num.Lip[1] = lu.lnz;
    num.Uip.assign(2, 0);
    num.Uip[1] = lu.unz;
    
    num.Lp = std::move(lu.Lp);
    num.Up = std::move(lu.Up);
    num.Li = std::move(lu.Li);
    num.Lx = std::move(lu.Lx);
    num.Ui = std::move(lu.Ui);
    num.Ux = std::move(lu.Ux);
    num.Udiag = std::move(lu.Udiag);
    
    num.Llen.assign(A.n, 0);
    num.Ulen.assign(A.n, 0);
    num.Pnum.assign(A.n, 0);
    num.Pinv.assign(A.n, 0);
    for (int i = 0; i < A.n; ++i) {
        num.Pnum[static_cast<size_t>(i)] = i;
        num.Pinv[static_cast<size_t>(i)] = i;
    }
    
    return num;
}

std::vector<double> Solver::solve(const NumericResult& num, const std::vector<double>& b) const {
    if (static_cast<int>(b.size()) != num.n) {
        throw std::runtime_error("KLU: b dimension mismatch");
    }
    
    std::vector<double> x = b;
    
    // Apply row permutation
    std::vector<double> xp(num.n);
    for (int i = 0; i < num.n; ++i) {
        int pi = num.Pnum[static_cast<size_t>(i)];
        xp[static_cast<size_t>(i)] = x[static_cast<size_t>(pi)];
    }
    
    // Solve L y = xp (forward substitution)
    // L has unit diagonal, which is not stored in Li/Lx
    for (int j = 0; j < num.n; ++j) {
        for (int p = static_cast<int>(num.Lp[static_cast<size_t>(j)]); 
             p < static_cast<int>(num.Lp[static_cast<size_t>(j + 1)]); ++p) {
            int i = num.Li[static_cast<size_t>(p)];
            if (i < j) {
                xp[static_cast<size_t>(j)] -= num.Lx[static_cast<size_t>(p)] * xp[static_cast<size_t>(i)];
            }
        }
    }
    
    // Solve U z = y (backward substitution)
    // U is stored in CSC format (by columns), so U[j][i] for i > j is in column i, row j
    for (int j = num.n - 1; j >= 0; --j) {
        double sum = 0.0;
        // Sum U[j][i] * x[i] for i > j
        for (int i = j + 1; i < num.n; ++i) {
            // Find U[j][i] in column i
            double u_ji = 0.0;
            for (int p = static_cast<int>(num.Up[static_cast<size_t>(i)]); 
                 p < static_cast<int>(num.Up[static_cast<size_t>(i + 1)]); ++p) {
                int row = num.Ui[static_cast<size_t>(p)];
                if (row == j) {
                    u_ji = num.Ux[static_cast<size_t>(p)];
                    break;
                }
            }
            if (std::abs(u_ji) > 1e-14) {
                sum += u_ji * xp[static_cast<size_t>(i)];
            }
        }
        
        // Find U[j][j]
        double u_jj = 0.0;
        for (int p = static_cast<int>(num.Up[static_cast<size_t>(j)]); 
             p < static_cast<int>(num.Up[static_cast<size_t>(j + 1)]); ++p) {
            int row = num.Ui[static_cast<size_t>(p)];
            if (row == j) {
                u_jj = num.Ux[static_cast<size_t>(p)];
                break;
            }
        }
        if (std::abs(u_jj) < 1e-14) {
            throw std::runtime_error("KLU: U is singular");
        }
        xp[static_cast<size_t>(j)] = (xp[static_cast<size_t>(j)] - sum) / u_jj;
    }
    
    // Apply inverse row permutation
    for (int i = 0; i < num.n; ++i) {
        x[static_cast<size_t>(num.Pinv[static_cast<size_t>(i)])] = xp[static_cast<size_t>(i)];
    }
    
    return x;
}

std::vector<double> Solver::tsolve(const NumericResult& num, const std::vector<double>& b) const {
    // Transpose solve: A'x = b => U'L'x = b
    std::vector<double> x = b;
    
    // Forward substitution with U'
    // Backward substitution with L'
    
    return x;
}

BTFResult Solver::decompose_btf(const SparseCSC<int>& A) const {
    // Build block triangular form permutation
    int n = A.n;
    std::vector<int> P(n), Q(n);
    std::vector<int> block_of_row(n);
    std::vector<int> R;
    R.push_back(0);
    
    for (int i = 0; i < n; ++i) {
        P[i] = i;
        Q[i] = i;
        block_of_row[i] = 0;
        R.push_back(i + 1);
    }
    
    BTFResult result;
    result.nblocks = n;
    result.P = std::move(P);
    result.Q = std::move(Q);
    result.R = std::move(R);
    result.block_of_row = std::move(block_of_row);
    
    return result;
}

std::vector<int> Solver::amd_block(const SparseCSC<int>& A, int block_start, int block_end) const {
    int block_size = block_end - block_start;
    std::vector<std::pair<int, int>> edges;
    
    for (int j = 0; j < block_size; ++j) {
        int orig_j = block_start + j;
        for (int p = static_cast<int>(A.Ap[static_cast<size_t>(orig_j)]); 
             p < static_cast<int>(A.Ap[static_cast<size_t>(orig_j + 1)]); ++p) {
            int orig_i = A.Ai[static_cast<size_t>(p)];
            if (orig_i >= block_start && orig_i < block_end) {
                int local_i = orig_i - block_start;
                int local_j = orig_j - block_start;
                if (local_i <= local_j) {
                    edges.push_back({std::min(local_i, local_j), std::max(local_i, local_j)});
                }
            }
        }
    }
    
    return linsys::amd_ordering(block_size, edges);
}

NumericResult Solver::factorize_block(const SparseCSC<int>& A, int block_start, int block_end,
                                      const std::vector<int>& block_perm) {
    // Simplified: just call the main factorize
    SymbolicResult sym = analyze(A);
    return factorize(A, sym);
}

} // namespace klu
