#include "linear_system/klu/klu.h"
#include "linear_system/common/ordering.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <string>

namespace klu {
namespace {

constexpr double drop_tolerance = 1e-15;

void validate(const SparseCSC<int>& A) {
    if (A.n < 0 || A.Ap.size() != static_cast<size_t>(A.n + 1) ||
        A.Ap.empty() || A.Ap.front() != 0 || A.Ap.back() < 0 ||
        static_cast<size_t>(A.Ap.back()) != A.Ai.size() || A.Ai.size() != A.Ax.size()) {
        throw std::invalid_argument("KLU: invalid CSC matrix");
    }
    for (int j = 0; j < A.n; ++j) {
        if (A.Ap[j] > A.Ap[j + 1]) throw std::invalid_argument("KLU: invalid column pointers");
        int previous = -1;
        for (int p = A.Ap[j]; p < A.Ap[j + 1]; ++p) {
            if (A.Ai[p] < 0 || A.Ai[p] >= A.n || A.Ai[p] <= previous || !std::isfinite(A.Ax[p]))
                throw std::invalid_argument("KLU: row indices must be sorted, unique, and in range");
            previous = A.Ai[p];
        }
    }
}

struct Components {
    std::vector<int> id;
    std::vector<std::vector<int>> nodes;
};

Components strongly_connected_components(const SparseCSC<int>& A) {
    std::vector<std::vector<int>> graph(A.n);
    for (int j = 0; j < A.n; ++j)
        for (int p = A.Ap[j]; p < A.Ap[j + 1]; ++p)
            if (A.Ai[p] != j) graph[j].push_back(A.Ai[p]);

    std::vector<int> index(A.n, -1), low(A.n), stack, component(A.n, -1);
    std::vector<char> active(A.n, false);
    std::vector<std::vector<int>> nodes;
    int next = 0;
    std::function<void(int)> visit = [&](int v) {
        index[v] = low[v] = next++;
        stack.push_back(v); active[v] = true;
        for (int w : graph[v]) {
            if (index[w] < 0) { visit(w); low[v] = std::min(low[v], low[w]); }
            else if (active[w]) low[v] = std::min(low[v], index[w]);
        }
        if (low[v] == index[v]) {
            nodes.emplace_back();
            for (;;) {
                int w = stack.back(); stack.pop_back(); active[w] = false;
                component[w] = static_cast<int>(nodes.size()) - 1;
                nodes.back().push_back(w);
                if (w == v) break;
            }
        }
    };
    for (int v = 0; v < A.n; ++v) if (index[v] < 0) visit(v);
    // Tarjan emits sink components first.  This is the order required for
    // upper block triangular P*A*Q.
    return {std::move(component), std::move(nodes)};
}

std::vector<int> order_component(const SparseCSC<int>& A, const std::vector<int>& nodes) {
    if (nodes.size() < 3) return nodes;
    std::vector<int> local(A.n, -1);
    for (int k = 0; k < static_cast<int>(nodes.size()); ++k) local[nodes[k]] = k;
    std::set<std::pair<int, int>> unique_edges;
    for (int lj = 0; lj < static_cast<int>(nodes.size()); ++lj) {
        int j = nodes[lj];
        for (int p = A.Ap[j]; p < A.Ap[j + 1]; ++p) {
            int li = local[A.Ai[p]];
            if (li >= 0 && li != lj) unique_edges.emplace(std::min(li, lj), std::max(li, lj));
        }
    }
    std::vector<std::pair<int, int>> edges(unique_edges.begin(), unique_edges.end());
    auto permutation = linsys::amd_ordering(static_cast<int>(nodes.size()), edges);
    std::vector<int> result(nodes.size());
    // amd_ordering returns old -> new; symbolic Q is new -> old.
    for (int old = 0; old < static_cast<int>(nodes.size()); ++old) result[permutation[old]] = nodes[old];
    return result;
}

} // namespace

Solver::Solver() : tol_(0.001), scale_(2) {}

SymbolicResult Solver::analyze(const SparseCSC<int>& A) {
    validate(A);
    SymbolicResult sym{};
    sym.n = A.n;
    auto scc = strongly_connected_components(A);
    sym.nblocks = static_cast<int>(scc.nodes.size());
    sym.R.push_back(0);
    for (const auto& component : scc.nodes) {
        auto ordered = order_component(A, component);
        sym.Q.insert(sym.Q.end(), ordered.begin(), ordered.end());
        sym.R.push_back(static_cast<int>(sym.Q.size()));
    }
    // A simultaneous row/column permutation preserves the BTF structure.
    sym.P = sym.Q;
    sym.Pinv.assign(A.n, -1); sym.Qinv.assign(A.n, -1);
    for (int k = 0; k < A.n; ++k) {
        sym.Pinv[sym.P[k]] = k;
        sym.Qinv[sym.Q[k]] = k;
    }
    sym.etree.assign(A.n, -1);
    sym.Lp.assign(A.n + 1, 0);
    sym.Lnz.assign(A.n, 0);
    return sym;
}

NumericResult Solver::factorize(const SparseCSC<int>& A, const SymbolicResult& sym) {
    validate(A);
    if (sym.n != A.n || sym.P.size() != static_cast<size_t>(A.n) || sym.Q.size() != static_cast<size_t>(A.n))
        throw std::invalid_argument("KLU: symbolic analysis does not match matrix");

    const int n = A.n;
    NumericResult num{};
    num.n = n; num.nblocks = sym.nblocks; num.Qnum = sym.Q;
    num.Rs.assign(n, 1.0);
    if (scale_ != 0) {
        std::vector<double> sums(n, 0.0);
        for (int j = 0; j < n; ++j) for (int p = A.Ap[j]; p < A.Ap[j + 1]; ++p) {
            double a = std::abs(A.Ax[p]);
            if (scale_ == 1) sums[A.Ai[p]] += a;
            else sums[A.Ai[p]] = std::max(sums[A.Ai[p]], a);
        }
        for (int i = 0; i < n; ++i) if (sums[i] > 0.0) num.Rs[i] = sums[i];
    }

    // Rows are ordered maps: only structural nonzeros and fill are stored.
    std::vector<std::map<int, double>> rows(n), lower(n);
    for (int new_col = 0; new_col < n; ++new_col) {
        int old_col = sym.Q[new_col];
        for (int p = A.Ap[old_col]; p < A.Ap[old_col + 1]; ++p) {
            int new_row = sym.Pinv[A.Ai[p]];
            double value = A.Ax[p] / num.Rs[A.Ai[p]];
            if (value != 0.0) rows[new_row][new_col] = value;
        }
    }
    num.Pnum = sym.P;

    for (int k = 0; k < n; ++k) {
        int max_row = -1;
        double column_max = 0.0;
        for (int i = k; i < n; ++i) {
            auto it = rows[i].find(k);
            double value = it == rows[i].end() ? 0.0 : std::abs(it->second);
            if (value > column_max) { column_max = value; max_row = i; }
        }
        if (max_row < 0 || column_max == 0.0)
            throw std::runtime_error("KLU: matrix is singular at column " + std::to_string(sym.Q[k]));
        auto diagonal = rows[k].find(k);
        int pivot = (diagonal != rows[k].end() && std::abs(diagonal->second) >= tol_ * column_max) ? k : max_row;
        if (pivot != k) {
            std::swap(rows[pivot], rows[k]);
            std::swap(lower[pivot], lower[k]);
            std::swap(num.Pnum[pivot], num.Pnum[k]);
        }
        const double pivot_value = rows[k].at(k);
        for (int i = k + 1; i < n; ++i) {
            auto position = rows[i].find(k);
            if (position == rows[i].end()) continue;
            double multiplier = position->second / pivot_value;
            rows[i].erase(position);
            if (std::abs(multiplier) > drop_tolerance) lower[i][k] = multiplier;
            for (auto it = std::next(rows[k].find(k)); it != rows[k].end(); ++it) {
                double& target = rows[i][it->first];
                target -= multiplier * it->second;
                if (std::abs(target) <= drop_tolerance * std::max(1.0, std::abs(multiplier * it->second)))
                    rows[i].erase(it->first);
            }
        }
    }

    num.Pinv.assign(n, -1);
    for (int i = 0; i < n; ++i) num.Pinv[num.Pnum[i]] = i;
    num.Lp.assign(n + 1, 0); num.Up.assign(n + 1, 0);
    num.Llen.assign(n, 0); num.Ulen.assign(n, 0); num.Udiag.assign(n, 0.0);
    // Convert row-oriented factors to conventional CSC.
    std::vector<std::vector<std::pair<int, double>>> Lcols(n), Ucols(n);
    for (int i = 0; i < n; ++i) {
        Lcols[i].push_back({i, 1.0});
        for (const auto& [j, x] : lower[i]) Lcols[j].push_back({i, x});
        for (const auto& [j, x] : rows[i]) if (j >= i) Ucols[j].push_back({i, x});
    }
    for (int j = 0; j < n; ++j) {
        for (auto [i, x] : Lcols[j]) { num.Li.push_back(i); num.Lx.push_back(x); }
        num.Lp[j + 1] = static_cast<int>(num.Li.size()); num.Llen[j] = num.Lp[j + 1] - num.Lp[j];
        for (auto [i, x] : Ucols[j]) { num.Ui.push_back(i); num.Ux.push_back(x); if (i == j) num.Udiag[j] = x; }
        num.Up[j + 1] = static_cast<int>(num.Ui.size()); num.Ulen[j] = num.Up[j + 1] - num.Up[j];
    }
    num.lnz = static_cast<int>(num.Li.size()); num.unz = static_cast<int>(num.Ui.size());
    num.Lip = sym.R; num.Uip = sym.R;
    return num;
}

void Solver::solve_lower(const std::vector<int>& Li, const std::vector<double>& Lx,
                         const std::vector<int>& Lp, std::vector<double>& x) const {
    const int n = static_cast<int>(Lp.size()) - 1;
    for (int j = 0; j < n; ++j)
        for (int p = Lp[j]; p < Lp[j + 1]; ++p)
            if (Li[p] > j) x[Li[p]] -= Lx[p] * x[j];
}

void Solver::solve_upper(const std::vector<int>& Ui, const std::vector<double>& Ux,
                         const std::vector<int>& Up, std::vector<double>& x) const {
    const int n = static_cast<int>(Up.size()) - 1;
    for (int j = n - 1; j >= 0; --j) {
        double diagonal = 0.0;
        for (int p = Up[j]; p < Up[j + 1]; ++p) if (Ui[p] == j) diagonal = Ux[p];
        if (diagonal == 0.0) throw std::runtime_error("KLU: U is singular");
        x[j] /= diagonal;
        for (int p = Up[j]; p < Up[j + 1]; ++p) if (Ui[p] < j) x[Ui[p]] -= Ux[p] * x[j];
    }
}

std::vector<double> Solver::solve(const NumericResult& num, const std::vector<double>& b) const {
    if (b.size() != static_cast<size_t>(num.n)) throw std::invalid_argument("KLU: b dimension mismatch");
    std::vector<double> work(num.n), x(num.n);
    for (int i = 0; i < num.n; ++i) work[i] = b[num.Pnum[i]] / num.Rs[num.Pnum[i]];
    solve_lower(num.Li, num.Lx, num.Lp, work);
    solve_upper(num.Ui, num.Ux, num.Up, work);
    for (int j = 0; j < num.n; ++j) x[num.Qnum[j]] = work[j];
    return x;
}

std::vector<double> Solver::tsolve(const NumericResult& num, const std::vector<double>& b) const {
    if (b.size() != static_cast<size_t>(num.n)) throw std::invalid_argument("KLU: b dimension mismatch");
    std::vector<double> work(num.n), x(num.n);
    for (int j = 0; j < num.n; ++j) work[j] = b[num.Qnum[j]];
    // U' y = Q' b.
    for (int j = 0; j < num.n; ++j) {
        double diagonal = 0.0;
        for (int p = num.Up[j]; p < num.Up[j + 1]; ++p) {
            if (num.Ui[p] < j) work[j] -= num.Ux[p] * work[num.Ui[p]];
            else if (num.Ui[p] == j) diagonal = num.Ux[p];
        }
        if (diagonal == 0.0) throw std::runtime_error("KLU: U' is singular");
        work[j] /= diagonal;
    }
    // L' z = y (unit diagonal).
    for (int j = num.n - 1; j >= 0; --j)
        for (int p = num.Lp[j]; p < num.Lp[j + 1]; ++p)
            if (num.Li[p] > j) work[j] -= num.Lx[p] * work[num.Li[p]];
    for (int i = 0; i < num.n; ++i) x[num.Pnum[i]] = work[i] / num.Rs[num.Pnum[i]];
    return x;
}

BTFResult Solver::decompose_btf(const SparseCSC<int>& A) const {
    auto sym = const_cast<Solver*>(this)->analyze(A);
    BTFResult out{sym.nblocks, sym.P, sym.Q, sym.R, std::vector<int>(A.n)};
    for (int b = 0; b < out.nblocks; ++b)
        for (int k = out.R[b]; k < out.R[b + 1]; ++k) out.block_of_row[out.P[k]] = b;
    return out;
}

std::vector<int> Solver::amd_block(const SparseCSC<int>& A, int block_start, int block_end) const {
    std::vector<int> nodes(block_end - block_start);
    std::iota(nodes.begin(), nodes.end(), block_start);
    return order_component(A, nodes);
}

} // namespace klu
