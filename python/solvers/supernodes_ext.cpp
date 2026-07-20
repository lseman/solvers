#include "linear_system/supernodes.h"

#include <Eigen/Dense>
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace {

using Int = int32_t;

// Elimination tree from an upper-triangular CSC pattern (ancestor
// path-compression), matching the algorithm used by supernodal_ldlt.h.
std::vector<Int> compute_etree(Int n, const std::vector<Int> &ap,
                               const std::vector<Int> &ai) {
    std::vector<Int> parent(static_cast<size_t>(n), Int{-1});
    std::vector<Int> ancestor(static_cast<size_t>(n), Int{-1});

    for (Int j = 0; j < n; ++j) {
        for (Int p = ap[static_cast<size_t>(j)]; p < ap[static_cast<size_t>(j) + 1]; ++p) {
            Int i = ai[static_cast<size_t>(p)];
            while (i != Int{-1} && i < j) {
                const Int next = ancestor[static_cast<size_t>(i)];
                ancestor[static_cast<size_t>(i)] = j;
                if (next == Int{-1}) {
                    parent[static_cast<size_t>(i)] = j;
                }
                i = next;
            }
        }
    }

    return parent;
}

// Build upper-triangular CSC (including the diagonal) from a dense symmetric
// matrix, keeping only entries with row <= col.
void upper_csc_from_dense(const Eigen::Ref<const Eigen::MatrixXd> &A, Int n,
                          std::vector<Int> &ap, std::vector<Int> &ai,
                          std::vector<double> &ax) {
    ap.assign(static_cast<size_t>(n) + 1, 0);
    ai.clear();
    ax.clear();

    for (Int j = 0; j < n; ++j) {
        ap[static_cast<size_t>(j)] = static_cast<Int>(ai.size());
        for (Int i = 0; i <= j; ++i) {
            const double v = A(i, j);
            if (v != 0.0) {
                ai.push_back(i);
                ax.push_back(v);
            }
        }
    }
    ap[static_cast<size_t>(n)] = static_cast<Int>(ai.size());
}

nb::dict identify_supernodes_dense(const Eigen::Ref<const Eigen::MatrixXd> &A,
                                   Int relax_abs, double relax_rel, double tau,
                                   Int max_size) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("A must be square");
    }
    const Int n = static_cast<Int>(A.rows());

    std::vector<Int> ap, ai;
    std::vector<double> ax;
    upper_csc_from_dense(A, n, ap, ai, ax);

    const std::vector<Int> etree = compute_etree(n, ap, ai);

    snode::SparseUpperCSC<Int> B{n, &ap, &ai, &ax};
    snode::Symbolic<Int> S{n, &etree, nullptr, nullptr};

    const auto info = snode::identify_supernodes<Int>(B, S, relax_abs, relax_rel, tau, max_size);

    nb::list ranges;
    for (const auto &r : info.ranges) {
        ranges.append(nb::make_tuple(r.first, r.second));
    }

    nb::dict out;
    out["ranges"] = ranges;
    out["col2sn"] = info.col2sn;
    out["etree"] = info.etree;
    out["post"] = info.post;
    out["num_supernodes"] = static_cast<int>(info.ranges.size());
    return out;
}

// Accepts scipy.sparse matrices (via .toarray()) or anything castable to a
// dense Eigen matrix.
nb::dict identify_supernodes_generic(nb::object A_obj, Int relax_abs, double relax_rel,
                                     double tau, Int max_size) {
    Eigen::MatrixXd A_dense;
    if (nb::hasattr(A_obj, "toarray")) {
        A_dense = nb::cast<Eigen::MatrixXd>(A_obj.attr("toarray")());
    } else {
        A_dense = nb::cast<Eigen::MatrixXd>(A_obj);
    }
    return identify_supernodes_dense(A_dense, relax_abs, relax_rel, tau, max_size);
}

} // namespace

NB_MODULE(supernodes, m) {
    m.doc() = "Standalone supernode detection (snode::identify_supernodes from supernodes.h)";

    m.def("identify_supernodes", &identify_supernodes_generic, nb::arg("A"),
          nb::arg("relax_abs") = 0, nb::arg("relax_rel") = 0.0, nb::arg("tau") = 1.0,
          nb::arg("max_size") = std::numeric_limits<Int>::max(),
          "Detect supernodes in a symmetric matrix's upper-triangular pattern "
          "(accepts numpy dense arrays or scipy.sparse matrices).\n"
          "Returns dict with ranges, col2sn, etree, post, num_supernodes.");
}
