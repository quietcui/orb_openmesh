#include "Flattening.h"
#include "MeshUtils.h"
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <cmath>

// ====================================================================
// ====================================================================


    for(const auto& path : paths) {
        for(size_t i = 1; i < path.size() - 1; ++i) {
    }
    }



        for (int k = 0; k < num_copies; ++k) {
        }
    }


        }

        }



            } else {
            }
        }
        }
    }

}

// ====================================================================
// ====================================================================
) {



    }


    Eigen::SparseMatrix<double> L_cut;

    }



    }



        }
    }


    }

            free_indices.push_back(i);
    }
    }

    std::vector<Eigen::Triplet<double>> L_inner_triplets;

    for (int k = 0; k < L_cut.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(L_cut, k); it; ++it) {


                }
            }
        }
    }


    L_inner.setFromTriplets(L_inner_triplets.begin(), L_inner_triplets.end());


    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(L_inner);
    if(solver.info() != Eigen::Success) {
    }



    }


}