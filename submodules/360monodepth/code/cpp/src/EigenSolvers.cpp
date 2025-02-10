//
// Created by manuel on 11/10/2021.
//
#include<Eigen/SparseCore>
#include<Eigen/SparseCholesky>
#include<Eigen/SparseQR>
#include<Eigen/SparseLU>
#include<Eigen/IterativeLinearSolvers>
#include <iostream>
#include <memory>
#include <glog/logging.h>
#include "chrono"
#include <iomanip>
#include "pybind11/pybind11.h"
#include "pybind11/eigen.h"

class LinearSolver
{

public:
    // Supported Solvers. More can be added
    enum solverType
    {
        SimplicialLLT,
        SimplicialLDLT,
        SparseLU,
        ConjugateGradient,
        LeastSquaresConjugateGradient,
        BiCGSTAB
    };

    // Constructors
    LinearSolver(const Eigen::SparseMatrix<double>& A, solverType type) : A(A), type(type) {
        factorize();
    };

    LinearSolver(const Eigen::SparseMatrix<double>& A, solverType type, int maxiters, double tol) :
            A(A), type(type), maxiters(maxiters), tol(tol) {
        factorize();
    };

    LinearSolver(){
        type = solverType::BiCGSTAB;
    }

    explicit LinearSolver(solverType type) : type(type){ }

    ~LinearSolver() = default;

    // Compute Factorization of A matrix as soon as the object is created.
    void factorize(){
        switch (type) {
            case SimplicialLLT:
                solverLLT = std::make_unique<Eigen::SimplicialLLT<Eigen::SparseMatrix<double>>>();
                solverLLT->compute(A);
                break;
            case SimplicialLDLT:
                solverLDLT = std::make_unique<Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>>();
                solverLDLT->compute(A);
                break;
            case SparseLU:
                solverLU = std::make_unique<Eigen::SparseLU<Eigen::SparseMatrix<double>>>();
                solverLU->compute(A);
                break;
            case ConjugateGradient:
                solverCG = std::make_unique<Eigen::ConjugateGradient<Eigen::SparseMatrix<double>>>();
                solverCG->setMaxIterations(maxiters);
                solverCG->setTolerance(tol);
                solverCG->compute(A);
                break;
            case LeastSquaresConjugateGradient:
                solverLSCG = std::make_unique<Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>>>();
                solverLSCG->setMaxIterations(maxiters);
                solverLSCG->setTolerance(tol);
                solverLSCG->compute(A);
                break;
            case BiCGSTAB:
                solverBiCGSTAB = std::make_unique<Eigen::BiCGSTAB<Eigen::SparseMatrix<double>>>();
                solverBiCGSTAB->setMaxIterations(maxiters);
                solverBiCGSTAB->setTolerance(tol);
                solverBiCGSTAB->compute(A);
                break;
        }
    }

    // Solve the system for an arbitrary b vector
    Eigen::VectorXd solve(const Eigen::VectorXd& b) {
        Eigen::VectorXd x;
        auto start_time = std::chrono::system_clock::now();
        switch (LinearSolver::type) {
            case SimplicialLLT:
                LOG(INFO) << "Solving with SimplicialLLT ... \n";
                x = solverLLT->solve(b);
                break;
            case SimplicialLDLT:
                LOG(INFO) << "Solving with SimplicialLDLT ... \n";
                x = solverLDLT->solve(b);
                break;
            case SparseLU:
                LOG(INFO) << "Solving with Sparse LU ... \n";
                x = solverLU->solve(b);
                break;
            case ConjugateGradient:
                LOG(INFO) << "Solving with Conjugate Gradient ... \n";
                x = solverCG->solve(b);
                break;
            case LeastSquaresConjugateGradient:
                LOG(INFO) << "Solving with LeastSquaresConjugateGradient ... \n";
                x = solverLSCG->solve(b);
                break;
            case BiCGSTAB:
                LOG(INFO) << "Solving with BiCGSTAB ... \n";
                x = solverBiCGSTAB->solve(b);
                break;
        }

        std::chrono::duration<double> elapsed_time = std::chrono::system_clock::now() - start_time;
        double solving_time = elapsed_time.count();
        std::stringstream stream;
        stream << std::fixed << std::setprecision(5) << solving_time;
        std::string solving_time_str = stream.str();
        Eigen::Matrix<double, Eigen::Dynamic, 1> temp = A * x;
        rel_error = (b-temp).norm()/b.norm();
        LOG(INFO) << "Solving time   = " << solving_time_str << std::endl;
        LOG(INFO) << "Relative error = " << rel_error << std::endl;
        return x;
    }

    void setIters(int iters){
        maxiters = iters;
        switch (type) {
            case SimplicialLLT:
                LOG(WARNING) << "SimplicialLLT does not use maxiters";
                return;
            case SimplicialLDLT:
                LOG(WARNING) << "SimplicialLDLT does not use maxiters";
                return;
            case SparseLU:
                LOG(WARNING) << "SparseLU does not use maxiters";
                return;
            case ConjugateGradient:
                solverCG->setMaxIterations(maxiters);
                return;
            case LeastSquaresConjugateGradient:
                solverLSCG->setMaxIterations(maxiters);
                return;
            case BiCGSTAB:
                solverBiCGSTAB->setMaxIterations(maxiters);
                return;
        }
    }

    void setTol(const double& new_tol) {
        tol = new_tol;
        switch (type) {
            case SimplicialLLT:
                LOG(WARNING) << "SimplicialLLT does not use tol";
                return;
            case SimplicialLDLT:
                LOG(WARNING) << "SimplicialLDLT does not use tol";
                return;
            case SparseLU:
                LOG(WARNING) << "SparseLU does not use tol";
                return;
            case ConjugateGradient:
                solverCG->setTolerance(tol);
                return;
            case LeastSquaresConjugateGradient:
                solverLSCG->setTolerance(tol);
                return;
            case BiCGSTAB:
                solverBiCGSTAB->setTolerance(tol);
                return;
        }
    }

    std::string getType() {
        switch (type)
        {
            case SimplicialLLT:                 return "SimplicialLLT";
            case SimplicialLDLT:                return "SimplicialLDLT";
            case SparseLU:                      return "SparseLU";
            case ConjugateGradient:             return "ConjugateGradient";
            case LeastSquaresConjugateGradient: return "LeastSquaresConjugateGradient";
            case BiCGSTAB:                      return "BiCGSTAB";
            default:                            return "NO SOLVER SELECTED";
        }
    }

    double getTol() const { return tol; }
    int getMaxIters() const { return maxiters; }
    Eigen::SparseMatrix<double> getA() const { return A; }
    void setA (const Eigen::SparseMatrix<double>& new_A){
        A = new_A;
        factorize();
    }

private:
    solverType type;
    double rel_error = 0.0;

    //Only used for iterative solvers
    int maxiters = 100;
    double tol = 1e-08;

    Eigen::SparseMatrix<double> A;
//    std::shared_ptr<Eigen::internal::noncopyable> *solver;
    std::unique_ptr<Eigen::SimplicialLLT<Eigen::SparseMatrix<double>>> solverLLT;
    std::unique_ptr<Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>> solverLDLT;
    std::unique_ptr<Eigen::SparseLU<Eigen::SparseMatrix<double>>> solverLU;
    std::unique_ptr<Eigen::ConjugateGradient<Eigen::SparseMatrix<double>>> solverCG;
    std::unique_ptr<Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>>> solverLSCG;
    std::unique_ptr<Eigen::BiCGSTAB<Eigen::SparseMatrix<double>>> solverBiCGSTAB;
};


namespace py = pybind11;

PYBIND11_MODULE(EigenSolvers, m) {
    m.doc() = "Python binding for Eigen Linear System Solvers"; // optional module docstring

    py::class_<LinearSolver> linearSolver(m, "LinearSolver"); // Define class
    linearSolver.def(py::init<const Eigen::SparseMatrix<double>&, LinearSolver::solverType, int, double>()); //ctor
    linearSolver.def(py::init<const Eigen::SparseMatrix<double>&, LinearSolver::solverType>()); //ctor
    linearSolver.def(py::init<>()); //ctor
    linearSolver.def(py::init<LinearSolver::solverType>()); //ctor
    linearSolver.def("solve", &LinearSolver::solve);
    linearSolver.def_property("maxiters", &LinearSolver::getMaxIters, &LinearSolver::setIters);
    linearSolver.def_property("tol", &LinearSolver::getTol, &LinearSolver::setTol);
    linearSolver.def_property("A", &LinearSolver::getA, &LinearSolver::setA);
    linearSolver.def_property_readonly("solver_type", &LinearSolver::getType);

    py::enum_<LinearSolver::solverType>(linearSolver, "solverType") //enum type
            .value("SimplicialLLT", LinearSolver::solverType::SimplicialLLT)
            .value("SimplicialLDLT", LinearSolver::solverType::SimplicialLDLT)
            .value("SparseLU", LinearSolver::solverType::SparseLU)
            .value("ConjugateGradient", LinearSolver::solverType::ConjugateGradient)
            .value("LeastSquaresConjugateGradient", LinearSolver::solverType::LeastSquaresConjugateGradient)
            .value("BiCGSTAB", LinearSolver::solverType::BiCGSTAB)
            .export_values();
}