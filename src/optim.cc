#include "optim.h"
#include "epigraph.hpp"

// This example solves the portfolio optimization problem in QP form

using namespace cvx;

void Optimizer::main() {
  size_t n = 5; // Assets
  size_t m = 2; // Factors

  // Set up problem data.
  double gamma = 0.5;          // risk aversion parameter
  Eigen::VectorXd mu(n);       // vector of expected returns
  Eigen::MatrixXd F(n, m);     // factor-loading matrix
  Eigen::VectorXd D(n);        // diagonal of idiosyncratic risk
  Eigen::MatrixXd Sigma(n, n); // asset return covariance

  mu.setRandom();
  F.setRandom();
  D.setRandom();

  mu = mu.cwiseAbs();
  F = F.cwiseAbs();
  D = D.cwiseAbs();
  Sigma = F * F.transpose();
  Sigma.diagonal() += D;

  // Formulate QP.
  OptimizationProblem qp;

  // Declare variables with...
  // addVariable(name) for scalars,
  // addVariable(name, rows) for vectors and
  // addVariable(name, rows, cols) for matrices.
  VectorX x = qp.addVariable("x", n);

  // Available constraint types are equalTo(), lessThan(), greaterThan() and box()
  qp.addConstraint(greaterThan(x, 0.));
  qp.addConstraint(equalTo(x.sum(), 1.));

  // Make mu dynamic in the cost function so we can change it later
  qp.addCostTerm(x.transpose() * par(gamma * Sigma) * x - dynpar(mu).dot(x));

  // Print the problem formulation for inspection
  cout << qp << endl;

  // Create and initialize the solver instance.
  osqp::OSQPSolver solver(qp);

  // Print the canonical problem formulation for inspection
  cout << solver << endl;

  // Solve problem and show solver output
  const bool verbose = true;
  solver.solve(verbose);

  cout << "Solver message:  " << solver.getResultString() << endl;
  cout << "Solver exitcode: " << solver.getExitCode() << endl;

  // Call eval() to get the variable values
  cout << "Solution:\n" << eval(x) << endl;

  // Update data
  mu.setRandom();
  mu = mu.cwiseAbs();

  // Solve again
  // OSQP will warm start automatically
  solver.solve(verbose);

  cout << "Solution after changing the cost function:\n" << eval(x) << endl;
}