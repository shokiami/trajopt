#include "optim.h"

Optimizer::Optimizer(Eigen::VectorXd r_i, Eigen::VectorXd r_f, Eigen::VectorXd v_i, Eigen::VectorXd v_f,
                     size_t n, double t_f, double u_max, double theta_max, double m) :
                     n(n), t_f(t_f), u_max(u_max), theta_max(theta_max), m(m) {
  Eigen::Vector3d g(0.0, 0.0, 9.81);
  double dt = t_f / n;

  // variables
  r = qp.addVariable("r", 3, n + 1);
  v = qp.addVariable("v", 3, n + 1);
  a = qp.addVariable("a", 3, n + 1);
  u = qp.addVariable("u", 3, n + 1);
  gamma = qp.addVariable("gamma", n + 1);

  // cost function
  qp.addCostTerm(gamma.squaredNorm());

  // dynamics
  for (size_t i = 0; i <= n; i++) {
    qp.addConstraint(equalTo(a.col(i), par(1.0 / m) * u.col(i) + par(g)));
  }
  for (size_t i = 0; i < n; i++) {
    qp.addConstraint(equalTo(r.col(i + 1), r.col(i) + par(dt) * v.col(i) + par(1.0 / 4.0 * dt * dt) * a.col(i) + par(1.0 / 4.0 * dt * dt) * a.col(i + 1)));
    qp.addConstraint(equalTo(v.col(i + 1), v.col(i) + par(1.0 / 2.0 * dt) * a.col(i) + par(1.0 / 2.0 * dt) * a.col(i + 1)));
  }

  // control constraints
  for (size_t i = 0; i <= n; i++) {
    qp.addConstraint(lessThan(a.col(i).norm(), gamma(i)));
    qp.addConstraint(greaterThan(gamma(i), 0.0));
    qp.addConstraint(lessThan(gamma(i), u_max));
    qp.addConstraint(lessThan(par(cos(theta_max)) * gamma(i), u.col(i)(2)));
  }

  // initial conditions
  qp.addConstraint(equalTo(r.col(0), par(r_i)));
  qp.addConstraint(equalTo(v.col(0), par(v_i)));
  qp.addConstraint(equalTo(u.col(0), par(g)));

  // final conditions
  qp.addConstraint(equalTo(r.col(n), par(r_f)));
  qp.addConstraint(equalTo(v.col(n), par(v_f)));
  qp.addConstraint(equalTo(u.col(n), par(g)));
}

void Optimizer::solve() {
  osqp::OSQPSolver solver = osqp::OSQPSolver(qp);
  const bool verbose = true;
  solver.solve(verbose);
  cout << "Solver message: " << solver.getResultString() << endl;
  cout << "Solver exitcode: " << solver.getExitCode() << endl << endl;
  cout << "Solution:" << endl;
  cout << std::setprecision(3) << std::fixed;
  for (size_t i = 0; i <= n; i++) {
    cout << eval(r.col(i).transpose()) << endl;
  }
}
