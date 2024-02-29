#include "optim.h"

Optimizer::Optimizer(Eigen::VectorXd r_i, Eigen::VectorXd r_f, Eigen::VectorXd v_i, Eigen::VectorXd v_f,
                     size_t n, double t_f, double u_max, double theta_max, double m) :
                     n(n), t_f(t_f), u_max(u_max), theta_max(theta_max), m(m) {

  // variables
  for (size_t i = 0; i <= n; i++) {
    r.push_back(qp.addVariable("r_" + to_string(i), 3));
    v.push_back(qp.addVariable("v_" + to_string(i), 3));
    a.push_back(qp.addVariable("a_" + to_string(i), 3));
    u.push_back(qp.addVariable("u_" + to_string(i), 3));
    gamma.push_back(qp.addVariable("gamma_" + to_string(i)));
  }

  // cost function
  for (size_t i = 0; i <= n; i++){
    qp.addCostTerm(gamma[i] * gamma[i]);
  }

  // dynamics
  Eigen::Vector3d g(0.0, 0.0, 9.81);
  for (size_t i = 0; i <= n; i++) {
    qp.addConstraint(equalTo(a[i], par(1.0 / m) * u[i] + par(g)));
  }
  double dt = t_f / n;
  for (size_t i = 0; i < n; i++) {
    qp.addConstraint(equalTo(r[i + 1], r[i] + par(dt) * v[i] + par(1.0 / 4.0 * dt * dt) * a[i] + par(1.0 / 4.0 * dt * dt) * a[i + 1]));
    qp.addConstraint(equalTo(v[i + 1], v[i] + par(1.0 / 2.0 * dt) * a[i] + par(1.0 / 2.0 * dt) * a[i + 1]));
  }

  // control constraints
  for (size_t i = 0; i <= n; i++) {
    qp.addConstraint(lessThan(a[i].norm(), gamma[i]));
    qp.addConstraint(greaterThan(gamma[i], 0.0));
    qp.addConstraint(lessThan(gamma[i], u_max));
    qp.addConstraint(lessThan(par(cos(theta_max)) * gamma[i], u[i](2)));
  }

  // initial conditions
  qp.addConstraint(equalTo(r[0], par(r_i)));
  qp.addConstraint(equalTo(v[0], par(v_i)));
  qp.addConstraint(equalTo(u[0], par(g)));

  // final conditions
  qp.addConstraint(equalTo(r[n], par(r_f)));
  qp.addConstraint(equalTo(v[n], par(v_f)));
  qp.addConstraint(equalTo(u[n], par(g)));
}

void Optimizer::solve() {
  osqp::OSQPSolver solver = osqp::OSQPSolver(qp);
  const bool verbose = true;
  solver.solve(verbose);
  cout << "Solver message: " << solver.getResultString() << endl;
  cout << "Solver exitcode: " << solver.getExitCode() << endl << endl;
  cout << "Solution:" << endl;
  cout << setprecision(3) << fixed;
  for (size_t i = 0; i <= n; i++) {
    cout << eval(r[i].transpose()) << endl;
  }
}
