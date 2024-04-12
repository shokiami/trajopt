#include "optim.h"

Optimizer::Optimizer(Eigen::Vector3d r_i, Eigen::Vector3d r_f, Eigen::Vector3d v_i, Eigen::Vector3d v_f,
                     int n, double t_f, double u_min, double u_max, double theta_max, double m) :
                     n(n), t_f(t_f), u_min(u_min), u_max(u_max), theta_max(theta_max), m(m) {

  // variables
  for (int i = 0; i <= n; i++) {
    r.push_back(qp.addVariable("r_" + to_string(i), 3));
    v.push_back(qp.addVariable("v_" + to_string(i), 3));
    a.push_back(qp.addVariable("a_" + to_string(i), 3));
    u.push_back(qp.addVariable("u_" + to_string(i), 3));
    gamma.push_back(qp.addVariable("gamma_" + to_string(i)));
  }

  // cost function
  for (int i = 0; i <= n; i++){
    qp.addCostTerm(gamma[i]);
  }

  // dynamics
  Eigen::Vector3d g(0.0, 0.0, 9.81);
  for (int i = 0; i <= n; i++) {
    qp.addConstraint(equalTo(a[i], par(1.0 / m) * u[i] - par(g)));
  }
  double dt = t_f / n;
  for (int i = 0; i < n; i++) {
    qp.addConstraint(equalTo(r[i + 1], r[i] + par(1.0 / 2.0 * dt) * v[i] + par(1.0 / 2.0 * dt) * v[i + 1]));
    qp.addConstraint(equalTo(v[i + 1], v[i] + par(1.0 / 2.0 * dt) * a[i] + par(1.0 / 2.0 * dt) * a[i + 1]));
  }

  // control constraints
  for (int i = 0; i <= n; i++) {
    qp.addConstraint(lessThan(u[i].norm(), gamma[i]));
    qp.addConstraint(greaterThan(gamma[i], u_min));
    qp.addConstraint(lessThan(gamma[i], u_max));
    qp.addConstraint(lessThan(par(cos(theta_max)) * gamma[i], u[i](2)));
  }

  // initial conditions
  qp.addConstraint(equalTo(r[0], par(r_i)));
  qp.addConstraint(equalTo(v[0], par(v_i)));
  qp.addConstraint(equalTo(a[0], 0.0));

  // final conditions
  qp.addConstraint(equalTo(r[n], par(r_f)));
  qp.addConstraint(equalTo(v[n], par(v_f)));
  qp.addConstraint(equalTo(a[n], 0.0));
}

void Optimizer::solve() {
  osqp::OSQPSolver solver = osqp::OSQPSolver(qp);
  const bool verbose = true;
  solver.solve(verbose);
  cout << "Solver message: " << solver.getResultString() << endl;
  cout << "Solver exitcode: " << solver.getExitCode() << endl << endl;
  cout << "Solution:" << endl;
  cout << setprecision(4) << fixed;
  for (int i = 0; i <= n; i++) {
    Eigen::Vector3d control = eval(u[i]);
    Eigen::Vector3d point = eval(r[i]);
    controls.push_back(control);
    traj.push_back(point);
    cout << point(0) << " " << point(1) << " " << point(2) << endl;
  }
}

void Optimizer::save(string path) {
  ofstream csv = ofstream(path);
  csv << "r_x,r_y,r_z,u_x,u_y,u_z" << endl;
  for (int i = 0; i <= n; i++) {
    csv << traj[i](0) << "," << traj[i](1) << "," << traj[i](2) << ",";
    csv << controls[i](0) << "," << controls[i](1) << "," << controls[i](2) << endl;
  }
}
