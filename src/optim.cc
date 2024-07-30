#include "optim.h"

Optimizer::Optimizer(Vector3d r_i, Vector3d r_f, Vector3d v_i, Vector3d v_f, int n, double t_f, double u_min,
                     double u_max, double theta_max, double mass, vector<pair<Vector3d, double>> obstacles) :
                     r_i(r_i), r_f(r_f), v_i(v_i), v_f(v_f), n(n), t_f(t_f), u_min(u_min), u_max(u_max),
                     theta_max(theta_max), mass(mass), obstacles(obstacles) {
  for (int i = 0; i <= n; i++) {
    traj.push_back((double) i / n * (r_f - r_i) + r_i);
  }
}

void Optimizer::init() {
  qp = OptimizationProblem();
  r = {};
  v = {};
  a = {};
  u = {};
  gamma = {};
  eta = {};
  trust = {};

  // variables
  for (int i = 0; i <= n; i++) {
    string idx = to_string(i);
    r.push_back(qp.addVariable("r_" + idx, 3));
    v.push_back(qp.addVariable("v_" + idx, 3));
    a.push_back(qp.addVariable("a_" + idx, 3));
    u.push_back(qp.addVariable("u_" + idx, 3));
    gamma.push_back(qp.addVariable("gamma_" + idx));
  }
  for (int i = 0; i <= n; i++) {
    vector<Scalar> row;
    for (int j = 0; j < obstacles.size(); j++) {
      string idx1 = to_string(i);
      string idx2 = to_string(j);
      row.push_back(qp.addVariable("eta_" + idx1 + "_" + idx2));
    }
    eta.push_back(row);
  }
  for (int i = 0; i <= n; i++) {
    string idx = to_string(i);
    trust.push_back(qp.addVariable("trust_" + idx, 3));
  }
  // for (int i = 0; i <= n; i++) {
  //   r[i] << traj[i](0), traj[i](1), traj[i](2);
  // }

  // cost function
  // qp.addCostTerm(0 * r[0]);
  for (int i = 0; i <= n; i++){
    qp.addCostTerm(gamma[i]);
  }

  for (int i = 0; i <= n; i++) {
    for (int j = 0; j < obstacles.size(); j++) {
      qp.addCostTerm(ETA_BUF * eta[i][j]);
    }
  }

  // dynamics
  Vector3d g(0.0, 0.0, 9.81);
  for (int i = 0; i <= n; i++) {
    qp.addConstraint(equalTo(a[i], par(1.0 / mass) * u[i] - par(g)));
  }
  double dt = t_f / n;
  double dt2 = dt * dt;
  for (int i = 0; i < n; i++) {
    qp.addConstraint(equalTo(r[i + 1], r[i] + par(dt) * v[i] + par(dt2 / 3.0) * a[i] + par(dt2 / 6.0) * a[i + 1]));
    qp.addConstraint(equalTo(v[i + 1], v[i] + par(dt / 2.0) * a[i] + par(dt / 2.0) * a[i + 1]));
  }

  // control constraints
  for (int i = 0; i <= n; i++) {
    qp.addConstraint(lessThan(u[i].norm(), gamma[i]));
    qp.addConstraint(box(par(u_min), gamma[i], par(u_max)));
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

  // obstacle conditions
  for (int i = 0; i <= n; i++) {
    for (int j = 0; j < obstacles.size(); j++) {
      Vector3d c = obstacles[j].first;
      double rad = obstacles[j].second;
      Vector3d d = traj[i] - c;
      RowVector3d dT = d.transpose();
      qp.addConstraint(lessThan(par(rad * rad - dT * d + 2.0 * dT * traj[i]) - par(2.0 * dT) * r[i], eta[i][j]));
      qp.addConstraint(greaterThan(eta[i][j], 0.0));
    }
  }

  for (int i = 0; i <= n; i++) {
    qp.addCostTerm(trust[i].transpose() * par(TRUST_BUF * Eigen::MatrixXd::Identity(3, 3)) * trust[i]);
  }
  for (int i = 0; i <= n; i++) {
    qp.addConstraint(equalTo(trust[i], r[i] - par(traj[i])));
  }
}

void Optimizer::scp() {
  bool converged = false;
  while (!converged) {
    init();
    converged = solve();
  }
  // init();
  // solve();

  cout << "Solution:" << endl;
  cout << setprecision(4) << fixed;
  for (Vector3d point : traj) {
    cout << point(0) << " " << point(1) << " " << point(2) << endl;
  }
}

bool Optimizer::solve() {
  osqp::OSQPSolver solver = osqp::OSQPSolver(qp);
  const bool verbose = true;
  solver.solve(verbose);
  cout << "Solver message: " << solver.getResultString() << endl;
  cout << "Solver exitcode: " << solver.getExitCode() << endl << endl;
  vector<Vector3d> new_traj;
  for (int i = 0; i <= n; i++) {
    Vector3d control = eval(u[i]);
    Vector3d point = eval(r[i]);
    controls.push_back(control);
    new_traj.push_back(point);
  }
  bool converged = false;
  if (traj.size() > 0) {
    converged = true;
    for (int i = 0; i < traj.size(); i++) {
      if ((new_traj[i] - traj[i]).norm() > CONV_EPS) {
        converged = false;
      }
    }
  }
  traj = new_traj;
  return converged;
}

void Optimizer::save(string dir) {
  ofstream obs_csv = ofstream(dir + "obs.csv");
  obs_csv << "c_x,c_y,c_z,r" << endl;
  for (pair<Vector3d, double> obstacle : obstacles) {
    Vector3d c = obstacle.first;
    double r = obstacle.second;
    obs_csv << c(0) << "," << c(1) << "," << c(2) << "," << r << endl;
  }
  ofstream traj_csv = ofstream(dir + "traj.csv");
  traj_csv << "r_x,r_y,r_z,u_x,u_y,u_z" << endl;
  for (int i = 0; i <= n; i++) {
    traj_csv << traj[i](0) << "," << traj[i](1) << "," << traj[i](2) << ",";
    traj_csv << controls[i](0) << "," << controls[i](1) << "," << controls[i](2) << endl;
  }
}
