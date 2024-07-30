#include "optim.h"

int main() {
  Eigen::Vector3d r_i(0.0, 0.0, 0.0);
  Eigen::Vector3d r_f(10.0, 10.0, 10.0);
  Eigen::Vector3d v_i(0.0, 0.0, 0.0);
  Eigen::Vector3d v_f(0.0, 0.0, 0.0);
  int n = 50;
  double t_f = 6;
  double u_min = 5;
  double u_max = 10;
  double theta_max = M_PI / 4;
  double mass = 1.0;

  vector<pair<Vector3d, double>> obstacles;
  obstacles.push_back({{3.0, 2.0, 5.0}, 2.0});
  obstacles.push_back({{5.0, 9.0, 9.0}, 4.0});

  Optimizer optim = Optimizer(r_i, r_f, v_i, v_f, n, t_f, u_min, u_max, theta_max, mass, obstacles);
  optim.scp();
  optim.save("data/");

  return 0;
}
