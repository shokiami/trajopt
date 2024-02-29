#include "optim.h"

int main() {
  Eigen::Vector3d r_i(0.0, 0.0, 0.0);
  Eigen::Vector3d r_f(10.0, 10.0, 10.0);
  Eigen::Vector3d v_i(0.0, 0.0, 0.0);
  Eigen::Vector3d v_f(0.0, 0.0, 0.0);
  size_t n = 10;
  double t_f = 6;
  double u_max = 10.0;
  double theta_max = M_PI / 2;
  double m = 1.0;

  Optimizer optim = Optimizer(r_i, r_f, v_i, v_f, n, t_f, u_max, theta_max, m);
  optim.solve();

  return 0;
}
