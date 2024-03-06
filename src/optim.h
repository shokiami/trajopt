#ifndef OPTIM_H_
#define OPTIM_H_

#include "defs.h"

using namespace cvx;

class Optimizer {
  public:
  Optimizer(Eigen::Vector3d r_i, Eigen::Vector3d r_f, Eigen::Vector3d v_i, Eigen::Vector3d v_f,
            int n, double t_f, double u_min, double u_max, double theta_max, double m);
  void solve();
  void save(string path);

  private:
  Eigen::Vector3d r_i;
  Eigen::Vector3d r_f;
  Eigen::Vector3d v_i;
  Eigen::Vector3d v_f;
  int n;
  double t_f;
  double u_min;
  double u_max;
  double theta_max;
  double m;

  OptimizationProblem qp;
  vector<VectorX> r;
  vector<VectorX> v;
  vector<VectorX> a;
  vector<VectorX> u;
  vector<Scalar> gamma;
  
  vector<Eigen::Vector3d> controls;
  vector<Eigen::Vector3d> traj;
};

#endif
