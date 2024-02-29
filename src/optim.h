#ifndef OPTIM_H_
#define OPTIM_H_

#include "defs.h"
#include "epigraph.hpp"

using namespace cvx;

class Optimizer {
  public:
  Optimizer(Eigen::VectorXd r_i, Eigen::VectorXd r_f, Eigen::VectorXd v_i, Eigen::VectorXd v_f,
            size_t n, double t_f, double u_max, double theta_max, double m);
  void solve();

  private:
  Eigen::VectorXd r_i;
  Eigen::VectorXd r_f;
  Eigen::VectorXd v_i;
  Eigen::VectorXd v_f;
  size_t n;
  double t_f;
  double u_max;
  double theta_max;
  double m;

  OptimizationProblem qp;
  vector<VectorX> r;
  vector<VectorX> v;
  vector<VectorX> a;
  vector<VectorX> u;
  vector<Scalar> gamma;
};

#endif
