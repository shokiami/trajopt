#ifndef OPTIM_H_
#define OPTIM_H_

#include "defs.h"

#define ETA_BUF 1e4
#define TRUST_BUF 1e2
#define CONV_EPS 1e-2

using namespace cvx;

class Optimizer {
  public:
  Optimizer(Vector3d r_i, Vector3d r_f, Vector3d v_i, Vector3d v_f, int n, double t_f, double u_min, double u_max,
            double theta_max, double mass, vector<pair<Vector3d, double>> obstacles);
  void init();
  void scp();
  bool solve();
  void save(string dir);

  private:
  Vector3d r_i;
  Vector3d r_f;
  Vector3d v_i;
  Vector3d v_f;
  int n;
  double t_f;
  double u_min;
  double u_max;
  double theta_max;
  double mass;
  vector<pair<Vector3d, double>> obstacles;

  OptimizationProblem qp;
  vector<VectorX> r;
  vector<VectorX> v;
  vector<VectorX> a;
  vector<VectorX> u;
  vector<Scalar> gamma;
  vector<vector<Scalar>> eta;
  vector<VectorX> trust;

  vector<Vector3d> controls;
  vector<Vector3d> traj;
};

#endif
