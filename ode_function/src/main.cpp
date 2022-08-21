#include "func.h"
#include "ode.h"


int main() {

    float interval_start = 0.;
    float delta = 0.1;
    int n_brackets = 5;

    /*
     * Example of solving simple 1D domain ODEs with Dirichlet BC
     */

      // graph function on the RHS of ODE
    std::function<float(float, float)> f_right = [](float x, float y){return y;};
    func<float, float> u(interval_start, n_brackets, delta, 0.);
    ode dirichletexp_ode(u, 1., f_right);

    //default RK4 parameters
    rk_pars rk;

      //solve using explicit Euler or Runge Kutta
    expeuler_solve(dirichletexp_ode);
    exprk_solve(dirichletexp_ode, rk);

      return 0;
}
