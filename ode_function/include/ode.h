
//parameters for (explicit) Runge Kutta methods
struct rk_pars{
    std::vector<float> b_weights{1./6., 1./3., 1./3. , 1./6.};
    std::vector<float> c_weights{0., 0.5, 0.5 , 1.};
    std::vector<std::vector<float>> a_matrix{
            {},
            {0.5},
            {0., 0.5},
            {0., 0., 1.}
    };

    rk_pars()= default;
    rk_pars(std::vector<float> &_b_weights, std::vector<float> &_c_weights, std::vector<std::vector<float>> &_a_matrix):
            b_weights(_b_weights), c_weights(_c_weights), a_matrix(_a_matrix)
    {};
};


/*
 * Simple ODE of the form u'(x) = f(x, u(x))
 * with initial conditions u(0) = ud (Dirichlet)
 */
struct ode{
    func<float, float> u{};
    float ud = 1;
    std::function<float(float, float)> f_callable{};

    ode(func<float, float> &_u, float _ud, std::function<float(float, float)> &_f_callable): u(_u), ud(_ud), f_callable(_f_callable)
    {};
};


//solving algorithm: explicit Euler
void expeuler_solve(ode &ode_to_solve)
{
    //set initial condition
    ode_to_solve.u.set(std::pair<float, float>(0, ode_to_solve.ud));

    float u0 = ode_to_solve.ud;
    float u1 = u0;

    for(int i = 0; i < ode_to_solve.u.n_brackets-1 ; ++i)
    {
        u1 = u0 + ode_to_solve.u.delta* ode_to_solve.f_callable(ode_to_solve.u.arg(i), u0);
        ode_to_solve.u.set(std::pair<float, float>(i+1, u1));
        u0 = u1;
    }

    ode_to_solve.u.printvals();
}


//solving algorithm: explicit RK method
void exprk_solve(ode &ode_to_solve, rk_pars &rk)
{
    //set initial condition
    ode_to_solve.u.set(std::pair<float, float>(0, ode_to_solve.ud));

    unsigned int n_stages = rk.b_weights.size();
    float delta = ode_to_solve.u.delta;

    //stages vector to be reused
    std::vector<float> u_stages(n_stages, 0.);

    //bracket loop
    for (int i = 0; i < ode_to_solve.u.n_brackets-1; ++i)
    {
        //TODO: check if 0 fill necessary (not for explicit methods)
        std::fill(u_stages.begin(), u_stages.end(), 0.);

        //stage loop
        for (int s = 0; s < n_stages; ++s)
        {
            float f_xarg = ode_to_solve.u.arg(i) + delta*rk.c_weights[s];

            float f_uarg = 0.0;
            //matrix multiplication for a weights
            for (int l = 0; l < rk.a_matrix[s].size(); ++l)
            {
                f_uarg += rk.a_matrix[s][l]*u_stages[l];
            }
            f_uarg *= delta;
            f_uarg += ode_to_solve.u[i];

            //evaluation of stages
            u_stages[s] = ode_to_solve.f_callable(f_xarg, f_uarg);
        }

        //next function value
        float u_new = 0.;
        for (int n = 0; n < n_stages; ++n)
        {
            u_new += rk.b_weights[n]*u_stages[n];
        }
        u_new *= delta;
        u_new += ode_to_solve.u[i];

        ode_to_solve.u.set(std::pair<float, float>(i + 1, u_new));
    }

    ode_to_solve.u.printvals();
}


