#include "func.h"
#include "ode.h"


int main() {

    float interval_start = 0.;
    float delta = 0.1;
    int n_brackets = 5;


    //testing construction from range
    func<float, float> f_callable(interval_start, n_brackets, delta, 0.);
    f_callable.printvals();


    //testing construction from two vecs
    std::vector<float> dvals(n_brackets,interval_start);
    float value = interval_start;
    for (float &it : dvals)
    {
        it = value;
        value += delta;
    }
//    std::vector<float> cdvals(n_brackets,0.3);
//    std::vector<float> cdvals(n_brackets+1,0.3);
    std::vector<float> cdvals{0.0, 0.1, 0.2, 0.3, 0.4};
    func<float, float> f1(dvals, cdvals);
    std::cout << "f1:" << std::endl;
    f1.printvals();


    //testing translation and copy constructor
    func<float, float> f1copy(f1);
    std::cout << "original f1:" << std::endl;
    f1.printvals();
    f1.translate(3);
    std::cout << "translated f1:" << std::endl;
    f1.printvals();
    std::cout << "original f1 copy:" << std::endl;
    f1copy.printvals();

    //testing addition
//    func<float, float> fdiff(interval_start, n_brackets-1, delta, 1.);
//    func<float, float> fsum = f1 + fdiff;
    std::cout << "f1 + f1copy:" << std::endl;
    func<float, float> fsum = f1 + f1copy;
    fsum.printvals();


//
//
//    //check boolean operator
    func<float, float> f1mod(f1);
    bool mod_equal = (f1mod == f1);
    std::cout << mod_equal << std::endl;
    float f1_firstval = f1[0] - 0.5 * std::numeric_limits<float>::epsilon();
    f1mod.set(std::pair<int, float>(0, f1_firstval));

    func<float, float> numshift(interval_start, n_brackets, delta, 0.5 * std::numeric_limits<float>::epsilon());

    f1mod = f1 + numshift;
    mod_equal = (f1mod == f1);
    std::cout << mod_equal << std::endl;
    f1mod.set(std::pair<int, float>(0, 100.));
    mod_equal = (f1mod == f1);
    std::cout << mod_equal << std::endl;



    //test function application
    auto square = [](float x, float y){return y*y;};
//    auto f1square = f1.applyfunctograph(square);
    std::cout << "f1" << std::endl;
    f1.printvals();
    f1.applyfunctograph(square);
    std::cout << "f1=f1*f1" << std::endl;
    f1.printvals();
//    f1square.printvals();

    return 0;
}
