#include <iostream>
#include <functional>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>


/*Function template
 * takes in two types (domain and codomain type)
 * takes in mapping x,y for bracketed function
 * implements simple manipulation (addition, multiplication, subtraction, translation, ...)
 */

template <typename D, typename C>
class func{
    //TODO: make private
public:
    D domain_start = 0.;
    int n_brackets = 10;
    D delta = 0.1;
private:
    std::vector<D> domain_vals;
    std::vector<C> codomain_vals;
public:

    func()= default;

    //copy constructor
    func(func<D,C> &other):
            n_brackets(other.n_brackets),
            domain_start(other.domain_start),
            delta(other.delta),
            domain_vals(other.domain_vals),
            codomain_vals(other.codomain_vals)
    {}

    //move constructor
    func(func<D,C> &&other):
            n_brackets(std::move(other.n_brackets)),
            domain_start(std::move(other.domain_start)),
            delta(std::move(other.delta)),
            domain_vals(std::move(other.domain_vals)),
            codomain_vals(std::move(other.codomain_vals))
    {}

    //construct from two vecs
    func(const std::vector<D> &_domain_vals, const std::vector<C> &_codomain_vals):
            n_brackets(_domain_vals.size()),
            domain_start(_domain_vals[0]),
            delta(_domain_vals[1]-_domain_vals[0]),
            domain_vals(std::move(_domain_vals)),
            codomain_vals(std::move(_codomain_vals))
    {
        if(domain_vals.size() != codomain_vals.size()){
            throw std::invalid_argument( "domain and codomain vector size mismatch" );
        }
    }

    //construct constant function on interval
    func(D _domain_start, int _n_brackets, D _delta, C _codomain_value) :
            n_brackets(_n_brackets),
            domain_start(_domain_start),
            delta(_delta),
            domain_vals(_n_brackets, _domain_start),
            codomain_vals(_n_brackets, _codomain_value)
    {
        D value = domain_start;
        for (D &it : domain_vals)
        {
            it = value;
            value += delta;
        }
    };

    //copy assignment operator
    func<D,C> &operator=(const func<D,C> &other){
        this->n_brackets = other.n_brackets;
        this->domain_start = other.domain_start;
        this->delta = other.delta;
        this->domain_vals = other.domain_vals;
        this->codomain_vals = other.codomain_vals;
        return *this;
    };

    //move assignment operator
    func<D,C> &operator=(func<D,C> &&other){
        this->n_brackets = std::move(other.n_brackets);
        this->domain_start = std::move(other.domain_start);
        this->delta = std::move(other.delta);
        this->domain_vals = std::move(other.domain_vals);
        this->codomain_vals = std::move(other.codomain_vals);
        return *this;
    }


    //print function mapping
    void printvals()
    {
        std::cout << "argument->image:" << std::endl;
        auto itd = domain_vals.begin();
        auto itcd = codomain_vals.begin();
        do
        {
            std::cout << *itd << "->" << *itcd  <<std::endl;
            ++itd;
            ++itcd;
        }
        while(itd != domain_vals.end());
    };

    //access both arguments and images of mapping
    D arg(int index){
        return domain_vals[index];
    };

    C operator[](int index) const {
        return codomain_vals[index];
    };

    void set(std::pair<int, C> index_val){
        codomain_vals[index_val.first] = index_val.second;
    };

    //translate function periodically
    void translate(int n_step){
        std::rotate(codomain_vals.begin(), codomain_vals.begin()+n_step, codomain_vals.end() );
    };


    //add functions (if possible)
    func<D,C> operator+(const func<D,C> other){
        if (havesamedomain(*this, other))
        {
            std::vector<C> new_codomain_vals(this->codomain_vals);
            for (int i = 0; i < this->codomain_vals.size(); ++i)
            {
                new_codomain_vals[i] = other.codomain_vals[i] + this->codomain_vals[i];
            }
            return func<D,C>(this->domain_vals, new_codomain_vals);
        }
        else
        {
            throw std::invalid_argument( "unable to add: domains do not match" );
        }
        return *this;
    };

    //comparison operator for floats
    bool operator==(const func<D,float> f2){
        if (havesamedomain(*this, f2))
        {
            std::vector<float> diff(this->codomain_vals);
            std::transform(diff.begin(), diff.end(), f2.codomain_vals.begin(), diff.begin(), std::minus<>());

            if(std::all_of(diff.begin(), diff.end(),
                           [](float diff_val){return (std::fabs(diff_val) < std::numeric_limits<float>::epsilon());}
            )
                    )
            {
                return true;
            }
        }
        return false;
    };


    template<typename F>
    void applyfunctograph(F functoapply){

        auto itd = domain_vals.begin();
        auto itcd = codomain_vals.begin();
        do
        {
            *itcd = functoapply(*itd, *itcd);
            ++itd;
            ++itcd;
        }
        while(itd != domain_vals.end());
    }


};

//check whether functions have similar domain
template <typename D, typename C>
bool havesamedomain(const func<D,C> &f1, const func<D,C> &f2){
    if( (f1.domain_start == f2.domain_start)  && (f1.delta == f2.delta) )
    {
        return true;
    }
    else
    {
        return false;
    }
};

//specialization for floats
template <typename C>
bool havesamedomain(const func<float,C> &f1, const func<float,C> &f2){
    if( std::fabs(f1.domain_start - f2.domain_start) < std::numeric_limits<float>::epsilon()
        && std::fabs(f1.delta - f2.delta) < std::numeric_limits<float>::epsilon()
        && (f1.n_brackets == f2.n_brackets))
    {
        return true;
    }
    else
    {
        return false;
    }
};
