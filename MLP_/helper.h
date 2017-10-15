#ifndef HELPER_H
#define HELPER_H

#include<vector>
#include<random>
#include<algorithm>

inline double rand_double(const double init , const double end)
{
    std::random_device rd;                              //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd());                             //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(init, end);    //determite range of distribution
    return (double)(dis(gen));
}

inline double sigmoid(const double z)
{
    return  1.0 / (1.0 + std::exp(-z));
}

inline double derived_sigmoid(const double z)
{
    return  sigmoid(z) * (1.0 - sigmoid(z));
}

inline double logistic_error(const double y, const double y_)
{
    return ((-y * std::log(y_)) - ((1.0 - y) * std::log(1.0 - y_)));
}

inline double hyper_tg(const double z)
{
    return (1.0 - std::exp(-z)) / (1.0 + std::exp(-z));
}


inline double derived_hyper_tg(const double z){
    double y = hyper_tg(z);
    return  (1.0 - (y * y));
}

inline double linear(const double z)
{
    return z;
}
inline double derived_linear(const double z)
{
    return 1.0;
}

inline double mean_square_error(double y, double y_){

    return std::sqrt(((y - y_) * (y - y_))) / 2.0;
}

inline void normalization(std::vector<std::vector<double>> &mat)
{
    std::vector<double> MAX(mat[0].size());

    for(size_t i = 0; i < mat.size(); i++)
    {
        for(size_t j = 0; j < mat[i].size(); j++)
        {
            if(mat[i][j] > MAX[j]) MAX[j] = mat[i][j];
        }
    }
    for(size_t i = 0; i < mat.size(); i++)
    {
        for(size_t j = 0; j < mat[i].size(); j++)
        {
            mat[i][j] = mat[i][j] / MAX[j];
        }
    }
}

inline void print_2DVec(const std::vector<std::vector<double>> &Vec2D)
{
    for(size_t i = 0; i < Vec2D.size(); i++)
    {
        for(size_t j = 0; j < Vec2D[i].size(); j++)
        {
            std::cout << Vec2D[i][j] <<" ";
        }
        std::cout << std::endl;
    }
}

inline void print_Vec(const std::vector<double> &Vec)
{
    for(size_t j = 0; j < Vec.size(); j++)
    {
        std::cout << Vec[j] <<" ";
    }
    std::cout << std::endl;

}
#endif // HELPER_H
