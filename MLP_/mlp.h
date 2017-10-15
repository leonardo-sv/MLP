#ifndef MLP_H
#define MLP_H


#include <iostream>
#include "helper.h"

#define No_func 0
#define Sigmoid 1
#define Hyper_Tang 2
#define Linear 3
#define CLASSIFICATION 4
#define REGRESSION 5
#define Yes true
#define No false

class mlp
{
public:
    struct neuron
    {
        double a;                   //neuron values pos activation function
        double z;                   //neuron values
        double error;               //error to minimize

        //constructor neuron
        neuron() : a(0.0), z(0.0), error(0.0){
        }
        void print(const unsigned int pos)
        {
            std::cout << "Neuron[" << pos << "]={"
                      << "a=" << a << " z=" << z
                      << " error=" << error << "}"<<std::endl;
        }
        //destructor
        ~neuron(){}
    };
    struct layer
    {
        std::vector<neuron> n;                                     //neurons
        std::vector<double> w;                                //weigths
        std::vector<double> g;                                //gradients
        std::vector<double> old_w;                            //old weigths to use in momentum
        double (*act_function)(double);                  //pointer to activatin function
        double (*derived)(double);                       //pointer to derived activatin function
        unsigned int num_neurons;                                  //number of neurons layer
        unsigned int num_weights;                                  //number of weights layer
        unsigned int w_per_neuron;                                 //weights associated each neuron

        layer() {}//std::cout << "Layer default constructor" << std::endl;}

        layer(const unsigned int num_neurons_, const unsigned int num_weights_, const unsigned int a):
            num_neurons(num_neurons_), num_weights(num_weights_)
        {
            w.resize(num_weights);
            old_w.resize(num_weights);
            g.resize(num_weights);
            n.resize(num_neurons);
            w_per_neuron = num_weights / num_neurons;

            switch (a) {
                case Sigmoid:
                {
                    act_function = &sigmoid;
                    derived      = &derived_sigmoid;
                    break;
                }
                case Hyper_Tang:
                {
                    act_function = &hyper_tg;
                    derived      = &derived_hyper_tg;
                    break;
                }
                case Linear:
                {
                    act_function = &linear;
                    derived      = &derived_linear;
                    break;
                }
            default:
                break;
            }

        }

        void w_init_rand(double init = -1.0, double end = 1.0){
            for(unsigned int i = 0; i < num_weights; i++){
                w[i] = rand_double(init, end);
            }
        }

//        void set_weights(std::vector<long long> v)
//        {
//            if(v.size() != num_weights) return;

//            for(size_t i = 0; i < v.size(); i++){
//                w[i] = v[i];
//            }

//        }

//        void gradient_zeros(){
//            for(size_t i = 0; i < num_weights; i++)
//            {
//                g[i] = 0.0;
//            }
//        }

        void print()
        {
            unsigned int k = 0;
            for(unsigned int i = 0; i < num_neurons; i++)
            {
                std::cout << std::endl;
                n[i].print(i);
                for(unsigned int j = 0; j < w_per_neuron; j++)
                {
                    std::cout << w[k] << std::endl;
                    k++;
                }
            }
        }
    };


    mlp(const unsigned int type, const double learning_rate, const double lambda, const double momentum,
        const double bias, const unsigned int epochs_, const std::vector<std::vector<unsigned int>> config);
    mlp();
    ~mlp();
    void print();
    void fit(std::vector<std::vector<double>> &X, std::vector<std::vector<double>> &y,
             const unsigned int epochs, const bool norm);

private:
    double learning_rate_;
    double momentum_;
    double lambda_;
    double bias_;
    unsigned int epochs_;
    unsigned int num_layers_;
    unsigned int M;
    unsigned int CV;
    unsigned int TEST;
    double (*function_error)(double, double);
    std::vector<layer> layers_;
    void l_random_weights();
    void save_weights(const unsigned int w);
    void set_weights_file(std::string path);
    void forward_propagation(const std::vector<double> &X);
    void add_biasX(std::vector<std::vector<double>> &X);
    void output_error(const std::vector<double> &y);
    void layer_error();
    void gradient(const std::vector<double> &X);
    void partial_derivatives();
//    void layers_grad_zeros();
    void back_propagation(const std::vector<std::vector<double>> &X,
                               const std::vector<std::vector<double>> &y);
    double cost_function(const std::vector<std::vector<double>> &X,
                              const std::vector<std::vector<double>> &y,
                              const unsigned int init, const unsigned int end);
    double regularization(const unsigned int M);
    void update_weights();
    void print_out();
};

#endif // MLP_H
