#include "mlp.h"
#include <fstream>
#include "files.h"

mlp::mlp(): learning_rate_(0.0), lambda_(0.0), momentum_(0.0), epochs_(0)
{

}


mlp::mlp(const unsigned int type, const double learning_rate, const double lambda, const double momentum,
         const double bias, const unsigned int epochs, const std::vector<std::vector<unsigned int>> config):
    learning_rate_(learning_rate), lambda_(lambda), momentum_(momentum), bias_(bias), epochs_(epochs)
{
    num_layers_ = config.size() - 1;
    layers_.resize(num_layers_);

    function_error = &logistic_error;
    switch (type) {
        case CLASSIFICATION: function_error = &logistic_error;
            break;
        case REGRESSION: function_error = &mean_square_error;
            break;
    default:
        break;
    }

    for(unsigned int i = 0; i < num_layers_; i++)
    {
        unsigned int size_back = config[i][0];
        unsigned int size_layer = config[i + 1][0];
        unsigned int num_weights = size_layer * (size_back + 1);
        layers_[i] = layer(size_layer, num_weights, config[i + 1][1]);
        layers_[i].w_per_neuron = num_weights/size_layer;
    }
    l_random_weights();
}

mlp::~mlp(){}

void mlp::l_random_weights()
{
    for(unsigned int i = 0; i < num_layers_; i++){
        layers_[i].w_init_rand();
    }
}

void mlp::add_biasX(std::vector<std::vector<double>> &X)
{
    //M = size of set of training examples
    size_t M = X.size();

    for(size_t i = 0; i < M; i++)
    {
        //add bias each training example
        X[i].push_back(bias_);
    }

}

void mlp::forward_propagation(const std::vector<double> &X)
{
    unsigned int w_control = layers_[0].w_per_neuron;
    for(size_t i = 0;  i < layers_[0].num_neurons; i++)
    {
        layers_[0].n[i].z = 0.0;
        for(size_t j = 0; j < X.size(); j++)
        {
            layers_[0].n[i].z = layers_[0].n[i].z + layers_[0].w[j + w_control * i] * X[j];
        }
        layers_[0].n[i].a = layers_[0].act_function(layers_[0].n[i].z);
    }

    for(size_t k = 1; k < num_layers_; k++)
    {
        w_control = layers_[k].w_per_neuron;
        for(size_t i = 0; i < layers_[k].num_neurons; i++)
        {
            unsigned int size_back_layer = layers_[k - 1].num_neurons;
            layers_[k].n[i].z = 0.0;
            for(size_t j = 0; j < size_back_layer; j++)
            {
                layers_[k].n[i].z = layers_[k].n[i].z + layers_[k].w[j + w_control * i] * layers_[k-1].n[j].a;
            }
            layers_[k].n[i].z = layers_[k].n[i].z + layers_[k].w[size_back_layer + w_control * i] * bias_;
            layers_[k].n[i].a = layers_[k].act_function(layers_[k].n[i].z);
        }
    }

}


void mlp::output_error(const std::vector<double>&y)
{
    unsigned int pos_output = num_layers_ - 1;

    for(size_t i = 0; i < layers_[pos_output].num_neurons; i++){
        layers_[pos_output].n[i].error = layers_[pos_output].n[i].a - y[i];
    }
}

void mlp::layer_error()
{
    unsigned int pos_front_layer = num_layers_ - 1;

    for(int k = pos_front_layer; k > 0; k--)
    {
        unsigned int size_back  = layers_[k - 1].num_neurons;
        unsigned int size_front = layers_[k].num_neurons;
        unsigned int w_control  = layers_[k].num_weights / layers_[k -1].num_neurons;
        for(size_t i = 0; i < size_back; i++)
        {
            for(size_t j = 0; j < size_front; j++)
            {
                layers_[k - 1].n[i].error = layers_[k - 1].n[i].error + (layers_[k].w[j + w_control * i]
                        * layers_[k].n[j].error);

             }
            layers_[k - 1].n[i].error = layers_[k - 1].n[i].error * layers_[k - 1].derived(layers_[k - 1].n[i].z);

        }
    }
}

void mlp::gradient(const std::vector<double> &X)
{
    unsigned int w_control = layers_[0].w_per_neuron;
    for(size_t i = 0; i < layers_[0].num_neurons; i++){
        for(size_t j = 0; j < X.size(); j++){
            layers_[0].g[j + w_control * i] = layers_[0].g[j + w_control * i]
                        + (layers_[0].n[i].error * X[j]);

        }
        layers_[0].n[i].error = 0.0;
    }

    for(size_t k = 1; k < num_layers_;k++)
    {
        w_control = layers_[k].w_per_neuron;
        for(size_t i  = 0; i < layers_[k].num_neurons; i++)
        {
            unsigned int back_size = layers_[k - 1].num_neurons;
            for(size_t j = 0; j < back_size; j++ )
            {
                layers_[k].g[j + w_control * i] = layers_[k].g[j + w_control * i]
                        + (layers_[k].n[i].error * layers_[k - 1].n[j].a);
            }
            layers_[k].g[back_size + w_control * i] = layers_[k].g[back_size + w_control * i]
                    + (layers_[k].n[i].error * bias_);
            layers_[k].n[i].error = 0.0;
        }

    }

}

void mlp::partial_derivatives()
{
    unsigned int w_bias, pos_bias;

    for(size_t k = 0; k < num_layers_; k++)
    {
        w_bias = layers_[k].w_per_neuron;
        pos_bias = w_bias - 1;
        for(size_t i = 0; i < layers_[k].num_weights; i++)
        {
            if(i != (pos_bias))
                layers_[k].g[i] = (layers_[k].g[i] / (double)M) +
                        ((lambda_ * layers_[k].w[i])/ (double)M);
            else
            {
                layers_[k].g[i] = (layers_[k].g[i] / (double)M);
                pos_bias = pos_bias + w_bias;
            }
        }

    }

}

//void mlp::layers_grad_zeros(){
//    for(size_t k = 0; k < num_layers_; k++)
//    {
//        layers_[k].gradient_zeros();
//    }
//}

void mlp::back_propagation(const std::vector<std::vector<double>> &X,
                           const std::vector<std::vector<double>> &y)
{

    for(size_t i = 0; i < M; i++)
    {
        forward_propagation(X[i]);
        output_error(y[i]);
        layer_error();
        gradient(X[i]);
    }
    partial_derivatives();

}

double mlp::cost_function(const std::vector<std::vector<double>> &X,
                               const std::vector<std::vector<double>> &y,
                               const unsigned int init, const unsigned int end)
{
    unsigned int output_pos = num_layers_ - 1 , M = end - init;
    double J = 0.0;
    for(size_t i = init; i < end; i ++){
        forward_propagation(X[i]);
        for(size_t j = 0; j < y[i].size(); j++)
        {
            J = J + function_error(y[i][j], layers_[output_pos].n[j].a);
        }

    }

    J = (J  / (double)M) + regularization(M);

    return J;

}

double mlp::regularization(const unsigned int M)
{
    double R = 0.0;
    unsigned int w_bias, pos_bias;

    for(size_t k = 0; k < num_layers_; k++)
    {   w_bias = layers_[k].w_per_neuron;
        pos_bias = w_bias - 1;
        for(size_t i = 0; i < layers_[k].num_weights; i++)
        {
            if(i != (pos_bias))
                R = R + (layers_[k].w[i] * layers_[k].w[i]);
            else
            {
                pos_bias = pos_bias + w_bias;
            }
        }

    }
    return (R * lambda_) / (2 * (double)M);
}

void mlp::update_weights()
{
    for(size_t k = 0; k < num_layers_; k++)
    {
        for(size_t i = 0; i < layers_[k].num_weights; i++)
        {
            layers_[k].w[i] = layers_[k].w[i] - (learning_rate_ * layers_[k].g[i]) +
                        (momentum_ * (layers_[k].w[i] - layers_[k].old_w[i]));
            layers_[k].old_w[i] = layers_[k].w[i];
            layers_[k].g[i] = 0.0;
        }
    }
}

void mlp::fit(std::vector<std::vector<double>> &X, std::vector<std::vector<double>> &y,
              const unsigned int epochs, const bool norm)
{
    if(norm){
        normalization(X);
        normalization(y);
    }
    add_biasX(X);
//    std::cout << "Entrada" <<std::endl;
//    print_2DVec(X);
//    std::cout << "Saida" <<std::endl;
//    print_2DVec(y);
    if(X.size() < 6){
        std::cout << "Set Training is smaller" << std::endl;
        return;
    }
    M    = (X.size() * 0.6);
    CV   = (X.size() * 0.2);
    TEST = (X.size() - (M + CV));
    double error_train;
    double error_cv;
    double old_error_cv = 0;

    for(size_t i = 0; i < epochs; i++)
    {
        back_propagation(X, y);
        error_train = cost_function(X, y, 0, M);
        error_cv    = cost_function(X, y, M, M + CV);
        update_weights();
        std::cout << "It:" << i << "Error:" << error_train << " " << error_cv << std::endl;
        save_value("../Error/Error_Train.txt","", error_train);
        save_value("../Error/Error_CV.txt","", error_cv);

    }

}

void mlp::save_weights(const unsigned int w)
{
    std::ofstream myfile ("../Weights/W" + std::to_string(w), std::ios::app);
    if (myfile.is_open())
    {
        for(size_t k = 0; k < num_layers_; k++)
        {
            for(size_t i = 0; i < layers_[k].num_weights; i++)
            {
                myfile << layers_[k].w[i] << " ";
            }
            myfile << std::endl;
        }

        myfile.close();
    }
    else std::cout << "ERROR: file not found" << std::endl;
}

void mlp::print_out()
{
    unsigned int pos_output = num_layers_ - 1;
    for(size_t i = 0; i < layers_[pos_output].num_neurons; i++)
    {
        std::cout << layers_[pos_output].n[i].a << " ";
    }
    std::cout << std::endl;
}

void mlp::print()
{
    for(unsigned int i = 0; i < num_layers_; i++){
        std::cout << "---------------Layer[" << i + 1<< "]---------------" ;
        layers_[i].print();
        std::cout << "-----------End Layer[" << i + 1 << "]---------------" << std::endl<< std::endl;
    }
}
