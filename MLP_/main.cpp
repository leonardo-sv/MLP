#include <iostream>
#include "files.h"
#include "mlp.h"
#include <vector>

using namespace std;

int main()
{
    double learning_rate = 0.6;
    double momentum = 0.7;
    double lambda = 1;
    double bias = 1.0;
    unsigned int epochs = 1000;
    vector<vector<double>> X;
    vector<vector<double>> y;
    vector<vector<unsigned int>> config = {{8, 0},
                                           {25, Sigmoid},
                                           {1, Sigmoid}};

    file_to_vector("../data/X.csv", X);
    file_to_vector("../data/y.csv", y);

//    cout << "IN:" << X.size() << " " << X[0].size() << endl;
//    cout << "OUT:" << y.size() << " " << y[0].size() << endl;

    mlp ann(REGRESSION, learning_rate,lambda, momentum, bias, epochs, config);
    //ann.print();
    ann.fit(X, y, epochs, Yes);


}
