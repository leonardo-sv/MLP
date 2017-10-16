
#include "files.h"
#include <regex>
#include <fstream>


void file_to_vector(std::string path, std::vector<std::vector<double>> &v){
    std::string line;
    std::ifstream myfile (path);
    if (myfile.is_open())
    {
        while (std::getline(myfile, line)) {
            std::istringstream buffer(line);
            std::vector<double> line_double((std::istream_iterator<double>(buffer)),
                                     std::istream_iterator<double>());

            v.push_back(line_double);
        }
        myfile.close();
    }
    else std::cout << "ERROR: file not found" << std::endl;

}

void save_value(std::string path, std::string reader, long double value)
{
    std::ofstream myfile (path, std::ios::app);
    if (myfile.is_open())
    {
        if(reader.size() != 0) myfile << reader << " ";
        myfile << value << std::endl;
        myfile.close();
    }
    else std::cout << "ERROR: file not found" << std::endl;

}

