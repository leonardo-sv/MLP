#ifndef FILES_H
#define FILES_H

#include <iostream>
#include <vector>

void file_to_vector(std::string path, std::vector<std::vector<double>> &v);

void save_value(std::string path, std::string c, double value);


#endif // FILES_H
