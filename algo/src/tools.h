#ifndef _TOOLS_
#define _TOOLS_
#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
using namespace Eigen;
using namespace std;
pair<pair<MatrixXd,VectorXd>,pair<int,int>> load_data(string fname);
float ridge_loss(MatrixXd X,MatrixXd y,VectorXd theta,float lambda = 0.1);
float get_alpha(MatrixXd X,MatrixXd y,VectorXd theta,VectorXd d, float lambda = 0.1);
int cond1(MatrixXd X,MatrixXd y,VectorXd theta,VectorXd d,float alpha,float rho = 0.6,float lambda = 0.1);
int cond2(MatrixXd X,MatrixXd y,VectorXd theta,VectorXd d,float alpha,float rho = 0.6,float sigma = 0.7,float lambda = 0.1);
float MSELoss(MatrixXd X,VectorXd y,VectorXd theta);
void write(string filename,vector<float> loss);
#endif