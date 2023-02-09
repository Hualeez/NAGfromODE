#include "tools.h"
VectorXd GradientDescent(MatrixXd X,VectorXd y,VectorXd theta,float lambda=0.01,int iters = 100);
VectorXd ConjugateDescent(MatrixXd X,VectorXd y,VectorXd theta,float lambda=0.01,int iters=100);
VectorXd quasiNewton(MatrixXd X,VectorXd y,VectorXd theta,float lambda=0.01,int iters=100);
VectorXd Nesterov(MatrixXd X,VectorXd y,VectorXd theta,float r=3,float lambda=0.01,int iters=100);
VectorXd rNesterov(MatrixXd X,VectorXd y,VectorXd theta,float r=3,float lambda=0.01,int iters=100);
VectorXd srNesterov(MatrixXd X,VectorXd y,VectorXd theta,float r=3,float lambda=0.01,int iters=100);
VectorXd ftrsNesterov(MatrixXd X,VectorXd y,VectorXd theta,float time=5,float lambda=0.01,int iters=100);
VectorXd fsNesterov(MatrixXd X,VectorXd y,VectorXd theta,float r=3,float lambda=0.01,int iters=100);