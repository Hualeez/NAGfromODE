#include "tools.h"
#include "Algorithm.h"


int main(){
    pair<pair<MatrixXd,VectorXd>,pair<int,int>> ret = load_data("..\\dataset\\bodyfat.txt");
    VectorXd theta = VectorXd::Zero(ret.second.second+1);
    MatrixXd X = ret.first.first;
    VectorXd y = ret.first.second;
    // cout<<X<<endl;
    // cout<<y<<endl;
    theta = GradientDescent(X,y,theta,0.01,100);
    cout<<"GD theta = "<<theta.transpose()<<endl;
    // theta = VectorXd::Zero(ret.second.second+1);
    // theta = ConjugateDescent(X,y,theta,0.01,100);
    // cout<<"CD theta = "<<theta.transpose()<<endl;
    // theta = VectorXd::Zero(ret.second.second+1);
    // theta = quasiNewton(X,y,theta,0.01,100);
    // cout<<"quasiNT theta = "<<theta.transpose()<<endl;
    theta = VectorXd::Zero(ret.second.second+1);
    theta = Nesterov(X,y,theta,0,0.01,100);
    cout<<"Nesterov theta = "<<theta.transpose()<<endl;
    theta = VectorXd::Zero(ret.second.second+1);
    theta = rNesterov(X,y,theta,5,0.01,100);
    cout<<"r = "<<5<<" Nesterov theta = "<<theta.transpose()<<endl;
    theta = VectorXd::Zero(ret.second.second+1);
    theta = srNesterov(X,y,theta,5,0.01,100);
    cout<<"srNesterov theta = "<<theta.transpose()<<endl;
    theta = VectorXd::Zero(ret.second.second+1);
    theta = ftrsNesterov(X,y,theta,5,0.01,100);
    cout<<"ftrsNesterov theta = "<<theta.transpose()<<endl;
    theta = VectorXd::Zero(ret.second.second+1);
    theta = fsNesterov(X,y,theta,3,0.01,100);
    cout<<"fsNesterov theta = "<<theta.transpose()<<endl;
}