#include "Algorithm.h"


VectorXd GradientDescent(MatrixXd X,VectorXd y,VectorXd theta,float lambda,int iters){
    vector<float> loss;
    string fname = "../output/GD.txt";
    int m = y.size();
    for(int i=0;i<iters;i++){
        VectorXd grad_theta = (1.0/m)*(X.transpose()*(X*theta-y)+lambda*theta);
        // cout<<"grad is:"<<grad_theta<<endl;
        // float alpha = get_alpha(X, y, theta, -grad_theta);
        float alpha = 0.000001;
        // cout<<grad_theta<<endl;
        theta = theta - alpha*grad_theta;
        float closs = MSELoss(X,y,theta);
        loss.push_back(closs);
        // cout<<"iter:"<<i<<",loss:"<<closs<<endl;
    }
    write(fname,loss);
    return theta;
}




VectorXd ConjugateDescent(MatrixXd X,VectorXd y,VectorXd theta,float lambda,int iters){
    float eps = 1e-6;
    vector<float> loss;
    string fname = "../output/CG.txt";
    int d = theta.size();
    MatrixXd I = MatrixXd::Identity(d,d);
    MatrixXd A = X.transpose()*X+lambda*I;
    MatrixXd b = X.transpose()*y;
    VectorXd r = b-(A*theta);
    VectorXd p = r;
    float alpha;
    float beta;
    float rTr = r.transpose()*r;
    for(int i=0;i<iters;i++){
        if(rTr<eps){
            float closs = MSELoss(X,y,theta);
            loss.push_back(closs);
            continue;
        }
        alpha = rTr/(p.transpose()*A*p);//cal alpha
        theta = theta+alpha*p;//set theta
        r = r-alpha*A*p;//set r
        beta = rTr;//r_i^T r_i
        rTr = r.transpose()*r;//set rTr = r_{i+1}^T r_{i+1}
        beta = rTr/beta;//set beta
        p = beta*p+r;
        float closs = MSELoss(X,y,theta);
        loss.push_back(closs);
        // cout<<"iter:"<<i<<",loss:"<<closs<<endl;
    }
    write(fname,loss);
    return theta;
}
VectorXd quasiNewton(MatrixXd X,VectorXd y,VectorXd theta,float lambda,int iters){
    float eps = 1e-6;
    vector<float> loss;
    string fname = "../output/quasiNT.txt";
    int m = theta.size();
    MatrixXd H = MatrixXd::Identity(m,m);
    for(int i=0;i<iters;i++){
        VectorXd grad_theta = (1.0/m)*(X.transpose()*(X*theta-y)+lambda*theta);
        if(grad_theta.transpose()*grad_theta<eps){
            float closs = MSELoss(X,y,theta);
            loss.push_back(closs);
            continue;
        }
        VectorXd d = -H*grad_theta;
        float alpha = get_alpha(X,y,theta,d);
        // cout<<alpha<<endl;
        // float alpha = 0.000001;
        VectorXd theta_ = theta+alpha*d;
        VectorXd grad_theta_ = (1.0/m)*(X.transpose()*(X*theta_-y)+lambda*theta_);
        VectorXd d_theta = theta_-theta;
        VectorXd d_g = grad_theta_-grad_theta;
        H = H + (d_theta-H*d_g)*d_theta.transpose()*H*1.0/(d_theta.transpose()*H*d_g);
        theta = theta_;
        float closs = MSELoss(X,y,theta);
        loss.push_back(closs);
        // cout<<"iter:"<<i<<",loss:"<<closs<<endl;
    }
    write(fname,loss);
    return theta;
}
VectorXd Nesterov(MatrixXd X,VectorXd y,VectorXd theta,float r,float lambda,int iters){
    float eps = 1e-6;
    vector<float> loss;
    string fname = "../output/Nesterov.txt";
    int m = theta.size();
    VectorXd x_0,y_0;
    float step=0.000001;
    x_0=theta;
    y_0=theta;
    VectorXd xk,xk1,yk,yk1;
    yk1=y_0;
    xk1=x_0;
    for(int i=1;i<=iters;i++){
        VectorXd grad_y = (1.0/m)*(X.transpose()*(X*yk1-y)+lambda*yk1);
        float alpha = get_alpha(X,y,yk1,-grad_y);
        // cout<<grad_y<<endl;
        xk=yk1-step*grad_y;
        yk=xk+(i-1)/(i+2)*(xk-xk1);

        yk1=yk;
        xk1=xk;
        float closs = MSELoss(X,y,yk);
        // cout<<yk.transpose()<<endl;
        loss.push_back(closs);
    }
    write(fname,loss);
    VectorXd ret_theta = yk;
    return ret_theta;
}
VectorXd rNesterov(MatrixXd X,VectorXd y,VectorXd theta,float r,float lambda,int iters){
    float eps = 1e-6;
    vector<float> loss;
    string fname = "../output/rNesterov.txt";
    int m = theta.size();
    VectorXd x_0,y_0;
    float step=0.000001;
    x_0=theta;
    y_0=theta;
    VectorXd xk,xk1,yk,yk1;
    yk1=y_0;
    xk1=x_0;
    for(int i=1;i<=iters;i++){
        VectorXd grad_y = (1.0/m)*(X.transpose()*(X*yk1-y)+lambda*yk1);
        float alpha = get_alpha(X,y,yk1,-grad_y);
        // cout<<grad_y<<endl;
        xk=yk1-step*grad_y;
        yk=xk+(i-1)/(i+r-1)*(xk-xk1);

        yk1=yk;
        xk1=xk;
        float closs = MSELoss(X,y,yk);
        // cout<<yk.transpose()<<endl;
        loss.push_back(closs);
    }
    write(fname,loss);
    VectorXd ret_theta = yk;
    return ret_theta;
}

VectorXd srNesterov(MatrixXd X,VectorXd y,VectorXd theta,float r,float lambda,int iters){
    float eps = 1e-6;
    vector<float> loss;
    string fname = "../output/srNesterov.txt";
    int m = theta.size();
    VectorXd x_0,y_0;
    float step=0.000001;
    float kmin=5;
    x_0=theta;
    y_0=theta;
    VectorXd xk,xk1,xk2,yk,yk1;
    yk1=y_0;
    xk1=x_0;
    xk2=x_0;
    float j = 1;
    for(int i=1;i<=iters;i++){
        VectorXd grad_y = (1.0/m)*(X.transpose()*(X*yk1-y)+lambda*yk1);
        float alpha = get_alpha(X,y,yk1,-grad_y);
        // cout<<grad_y<<endl;
        xk=yk1-step*grad_y;
        yk=xk+(j-1)/(j+2)*(xk-xk1);
        if((xk-xk1).norm()<(xk1-xk2).norm() && j>=kmin){
            j=1;
        }else{
            j=j+1;
        }
        yk1=yk;
        xk2=xk1;
        xk1=xk;
        float closs = MSELoss(X,y,yk);
        // cout<<yk.transpose()<<endl;
        loss.push_back(closs);
    }
    write(fname,loss);
    VectorXd ret_theta = yk;
    return ret_theta;
}
VectorXd ftrsNesterov(MatrixXd X,VectorXd y,VectorXd theta,float time,float lambda,int iters){
    float eps = 1e-6;
    vector<float> loss;
    string fname = "../output/ftrsNesterov.txt";
    int m = theta.size();
    VectorXd x_0,y_0;
    float step=0.000001;
    float kmin=5;
    x_0=theta;
    y_0=theta;
    VectorXd xk,xk1,xk2,yk,yk1;
    yk1=y_0;
    xk1=x_0;
    xk2=x_0;
    float j = 1;
    for(int i=1;i<=iters;i++){
        VectorXd grad_y = (1.0/m)*(X.transpose()*(X*yk1-y)+lambda*yk1);
        float alpha = get_alpha(X,y,yk1,-grad_y);
        // cout<<grad_y<<endl;
        xk=yk1-step*grad_y;
        yk=xk+(j-1)/(j+2)*(xk-xk1);
        if(j==time){
            j=1;
        }else{
            j=j+1;
        }
        yk1=yk;
        xk2=xk1;
        xk1=xk;
        float closs = MSELoss(X,y,yk);
        // cout<<yk.transpose()<<endl;
        loss.push_back(closs);
    }
    write(fname,loss);
    VectorXd ret_theta = yk;
    return ret_theta;
}

VectorXd fsNesterov(MatrixXd X,VectorXd y,VectorXd theta,float r,float lambda,int iters){
    float eps = 1e-6;
    vector<float> loss;
    string fname = "../output/fsNesterov.txt";
    int m = theta.size();
    VectorXd x_0,y_0;
    float step=0.000001;
    float kmin=5;
    x_0=theta;
    y_0=theta;
    VectorXd xk,xk1,xk2,yk,yk1;
    yk1=y_0;
    xk1=x_0;
    xk2=x_0;
    float j = 1;
    for(int i=1;i<=iters;i++){
        VectorXd grad_y = (1.0/m)*(X.transpose()*(X*yk1-y)+lambda*yk1);
        float alpha = get_alpha(X,y,yk1,-grad_y);
        // cout<<grad_y<<endl;
        xk=yk1-step*grad_y;
        yk=xk+(j-1)/(j+2)*(xk-xk1);
        if(grad_y.transpose()*(xk-xk1)>0){
            j=1;
        }else{
            j=j+1;
        }
        yk1=yk;
        xk2=xk1;
        xk1=xk;
        float closs = MSELoss(X,y,yk);
        // cout<<yk.transpose()<<endl;
        loss.push_back(closs);
    }
    write(fname,loss);
    VectorXd ret_theta = yk;
    return ret_theta;
}
