#include "tools.h"

pair<pair<MatrixXd,VectorXd>,pair<int,int>> load_data(string fname){
    int r=0,c=0;
    ifstream f;
    string line;
	f.open(fname);
    while(getline(f,line)){
        if(r == 1){
            for(int i=0;i<line.size();i++){
                if(line[i]==':'){
                    c++;
                }
            }
        }
        r++;
    }
    f.close();
    f.open(fname);
	MatrixXd X(r, c+1);
	VectorXd y(r);
    int cr=0;
    vector<char *> substrs;
    while(getline(f,line)){
        substrs.clear();
        char * line_c = (char*)line.data();
        char * tk = strtok(line_c," ");
        y(cr)=atof(tk);
        for(int k=0;k<c;k++){
            tk = strtok(NULL," ");
            substrs.push_back(tk); 
        }
        for(int k=0;k<c;k++){
            char* split = strtok(substrs[k],":");
            split = strtok(NULL,":");
            X(cr,k)=atof(split);
        }
        X(cr,c)=1;
        cr++;
    }
    f.close();
    return make_pair(make_pair(X,y),make_pair(r,c));
}



float ridge_loss(MatrixXd X,MatrixXd y,VectorXd theta,float lambda){
    float loss = 0;
    int row = y.size();
    int col = theta.size();
    for(int i=0;i<row;i++){
        float Xtheta_j = 0;
        for(int j=0;j<col;j++){
            Xtheta_j += X(i,j)*theta(j);
        }
        loss += (Xtheta_j-y(i))*(Xtheta_j-y(i));
    }
    loss = loss+ lambda * float(theta.transpose()*theta);
    loss = loss/(2*row);
    return loss;
}

float MSELoss(MatrixXd X,VectorXd y,VectorXd theta){
    float loss = 0;
    int row = y.size();
    int col = theta.size();
    for(int i=0;i<row;i++){
        float Xtheta_j = 0;
        for(int j=0;j<col;j++){
            Xtheta_j += X(i,j)*theta(j);
        }
        loss += (Xtheta_j-y(i))*(Xtheta_j-y(i));
    }
    loss = loss/(row);
    return loss;
}
float get_alpha(MatrixXd X,MatrixXd y,VectorXd theta,VectorXd d, float lambda){
	double alpha = 1.0;
	double beta = 1.0;
    int max = 100;
    int count = 0;
	if(cond1(X, y, theta, d, alpha) && cond2(X, y, theta, d, alpha))
    {
        return alpha;
    }else{
		while(!cond1(X, y, theta, d, alpha)) {
            alpha = alpha * 0.55;
            count++;
            if(count>=max){
                return alpha;
            }
        }
		while(!cond2(X, y, theta, d, alpha)) {
			beta = alpha / 0.55;
			double d_alpha = beta - alpha;
			while(!cond1(X, y, theta, d, alpha + d_alpha)) {
                d_alpha = d_alpha * 0.55;
                count++;
                if(count>=max){
                    return alpha;
                }
            }
			alpha += d_alpha;
            count++;
            if(count>=max){
                return alpha;
            }
		}
	}
	return alpha;
}


int cond1(MatrixXd X,MatrixXd y,VectorXd theta,VectorXd d,float alpha,float rho,float lambda){
    VectorXd grad = 1.0/(y.size())*(X.transpose()*(X*theta-y)+lambda*theta);
    float left = ridge_loss(X,y,theta+alpha*d);
    float right = ridge_loss(X,y,theta)+rho*alpha*(grad.transpose()*d).value();
    if(left<=right){return 1;}
    else{return 0;}
}
int cond2(MatrixXd X,MatrixXd y,VectorXd theta,VectorXd d,float alpha,float rho,float sigma,float lambda){
    VectorXd grad = 1.0/(theta.size())*(X.transpose()*(X*theta-y)+lambda*theta);
    VectorXd theta_ = theta+alpha*d;
    VectorXd grad_ = 1.0/(theta.size())*(X.transpose()*(X*theta_-y)+lambda*theta_);
    float left = float(grad_.transpose()*d);
    float right = sigma*float(grad.transpose()*d);
    if(left>=right){return 1;}
    else{return 0;}
}


void write(string filename,vector<float> loss){
    ofstream f;
    f.open(filename);
    for(int i=0;i<loss.size();i++){
        f<<loss[i]<<endl;
    }
    f.close();
}