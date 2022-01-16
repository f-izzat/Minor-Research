#ifndef _COMMON_H
#define _COMMON_H
#include <iostream>
#include <Eigen/Dense>
#define PI (3.141592654)
//#include "bfgs.h"
//#include "expert.h"
//#include "wrapper.h"

using std::shared_ptr;
using std::unique_ptr;
using std::make_shared;
using std::make_unique;

typedef Eigen::VectorXd TVector;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> TMatrix;

/* Utilities */
struct StandardScaler {
	StandardScaler() = default;
	StandardScaler(const TMatrix& X) { fit(X); }
	void fit(const TMatrix& X) {
		nrows = X.rows(); ncols = X.cols();
		mean = X.colwise().mean();
		TMatrix Xmean = X.rowwise() - X.colwise().mean();
		std_dev = (((Xmean).array().square().colwise().sum()) / ((Xmean).rows())).sqrt();
		fit_ = true;
	}
	void scale_ip(TMatrix& X) {
		if (!fit_) throw std::runtime_error("scaler.fit() not called");
		TMatrix Xmean = X.array().rowwise() - mean.array().transpose();
		X = (Xmean).array().rowwise() / std_dev.array().transpose();
	}
	TMatrix scale(const TMatrix& X) {
		if (!fit_) throw std::runtime_error("scaler.fit() not called");
		TMatrix Xmean = X.array().rowwise() - mean.array().transpose();
		return (Xmean).array().rowwise() / std_dev.array().transpose();
	}

	TMatrix rescale(const TMatrix& X) {
		TMatrix Z(nrows, ncols);
		Z = X.array().rowwise() * std_dev.transpose().array();
		Z.array().rowwise() += mean.transpose().array();
		return Z;
	}
	Eigen::Index nrows = 0;
	Eigen::Index ncols = 0;
	TVector mean;
	TVector std_dev;
private:
	bool fit_ = false;
};



#endif // !_COMMON_H





