#ifndef _SMO_H
#define _SMO_H

#include "common.h"

namespace BSVR::smo {
	enum class SetName { I0, I1, I2, I3, IA, IB };
	enum class DataSize { SPARSE, NORMAL, LARGE, XLARGE, XXLARGE };
	// Skip Method_Name
	enum class TKernel { Gaussian, Polynomial, Exponential };
	// Skip Training_Method
	// Skip Optimization_Method

	/* Data Node */
	/* Data List */

	typedef struct _CacheNode {
		struct _Alphas* alpha;
		struct _CacheNode* previous;
		struct _CacheNode* next;
	} CacheNode;
	typedef struct _CacheList {
		unsigned int count;
		CacheNode* front;
		CacheNode* rear;
	} CacheList;
	typedef struct _Alphas {
		// ai for input point
		double alpha_up = 0.0;
		// ai' for input point
		double alpha_dw = 0.0;
		// save Fi here if the pair is in Set Io
		double f_cache = 0.0;
		// diagonal entry
		double kernel = 0.0;
		// point to the Node in Cache List
		CacheNode* cache;
		//CacheNode* cache;
		SetName setname;
		// inputs[index]
		TVector pair;
		double output;
		unsigned int index;
	} Alphas;

	//struct Cache;
	//struct Alpha_ {
	//	/* Matrix
	//	*	0 - [alpha_dw ; alpha_up]
	//	*	1 - [alpha_dw ; alpha_up]
	//	*				  .
	//	*/

	//	TMatrix matrix;
	//	TVector f_cache;
	//	TVector kernel_diag;
	//	std::vector<SetName> setnames;
	//};
	//struct AlphaCache {
	//	TVector alpha;
	//	SetName setname;
	//	double	f_cache;
	//	double	kernel_diag;
	//	unsigned int index = -1;
	//	AlphaCache(const Alpha& alphas, const unsigned int& index) :
	//		index(index), alpha(alphas.matrix.row(index)),  setname(alphas.setnames[index]),
	//		f_cache(alphas.f_cache[index]), kernel_diag(alphas.kernel_diag[index])
	//	{}
	//};
	//struct Cache {
	//	std::vector<AlphaCache> alphas;
	//};


	struct Settings {
		/* SMO Settings */

		// Soft Insensitive Loss Function
		double beta = 0.3;
		bool fix_beta = true;
		// Regularization Parameter	
		double inf_vc = 0.01; double vc = 1.0; double sup_vc = 500.0;
		// Epsilon insensitive Loss Function	               
		double inf_epsilon = -5.0; double epsilon = 0.05; double sup_epsilon = -0.7;
		// 	
		double inf_kaio = -10.0; double kaio = 1.0; double sup_kaio = 5.0;
		//	
		double inf_kappao = -13.0; double kappao = 10.0; double sup_kappao = 10.0;
		// Tolerance Parameter in Loose KKT 
		double tol = 0.001;
		// Error Precision Setting
		double eps = 0.000001;

		TKernel kernel = TKernel::Gaussian;
		bool ARD = true;
		// Polynomial Power
		unsigned int p = 2;
		// Sigma square is Gaussian kernel	                  
		double inf_kai = 0.1; double kai = 0.5; double sup_kai = 5.0;
		// Inhomogegeous or not
		double kaip = 0.0;
		// Bias	      	
		double b_low; double bias; double b_up;
		long unsigned int i_up;
		long unsigned int i_low;
		// Lengthscale
		TVector lengthscale;
		double suplengthscale;
		double inflengthscale;


		/*---------------*/
		/* Remove? */
		//double* ard;                  // ARD hyperparameters for Kernel function
		//double pre_vc;                 // previous Regularization Parameter
		//double pre_epsilon;            // previous Epsilon insensitive Loss Function
		//double pre_beta;               // previous Soft Insensitive Loss Function		
		//double pre_kaio;
		//double pre_kappao;
		//double pre_kai;
		//double* pre_ard;              // previous ARD for Kernel function
		/*---------------*/
		//char* inputfile;              // the name of input data file
		//struct _Data_List pairs;       // data_list saving training data
		//char* testfile;               // the name of input data file 
		//struct _Data_List testdata;
		//BOOL normalized_input;         // normalize the input of training data if TRUE
		//BOOL normalized_output;        // normalize the output of training data if TRUE
		//unsigned int index;            // expert index	
		//unsigned int committee;        // the number of expert members 
		//Training_Method trainmethod;   // Training Methods 
		//Optimization_Method optimethod;// Optimization methods 
		// 
		//struct _Data_List * pairs ;
		//Method_Name method;            // Heuristic Methods 
		//BOOL smo_dumping;              // dumping detailed log file if TRUE
		//BOOL smo_quickmode;            // quick mode for sequential training 
		//BOOL smo_display;              // display message on screen if TRUE
		//BOOL smo_randominit;           // initialize parameters randomly if TRUE
		//BOOL smo_working;              // flag of active
		//BOOL smo_bayes;
		//double smo_timing;             // CPU time consumed by the routine
		//char* inputfile;              // the name of input data file 
		//double regular;
		//BOOL cacheall;					// less than 10000
		//unsigned int index;            // expert index in expert committee
		//BOOL abort;                    // flag of exit
		/*---------------*/

		/* LBFGSB Settings */
		int EvalLimit = 0;
		// total number of adjustable parameters
		unsigned int number;
		// number of adjustable parameters besides ARD (Regression = 3)
		unsigned int adjnum = 3;
		//double supVc;               // superior of C
		//double infVc;               // inferior of C 
		//double supEps;              // superior of Epsilon
		//double infEps;              // inferior of Epsilon
		//double supKaio;               // superior of C
		//double infKaio;               // inferior of C 
		//double supKappao;               // superior of Kappa_b
		//double infKappao;               // inferior of Kappa_b
		//BOOL smo_quickmode;         // quick mode for sequential training 
		//BOOL smo_fixbeta;           // beta is fixed in training 




	};

}




#endif