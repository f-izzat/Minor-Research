#ifndef _OPTIMIZER_H
#define _OPTIMIZER_H

#include "common.h"
#include "lbfgsb.h"

#define BFGSPRINT ((long int)-1)
#define BFGSFACTR (10000000)
#define BFGSPGTOL (0.00001)
#define BFGSNMAX ((long int)1024)
#define BFGSMMAX ((long int)17)
#define BFGSM ((long int)10)


namespace BSVR::optimizer {

	struct Problem {
		Problem(const unsigned int& dim) : dim(dim) {}
		virtual double objective_fxn(){}
		virtual void gradient(){}
		virtual void operator()(const TVector& theta, TVector& grad) = 0;

		unsigned int dim;
		double fopt = 0.0;
	};

	struct LBFGSB {

		integer iprint = (long int)-1;
		doublereal factr = 1E7;
		doublereal pgtol = 1E-5;
		integer m = (long int)10;

		LBFGSB() {}


		void solve(TVector& theta, Problem& problem) {
			/* System generated locals */
			integer i__1;    
			/* Local variables */
			char task[60];
			char csave[60];
			integer isave[44];
			logical lsave[4];
			integer iwa[BFGSNMAX * 3];
			doublereal g[BFGSNMAX];
			doublereal l[BFGSNMAX];
			doublereal wa[2 * BFGSMMAX * BFGSNMAX + 4 * BFGSNMAX + 12 * BFGSMMAX * BFGSMMAX + 12 * BFGSMMAX];
			doublereal dsave[29];

			doublereal f;
			integer i;
			
			/* Subroutine */
			extern int setulb_(integer*, integer*, doublereal*, doublereal*, doublereal*, 
							   integer*, doublereal*, doublereal*, doublereal*,
							   doublereal*, doublereal*, integer*, char*, integer*, 
							   char*, logical*, integer*, doublereal*, ftnlen,
							   ftnlen);


			integer n = problem.dim;			
			assert(n <= BFGSNMAX && m <= BFGSMMAX);
			i__1 = n;
			integer nbd[BFGSNMAX];
			for (i = 0; i < n; i++) {
				nbd[i] = 0;
			}
			/* We now define the starting point. */
			doublereal u[BFGSNMAX], x[BFGSNMAX];
			i__1 = n;
			for (i = 0; i < i__1; ++i) {
				x[i] = theta[i];
			}
			TVector grad = TVector::Zero(n);
			/*  ------- the beginning of the loop ---------- */
		L111:
			/* This is the call to the L-BFGS-B code. */
			lbfgsb(&n, &m, x, l, u, 
				   nbd, &f, g, &factr, &pgtol, 
				   wa, iwa, task, &iprint, csave, 
				   lsave, isave, dsave, 60L, 60L);
			i__1 = n;
			for (i = 0; i < i__1; i++)
			{
				theta[i] = x[i];
			}
			// Evaluate Objective Function and Gradient
			problem(theta, grad);
			i__1 = n;
			for (i = 0; i < i__1; i++) {
				g[i] = grad[i];
				x[i] = theta[i];
			}
			goto L111;






		}


	};







}



#endif