#ifndef _MODEL_H
#define _MODEL_H

#include "smo.h"
#include "optimizer.h"

#define ALPHADW(_idx)	(alpha[_idx, 0])
#define ALPHAUP(_idx)	(alpha[_idx, 1])
#define ALPHAKD(_idx)	(alpha[_idx, 2])
#define ALPHAF(_idx)	(alpha[_idx, 3])
#define ALPHASET(_idx)	(set_names[_idx])

#define CACHEI(_idx)	(static_cast<unsigned int>(cache[_idx](0)))
#define CACHEF(_idx)	(cache[_idx](1))

namespace BSVR::model {
	using namespace BSVR::smo;
	using BSVR::optimizer::Problem;
	using BSVR::optimizer::LBFGSB;


	class Model {
	public:
		unsigned int   nrows;
		unsigned int   ncols;
		unsigned int   i_ymax;
		unsigned int   i_ymin;
		/* Alphas matrix */
		// [alphadw; alphaup; kernel_diag; f_cache]
		TMatrix alpha;
		/* Set Names */
		std::vector<SetName> set_names;
		/* IO Cache */
		std::vector<TVector> cache;
		unsigned int cache_count = 0;
		// Data Scaler
		StandardScaler scaler;
		// Optimum Parameters
		TVector theta;
		TMatrix inputs;
		TMatrix outputs;
		Settings settings;
	private:


	public:
		Model(const TMatrix& inputs, const TMatrix& outputs)
			: inputs(inputs), outputs(outputs), scaler(inputs), nrows(inputs.rows()), ncols(inputs.cols())
		{
			// Output Min/Max Indices
			TMatrix::Index maxIndex, minIndex;
			double ymax = outputs.col(0).maxCoeff(&maxIndex);
			double ymin = outputs.col(0).minCoeff(&minIndex);
			i_ymax = static_cast<unsigned int>(maxIndex);
			i_ymin = static_cast<unsigned int>(minIndex);
			//
			double Ymean = (outputs.rowwise() - outputs.colwise().mean()).coeff(0, 0);
			double Ystd = pow(Ymean, 2) / static_cast<double>(outputs.rows());
			settings.kaio = Ystd * Ystd;
			// Update_def_Settings
			if (inputs.rows() < 200) settings.beta = 0.5;
			else if (inputs.rows() < 2000) settings.beta = 0.3;
			else if (inputs.rows() < 4000) settings.beta = 0.1;
			else if (inputs.rows() < 7000) settings.beta = 0.05;
			else settings.beta = 0.01;

			//if (settings.ARD) settings.lengthscale = TVector::Constant(inputs.cols(), settings.kai);
			settings.lengthscale = TVector::Constant(inputs.cols(), settings.kai);

			/* Create Alphas */
			alpha = TMatrix::Zero(nrows, 4);
			set_names.resize(nrows);
			for (unsigned int i = 0; i < nrows; ++i) {
				alpha(i, 2) = kernel(i, i);
				set_names[i] = SetName::I1;
			}
			/* LBFGSB Settings */
			{
				settings.sup_epsilon = log(Ystd * Ystd);
				settings.suplengthscale = settings.sup_kai;
				settings.inflengthscale = settings.inf_kai;
				if (settings.ARD) settings.number = settings.adjnum + ncols;
				else settings.number = settings.adjnum + 1;
				// Fill initial theta
				theta = TVector::Zero(settings.number);
				theta[0] = settings.vc;
				theta[1] = log(settings.epsilon);
				theta[2] = log(settings.kappao);
				if (settings.ARD)
					for (unsigned int i = 0; i < ncols; ++i)
						theta[settings.adjnum + i] = log(settings.kai);
				else
					theta[settings.adjnum + 1] = log(settings.kai);
			}

		}
		double kernel(const unsigned int& i, const unsigned int& j) {
			unsigned int dd = 0;
			double kk = settings.kappao;

			switch (settings.kernel)
			{
			case TKernel::Polynomial: {
				for (dd = 0; dd < ncols; ++dd) {
					if (inputs(i, dd) != 0 && inputs(j, dd) != 0)
						kk += settings.lengthscale.coeff(dd) * inputs(i, dd) * inputs(j, dd);
				}
				if (settings.p > 1) kk = settings.kaio * pow((kk + settings.kaip), (double)settings.p);
				else kk = settings.kaio * (kk + settings.kaip);
			}
			case TKernel::Exponential: {
				for (dd = 0; dd < ncols; ++dd) {
					if (inputs(i, dd) != 0 && inputs(j, dd) != 0)
						kk += settings.lengthscale.coeff(dd) * fabs((inputs(i, dd) - inputs(j, dd)));
				}
				kk = settings.kaio * exp(-kk);
			}
			default: {
				for (dd = 0; dd < ncols; ++dd) {
					if (inputs(i, dd) != 0 && inputs(j, dd) != 0)
						kk += settings.lengthscale.coeff(dd) * (inputs(i, dd) - inputs(j, dd)) * (inputs(i, dd) - inputs(j, dd));
				}
				kk = settings.kaio * exp(-kk / 2.0);
			}
			}
			// if settings.working
			if (i == j) kk += 2.0 * settings.beta * settings.epsilon / settings.vc;
			return kk;
		}
		virtual void train() {};
		virtual void set_params(const TVector& t) {}
		virtual bool take_step(const unsigned int& widx1, const unsigned int& widx2) { return false; }
		virtual unsigned int examine_example(const unsigned int& widx) { return 0; }
		virtual double calculate_fi(const unsigned int& widx) { return 0.0; }
		virtual double objective_value() { return 0.0; }
		SetName get_set(double& a, double& b) {
			double a_ = a;
			double b_ = b;

			if ((a_ * b_) != 0)
			{
				printf("\r\nFatal Error: alpha or VC in takeStep %f %f \r\n", a, b);
				a = 0; b = 0;
				return SetName::I1;
			}
			if (a_ > settings.vc)
			{
				printf("\r\nFatal Error: alpha or VC in takeStep %f %f \r\n", a, b);
				a = settings.vc;
				a_ = settings.vc;
				return SetName::I3;
			}
			if (b_ > settings.vc)
			{
				printf("\r\nFatal Error: alpha or VC in takeStep %f %f \r\n", a, b);
				b = settings.vc;
				b_ = settings.vc;
				return SetName::I2;
			}
			if (settings.vc == a_ && 0 == b_)
				return SetName::I3;
			else if (settings.vc == b_ && 0 == a_)
				return SetName::I2;
			else if (a_ > 0 && a_ < settings.vc && 0 == b_)
				return SetName::IA;
			else if (b_ > 0 && b_ < settings.vc && 0 == a_)
				return SetName::IB;
			else if (0 == a_ && 0 == b_)
				return SetName::I1;
			else
			{
				printf("\r\nFATAL ERROR : wrong alpha or VC in GetName. %f %f \r\n", a, b);
				a = 0; b = 0;
				return SetName::I1;
			}
		}
		void add_to_cache(const unsigned int& widx) {
			TVector tmp = TVector::Zero(2);
			tmp[0] = (double)widx;
			tmp[1] = ALPHAF(widx);
			cache.push_back(tmp);
			cache_count++;
		}
	};

	struct Objective : public Problem {
		Objective(Model* model) : Problem(model->settings.number), model(model) {}

		double objective_fxn(const TVector& theta) {
			model->set_params(theta);
			return model->objective_value();
		}

		void operator()(const TVector& theta) override {
			model->set_params(theta);
			model->objective_value();
		}

		Model* model;
	};

	class BayesianSVR : public Model {
	public:
		BayesianSVR(const TMatrix& inputs, const TMatrix& outputs) : Model(inputs, outputs) {}

		double calculate_fi(const unsigned int& widx) override {
			/* setandfi.cpp */
			double Fi = 0;
			for (unsigned j = 0; j < nrows; ++j) {
				if (ALPHAUP(j) != 0 || ALPHADW(j) != 0)
					Fi += (ALPHAUP(j) - ALPHADW(j)) * kernel(j, widx);
			}
			Fi = outputs(widx) - Fi;
			return Fi;
		}
		bool take_step(const unsigned int& widx1, const unsigned int& widx2) override {
			double a1 = 0.0, a1a = 0.0, a2 = 0.0, a2a = 0.0;	//old alpha
			double n1 = 0.0, n1a = 0.0, n2 = 0.0, n2a = 0.0;	//new alpha

			SetName name1, name2;
			unsigned int i1 = widx1; // alpha1
			unsigned int i2 = widx2; // alpha2
			if (i1 == i2) return false;
			/* ----------- */
			a1 = n1 = ALPHAUP(i1);
			a1a = n1a = ALPHADW(i1);
			a2 = n2 = ALPHAUP(i2);
			a2a = n2a = ALPHADW(i2);
			double F1 = ALPHAF(i1);
			double F2 = ALPHAF(i2);
			double K11 = ALPHAKD(i1);
			double K22 = ALPHAKD(i2);
			double K12 = kernel(i1, i2);
			double detH = K11 * K22 - K12 * K12;
			if (!((ALPHASET(i1) == SetName::IA || ALPHASET(i1) == SetName::IB) && (ALPHASET(i2) == SetName::IA || ALPHASET(i2) == SetName::IB))) {
				detH = (K11 * K22 - K12 * K12) / settings.tol;
			}
			/* ----------- */
			double G1, G2, G1a, G2a;
			bool case1 = false, case2 = false, case3 = false, case4 = false;
			bool finish = false, update = false;
			if (detH <= 0.0) return false;
			else {
				// consider four quadrants together
				while (!finish)
				{
					G1 = -F1 + (1 - settings.beta) * settings.epsilon;
					G2 = -F2 + (1 - settings.beta) * settings.epsilon;
					G1a = F1 + (1 - settings.beta) * settings.epsilon;
					G2a = F2 + (1 - settings.beta) * settings.epsilon;

					//this loop is passed at most three times.
					if (!case1 && (a1 > 0.0 || (a1a == 0.0 && (-K22 * G1 + K12 * G2) > 0.0)) && (a2 > 0.0 || (a2a == 0.0 && (K12 * G1 - K11 * G2) > 0.0)))
					{
						case1 = true; // (a1, a2)
						n1 = a1 + (-K22 * G1 + K12 * G2) / detH;
						n2 = a2 + (K12 * G1 - K11 * G2) / detH;
						//check constraints
						if (n1 < 0.0) n1 = 0.0;
						else if (n1 > settings.vc) n1 = settings.vc;
						if (n2 < 0.0) n2 = 0.0;
						else if (n2 > settings.vc) n2 = settings.vc;
						n1a = 0.0;
						n2a = 0.0;
						//update if significant 
						if (fabs(n1 - a1) + fabs(n2 - a2) > 0.0)
						{
							update = true;
						}
					}
					else if (!case2 && (a1 > 0.0 || (a1a == 0.0 && (-K22 * G1 - K12 * G2a) > 0.0)) && (a2a > 0.0 || (a2 == 0.0 && (-K12 * G1 - K11 * G2a) > 0.0)))
					{
						case2 = true; // (a1, a2a)	
						n1 = a1 + (-K22 * G1 - K12 * G2a) / detH;
						n2a = a2a + (-K12 * G1 - K11 * G2a) / detH;
						//check constraints
						if (n1 < 0.0) n1 = 0.0;
						else if (n1 > settings.vc) n1 = settings.vc;
						if (n2a < 0.0) n2a = 0.0;
						else if (n2a > settings.vc) n2a = settings.vc;
						n1a = 0.0;
						n2 = 0.0;
						//update if significant 
						if (fabs(n1 - a1) + fabs(n2a - a2a) > 0.0)
						{
							update = true;
						}
					}
					else if (!case3 && (a1a > 0.0 || (a1 == 0.0 && (-K22 * G1a - K12 * G2) > 0.0)) && (a2 > 0.0 || (a2a == 0.0 && (-K12 * G1a - K11 * G2) > 0.0))) {
						case3 = true; // (a1a, a2)
						n1a = a1a + (-K22 * G1a - K12 * G2) / detH;
						n2 = a2 + (-K12 * G1a - K11 * G2) / detH;
						//check constraints
						if (n1a < 0.0) n1a = 0.0;
						else if (n1a > settings.vc) n1a = settings.vc;
						if (n2 < 0.0) n2 = 0.0;
						else if (n2 > settings.vc) n2 = settings.vc;
						n1 = 0.0;
						n2a = 0.0;
						//update if significant 
						if (fabs(n1a - a1a) + fabs(n2 - a2) > 0.0) {
							update = true;
						}
					}
					else if (!case4 && (a1a > 0.0 || (a1 == 0.0 && (-K22 * G1a + K12 * G2a) > 0.0)) && (a2a > 0.0 || (a2 == 0.0 && (K12 * G1a - K11 * G2a) > 0.0))) {
						case4 = true; // (a1a, a2a)			
						n1a = a1a + (-K22 * G1a + K12 * G2a) / detH;
						n2a = a2a + (K12 * G1a - K11 * G2a) / detH;
						//check constraints
						if (n1a < 0.0) n1a = 0.0;
						else if (n1a > settings.vc) n1a = settings.vc;
						if (n2a < 0.0) n2a = 0.0;
						else if (n2a > settings.vc) n2a = settings.vc;
						n1 = 0.0;
						n2 = 0.0;
						//update if significant 
						if (fabs(n1a - a1a) + fabs(n2a - a2a) > 0.0)
						{
							update = true;
						}
					}
					else { finish = true; }
					// update Fi cache if necessary
					if (update)
					{
						//update F1 & F2
						F1 = F1 + (a1 - a1a - n1 + n1a) * K11 + (a2 - a2a - n2 + n2a) * K12;
						F2 = F2 + (a2 - a2a - n2 + n2a) * K22 + (a1 - a1a - n1 + n1a) * K12;
						a1 = n1; a2 = n2; a1a = n1a; a2a = n2a;
						update = false;
					}
				}
				// end of while
			}
			/* ----------- */
			// update Alpha List if necessary, then update Io_Cache, and vote settings.b_low & settings.b_up
			if (fabs((n2 - n2a) - (ALPHAUP(i2) - ALPHADW(i2))) > 0.0 || fabs((n1 - n1a) - (ALPHAUP(i1) - ALPHADW(i1))) > 0.0) {
				// store alphas in Alpha List
				a1a = ALPHADW(i1);  a1 = ALPHAUP(i1);
				a2a = ALPHADW(i2); a2 = ALPHAUP(i2);
				ALPHADW(i1) = n1a; ALPHAUP(i1) = n1;
				ALPHADW(i1) = n2a; ALPHAUP(i2) = n2;
				/* ----------- */
				// update Set & Cache_List  
				name1 = get_set(n1, n1a); name2 = get_set(n2, n2a);
				if (ALPHASET(i1) != name1)
				{
					if ((name1 == SetName::IA || name1 == SetName::IB) && (ALPHASET(i1) != SetName::IA && ALPHASET(i1) != SetName::IB)) {
						// Add_Cache_Node( &Io_CACHE, alpha1 ) ; // insert into Io
						add_to_cache(i1);
					}
					if ((ALPHASET(i1) == SetName::IA || ALPHASET(i1) == SetName::IB) && name1 != SetName::IA && name1 != SetName::IB) {
						// Del_Cache_Node( &Io_CACHE, alpha1 ) ;
						cache.erase(cache.begin() + i1);
					}
					ALPHASET(i1) = name1;
				}
				if (ALPHASET(i2) != name2)
				{
					if ((name2 == SetName::IA || name2 == SetName::IB) && (ALPHASET(i2) != SetName::IA && ALPHASET(i2) != SetName::IB)) {
						// Add_Cache_Node( &Io_CACHE, alpha1 ) ; // insert into Io 
						add_to_cache(i2);
					}
					if ((ALPHASET(i2) == SetName::IA || ALPHASET(i2) == SetName::IB) && name2 != SetName::IA && name2 != SetName::IB) {
						// Del_Cache_Node( &Io_CACHE, alpha2 ) ;
						cache.erase(cache.begin() + i2);
					}
					ALPHASET(i2) = name2;
				}
				/* ----------- */
				// update f-cache of i1 & i2 if not in Io_Cache
				if (ALPHASET(i1) != SetName::IA && ALPHASET(i1) != SetName::IB) {
					ALPHAF(i1) = ALPHAF(i1) - ((ALPHAUP(i1) - ALPHADW(i1)) - (a1 - a1a)) * K11 - ((ALPHAUP(i2) - ALPHADW(i2)) - (a2 - a2a)) * K12;
				}
				if (ALPHASET(i2) != SetName::IA && ALPHASET(i2) != SetName::IB) {
					ALPHAF(i2) = ALPHAF(i2) - ((ALPHAUP(i1) - ALPHADW(i1)) - (a1 - a1a)) * K12 - ((ALPHAUP(i2) - ALPHADW(i2)) - (a2 - a2a)) * K22;
				}
				/* ----------- */
				// update Fi in Io_Cache and vote settings.b_low & settings.b_up if possible
				if (cache.size() > 1) { // if (NULL != cache)
					CACHEF(0) -= ((ALPHAUP(i1) - ALPHADW(i1)) - (a1 - a1a)) * kernel(i1, CACHEI(0));
					CACHEF(0) -= ((ALPHAUP(i2) - ALPHADW(i2)) - (a1 - a1a)) * kernel(i2, CACHEI(0));
					settings.i_up = settings.i_low = CACHEI(0);
					if (ALPHASET(CACHEI(0)) == SetName::IA) {
						settings.b_up = settings.b_low = CACHEF(0) - settings.epsilon * (1.0 - settings.beta);
					}
					else if (ALPHASET(CACHEI(0)) == SetName::IB) {
						settings.b_up = settings.b_low = CACHEF(0) + settings.epsilon * (1.0 - settings.beta);
					}
					else { printf("Error in Io_Cache List \n"); }

					double tmp = 0.0;
					for (unsigned int i = 1; i < cache.size(); ++i) {
						CACHEF(i) -= ((ALPHAUP(i1) - ALPHADW(i1)) - (a1 - a1a)) * kernel(i1, CACHEI(i));
						CACHEF(i) -= ((ALPHAUP(i2) - ALPHADW(i2)) - (a1 - a1a)) * kernel(i2, CACHEI(i));

						if (ALPHASET(CACHEI(i)) == SetName::IA) {
							tmp = CACHEF(i) - settings.epsilon * (1.0 - settings.beta);
						}
						else if (ALPHASET(CACHEI(i)) == SetName::IB) {
							tmp = CACHEF(i) + settings.epsilon * (1.0 - settings.beta);
						}
						else { printf("error in takeStep to create Io_Cache\n"); }

						if (tmp < settings.b_up) {
							settings.i_up = CACHEI(i);
							settings.b_up = tmp;
						}
						else if (tmp > settings.b_low) {
							settings.i_low = CACHEI(i);
							settings.b_low = tmp;
						}
					}
					/* check i1 & i2 here */
					if ((name1 != SetName::IA) && (name1 != SetName::IB)) {
						if (name1 == SetName::I1)
						{
							settings.i_up = (ALPHAF(i1) + (1.0 - settings.beta) * settings.epsilon < settings.b_up) ? i1 : settings.i_up;
							settings.b_up = (ALPHAF(i1) + (1.0 - settings.beta) * settings.epsilon < settings.b_up) ? ALPHAF(i1) + (1.0 - settings.beta) * settings.epsilon : settings.b_up;
							settings.i_low = (ALPHAF(i1) - (1.0 - settings.beta) * settings.epsilon > settings.b_low) ? i1 : settings.i_low;
							settings.b_low = (ALPHAF(i1) - (1.0 - settings.beta) * settings.epsilon > settings.b_low) ? ALPHAF(i1) - (1.0 - settings.beta) * settings.epsilon : settings.b_low;
						}
						else if (name1 == SetName::I2)
						{
							settings.i_low = (ALPHAF(i1) + (1.0 - settings.beta) * settings.epsilon > settings.b_low) ? i1 : settings.i_low;
							settings.b_low = (ALPHAF(i1) + (1.0 - settings.beta) * settings.epsilon > settings.b_low) ? ALPHAF(i1) + (1.0 - settings.beta) * settings.epsilon : settings.b_low;
						}
						else if (name1 == SetName::I3)
						{
							settings.i_up = (ALPHAF(i1) - (1.0 - settings.beta) * settings.epsilon < settings.b_up) ? i1 : settings.i_up;
							settings.b_up = (ALPHAF(i1) - (1.0 - settings.beta) * settings.epsilon < settings.b_up) ? ALPHAF(i1) - (1.0 - settings.beta) * settings.epsilon : settings.b_up;
						}
						else { printf("error in takeStep fail to add i1 to Io_Cache\n"); }
					}
					if ((name2 != SetName::IA) && (name2 != SetName::IB)) {
						if (name2 == SetName::I1)
						{
							settings.i_up = (ALPHAF(i2) + (1.0 - settings.beta) * settings.epsilon < settings.b_up) ? i2 : settings.i_up;
							settings.b_up = (ALPHAF(i2) + (1.0 - settings.beta) * settings.epsilon < settings.b_up) ? ALPHAF(i2) + (1.0 - settings.beta) * settings.epsilon : settings.b_up;
							settings.i_low = (ALPHAF(i2) - (1.0 - settings.beta) * settings.epsilon > settings.b_low) ? i2 : settings.i_low;
							settings.b_low = (ALPHAF(i2) - (1.0 - settings.beta) * settings.epsilon > settings.b_low) ? ALPHAF(i2) - (1.0 - settings.beta) * settings.epsilon : settings.b_low;
						}
						else if (name2 == SetName::I2)
						{
							settings.i_low = (ALPHAF(i2) + (1.0 - settings.beta) * settings.epsilon > settings.b_low) ? i2 : settings.i_low;
							settings.b_low = (ALPHAF(i2) + (1.0 - settings.beta) * settings.epsilon > settings.b_low) ? ALPHAF(i2) + (1.0 - settings.beta) * settings.epsilon : settings.b_low;
						}
						else if (name2 == SetName::I3)
						{
							settings.i_up = (ALPHAF(i2) - (1.0 - settings.beta) * settings.epsilon < settings.b_up) ? i2 : settings.i_up;
							settings.b_up = (ALPHAF(i2) - (1.0 - settings.beta) * settings.epsilon < settings.b_up) ? ALPHAF(i2) - (1.0 - settings.beta) * settings.epsilon : settings.b_up;
						}
						else { printf("error in takeStep fail to add i1 to Io_Cache\n"); }
					}
				}
				else {
					if (name1 == SetName::I3 && name2 == SetName::I2)
					{
						settings.i_up = i1; settings.b_up = ALPHAF(i1) - (1.0 - settings.beta) * settings.epsilon;
						settings.i_low = i2; settings.b_low = ALPHAF(i2) + (1.0 - settings.beta) * settings.epsilon;
					}
					else if (name1 == SetName::I2 && name2 == SetName::I3)
					{
						settings.i_up = i2;	settings.b_up = ALPHAF(i2) - (1.0 - settings.beta) * settings.epsilon;
						settings.i_low = i1; settings.b_low = ALPHAF(i1) + (1.0 - settings.beta) * settings.epsilon;
					}
					else if (name1 == SetName::I1 && name2 == SetName::I2)
					{
						settings.i_up = i1; settings.b_up = ALPHAF(i1) + (1.0 - settings.beta) * settings.epsilon;
						settings.i_low = (ALPHAF(i1) - (1.0 - settings.beta) * settings.epsilon > ALPHAF(i2) + (1.0 - settings.beta) * settings.epsilon) ? i1 : i2;
						settings.b_low = (ALPHAF(i1) - (1.0 - settings.beta) * settings.epsilon > ALPHAF(i2) + (1.0 - settings.beta) * settings.epsilon) ? ALPHAF(i1) - (1.0 - settings.beta) * settings.epsilon : ALPHAF(i2) + (1.0 - settings.beta) * settings.epsilon;
					}
					else if (name1 == SetName::I1 && name2 == SetName::I3)
					{
						settings.i_low = i1; settings.b_low = ALPHAF(i1) - (1.0 - settings.beta) * settings.epsilon;
						settings.i_up = (ALPHAF(i1) + (1.0 - settings.beta) * settings.epsilon < ALPHAF(i2) - (1.0 - settings.beta) * settings.epsilon) ? i1 : i2;
						settings.b_up = (ALPHAF(i1) + (1.0 - settings.beta) * settings.epsilon < ALPHAF(i2) - (1.0 - settings.beta) * settings.epsilon) ? ALPHAF(i1) + (1.0 - settings.beta) * settings.epsilon : ALPHAF(i2) - (1.0 - settings.beta) * settings.epsilon;
					}
					else if (name1 == SetName::I2 && name2 == SetName::I1)
					{
						settings.i_up = i2; settings.b_up = ALPHAF(i2) + (1.0 - settings.beta) * settings.epsilon;
						settings.i_low = (ALPHAF(i1) + (1.0 - settings.beta) * settings.epsilon > ALPHAF(i2) - (1.0 - settings.beta) * settings.epsilon) ? i1 : i2;
						settings.b_low = (ALPHAF(i1) + (1.0 - settings.beta) * settings.epsilon > ALPHAF(i2) - (1.0 - settings.beta) * settings.epsilon) ? ALPHAF(i1) + (1.0 - settings.beta) * settings.epsilon : ALPHAF(i2) - (1.0 - settings.beta) * settings.epsilon;
					}
					else if (name1 == SetName::I3 && name2 == SetName::I1)
					{
						settings.i_low = i2; settings.b_low = ALPHAF(i2) - (1.0 - settings.beta) * settings.epsilon;
						settings.i_up = (ALPHAF(i1) - (1.0 - settings.beta) * settings.epsilon < ALPHAF(i2) + (1.0 - settings.beta) * settings.epsilon) ? i1 : i2;
						settings.b_up = (ALPHAF(i1) - (1.0 - settings.beta) * settings.epsilon < ALPHAF(i2) + (1.0 - settings.beta) * settings.epsilon) ? ALPHAF(i1) - (1.0 - settings.beta) * settings.epsilon : ALPHAF(i2) + (1.0 - settings.beta) * settings.epsilon;
					}
					else if (name1 == SetName::I1 && name2 == SetName::I1) {
						if (ALPHAF(i1) < ALPHAF(i2)) {
							settings.i_up = i1; settings.i_low = i2;
							settings.b_up = ALPHAF(i1) + (1.0 - settings.beta) * settings.epsilon;
							settings.b_low = ALPHAF(i2) - (1.0 - settings.beta) * settings.epsilon;
						}
						else {
							settings.i_up = i2; settings.i_low = i1;
							settings.b_up = ALPHAF(i2) + (1.0 - settings.beta) * settings.epsilon;
							settings.b_low = ALPHAF(i1) - (1.0 - settings.beta) * settings.epsilon;
						}
					}
					else if (name1 == SetName::I2 && name2 == SetName::I2) {
						settings.i_low = (ALPHAF(i1) + (1.0 - settings.beta) * settings.epsilon > ALPHAF(i2) + (1.0 - settings.beta) * settings.epsilon) ? i1 : i2;
						settings.b_low = (ALPHAF(i1) + (1.0 - settings.beta) * settings.epsilon > ALPHAF(i2) + (1.0 - settings.beta) * settings.epsilon) ? ALPHAF(i1) + (1.0 - settings.beta) * settings.epsilon : ALPHAF(i2) + (1.0 - settings.beta) * settings.epsilon;
						settings.b_up = ALPHAF(settings.i_up) + (1.0 - settings.beta) * settings.epsilon;
					}
					else if (name1 == SetName::I3 && name2 == SetName::I3) {
						settings.i_up = (ALPHAF(i1) - (1.0 - settings.beta) * settings.epsilon < ALPHAF(i2) - (1.0 - settings.beta) * settings.epsilon) ? i1 : i2;
						settings.b_up = (ALPHAF(i1) - (1.0 - settings.beta) * settings.epsilon < ALPHAF(i2) - (1.0 - settings.beta) * settings.epsilon) ? \
							ALPHAF(i1) - (1.0 - settings.beta) * settings.epsilon : ALPHAF(i2) - (1.0 - settings.beta) * settings.epsilon;
						settings.b_low = ALPHAF(settings.i_low) - (1.0 - settings.beta) * settings.epsilon;
					}
					else { printf("error in takeStep, can not locate setname, %d and %d.\n", name1, name2); }
				}
				return true;
			} // end of update 
			else { return false; }
		}
		unsigned int examine_example(const unsigned int& widx) override {
			double F2 = 0.0, tmp = 0.0;
			bool optimal = true;
			SetName set = ALPHASET(widx);
			unsigned int i1 = 0;
			unsigned int i2 = widx;

			if (settings.beta < 0.0) settings.beta = 0.0;
			else if (settings.beta > 1.0) settings.beta = 1.0;

			/* Calculate F2 */
			if (set == SetName::IA || set == SetName::IB) {
				F2 = ALPHAF(widx);
			}
			else {
				F2 = calculate_fi(i2);
				ALPHAF(widx) = F2;
				switch (set) {
				case SetName::I1: {
					tmp = F2 + (1.0 - settings.beta) * settings.epsilon;
					if (tmp < settings.b_up)
					{
						settings.b_up = tmp;
						settings.i_up = i2;
					}
					tmp = F2 - (1.0 - settings.beta) * settings.epsilon;
					if (tmp > settings.b_low)
					{
						settings.b_low = tmp;
						settings.i_low = i2;
					}
					break;
				}
				case SetName::I2: {
					tmp = F2 + (1.0 - settings.beta) * settings.epsilon;
					if (tmp > settings.b_low)
					{
						settings.b_low = tmp;
						settings.i_low = i2;
					}
					break;
				}
				case SetName::I3: {
					tmp = F2 - (1.0 - settings.beta) * settings.epsilon;
					if (tmp < settings.b_up)
					{
						settings.b_up = tmp;
						settings.i_up = i2;
					}
					break;
				}
				default:
					printf("The Sample has a wrong Set Name.");
				}
			}
			/* ----------- */
			switch (set) {
			case SetName::IA: {
				tmp = F2 - (1.0 - settings.beta) * settings.epsilon;
				if (-tmp > settings.tol || tmp > settings.tol)
				{
					optimal = false;
					if (tmp > -tmp) i1 = settings.i_up;
					else i1 = settings.i_low;
				}
				break;
			}
			case SetName::IB: {
				tmp = F2 + (1.0 - settings.beta) * settings.epsilon;
				if (-tmp > settings.tol || tmp > settings.tol)
				{
					optimal = false;
					if (tmp > -tmp) i1 = settings.i_up;
					else i1 = settings.i_low;
				}
				break;
			}
			case SetName::I1: {
				tmp = F2 + (1.0 - settings.beta) * settings.epsilon;
				if (-tmp > settings.tol)
				{
					optimal = false;
					i1 = settings.i_low;
					if (tmp > -tmp) i1 = settings.i_up;
				}
				else
				{
					tmp = F2 - (1.0 - settings.beta) * settings.epsilon;
					if (tmp > settings.tol)
					{
						optimal = false;
						i1 = settings.i_up;
						if (-tmp > tmp) i1 = settings.i_low;
					}
				}
				break;
			}
			case SetName::I2: {
				if ((F2 + (1.0 - settings.beta) * settings.epsilon) > settings.tol)
				{
					optimal = false;
					i1 = settings.i_up;
				}
				break;
			}
			case SetName::I3: {
				if (-(F2 - (1.0 - settings.beta) * settings.epsilon) > settings.tol)
				{
					optimal = false;
					i1 = settings.i_low;
				}
				break;
			}
			default:
				printf("The Sample has a wrong Set Name.");
			}
			/* ----------- */
			if (optimal) return 0;
			else if (take_step(i1, i2)) return 1;
			else return 0;

		}
		double objective_value() override {
			/* Objective Function : Call USMO Routine to find MAP (bismo_regression : bismo_routine.cpp) */
			{
				//CacheNode* cache = NULL; Alphas* alpha = NULL;
				/* if ( FALSE == SMO_QUICKMODE ) */
				{
					// alpha = alpha + i_ymin;
					unsigned int widx = i_ymin;
					//settings.b_up = alpha->pair->target + settings.epsilon * (1.0 - settings.beta);
					settings.b_up = outputs(widx, 0) + settings.epsilon * (1.0 - settings.beta);
					//alpha->f_cache = alpha->output;
					alpha[widx, 3] = outputs(widx, 0);
					//settings.i_up = alpha->pair->index;
					settings.i_up = widx;
					//alpha = ALPHA + settings->pairs->i_ymax - 1;
					widx = i_ymax;
					//settings.b_low = alpha->pair->target - settings.epsilon * (1.0 - settings.beta);
					settings.b_low = outputs(widx, 0) - settings.epsilon * (1.0 - settings.beta);
					//alpha->f_cache = alpha->pair->target;
					alpha[widx, 3] = outputs(widx, 0);
					//settings.i_low = alpha->pair->index;
					settings.i_low = widx;
				}
				/* Main Routine */
				{
					bool examineAll = true;
					unsigned int numChanged = 0;
					while (numChanged > 0 || examineAll) {
						numChanged = 0;
						if (examineAll) {
							// loop over all pairs
							for (unsigned int idx = 0; idx <= nrows; idx++) {
								numChanged += examine_example(idx);
							}
							std::cout << "b_up - b_low = " << settings.b_up - settings.b_low << " | " << "Changed = " << numChanged << std::endl;
						}
						else {
							/* Default Method SKTWO */
							// check the worse pair	
							numChanged = 1;
							while (!(settings.b_up > -settings.tol && settings.b_low < settings.tol) && numChanged > 0.5) {
								if (!take_step(settings.i_up, settings.i_low)) {
									std::cout << "I_UP = " << settings.i_up << " | " << "I_LOW = " << settings.i_low << " Failed to update" << std::endl;
									numChanged = 0;
								}
							}
							numChanged = 0;
							std::cout << "b_up - b_low = " << settings.b_up - settings.b_low << " | " << "Cache Size = " << cache.size() << std::endl;
						}

						if (examineAll) examineAll = false;
						else if (numChanged == 0) examineAll = true;
					}
				}

				settings.bias = 0.0;

				if (!(settings.b_up > -settings.tol && settings.b_low < settings.tol))
				{
					printf("Warning: KKT conditions are violated on bias!!! %f\r\n", settings.b_low - settings.b_up);
					printf("C=%f, Epsilon=%f, Beta= %f, Kaio=%f, Kai=%f", settings.vc, settings.epsilon, settings.beta, settings.kaio, settings.kai);
				}
			}
			/* Perform Checks */
			{}

			/* Calculate Error Term */
			// Find Indices where SetName::IA || SetName::IB
			std::vector<unsigned int> indices;
			unsigned int num = 0, mnum = 0;
			double errorterm = 0.0, errorterma = 0.0, errortermb = 0.0, errortermc = 0.0;
			for (unsigned int i = 0; i < nrows; ++i) {
				if (set_names[i] == SetName::I1) {
					num++;
					if (set_names[i] == SetName::I2 || set_names[i] == SetName::I3) {
						errorterm = errorterm + fabs(ALPHAF(i) + (ALPHAUP(i) - ALPHADW(i)) * 2.0 * settings.beta * settings.epsilon / settings.vc - settings.bias) - settings.epsilon;
						errortermc += settings.epsilon;
					}
					else if (set_names[i] == SetName::IA || set_names[i] == SetName::IB) {
						mnum++;
						indices.push_back(i);
						errorterma += (fabs(ALPHAF(i) + (ALPHAUP(i) - ALPHADW(i)) * 2.0 * settings.beta * settings.epsilon / settings.vc - settings.bias) - settings.epsilon * (1.0 - settings.beta)) * (fabs(ALPHAF(i) + (ALPHAUP(i) - ALPHADW(i)) * 2.0 * settings.beta * settings.epsilon / settings.vc - settings.bias) - settings.epsilon * (1.0 - settings.beta));
						errortermb += (fabs(ALPHAF(i) + (ALPHAUP(i) - ALPHADW(i)) * 2.0 * settings.beta * settings.epsilon / settings.vc - settings.bias) - settings.epsilon * (1.0 - settings.beta));
					}

				}
			}

			errortermc *= settings.vc;
			errorterma = errorterma / 4.0 / settings.beta / settings.epsilon;
			errortermb = errortermb / 2.0;
			errorterm += errorterma;

			if (cache.size() != mnum || num == 0)
			{
				printf("\r\nWarning: SIGMAm Matrix is shrinking.\r\n");
			}
			TMatrix sigmam = TMatrix::Zero(mnum, mnum);
			TMatrix adjsgmm = TMatrix::Zero(mnum, mnum);
			TVector t1 = TVector::Zero(mnum);
			TVector t2 = TVector::Zero(mnum);
			// If ARD
			TMatrix dsigmam = TMatrix::Zero(mnum, mnum);

			// Initialize Matrix Sigma_m
			for(const unsigned int& i : indices) {
				for (const unsigned int& j : indices) {
					sigmam(i, j) = kernel(i, j);					
					adjsgmm(i, j) = sigmam(i, j) + (2.0 * settings.beta * settings.epsilon / settings.vc);
				}
			}


			return 0.0;
		}

		void set_params(const TVector& t) override {
			/* Update Theta */
			for (unsigned int i = settings.adjnum; i < settings.number; i++)
			{
				theta[i] = t[i];
			}
			/* check the validation of hyperparameter */
			{
				for (unsigned int i = settings.adjnum; i < settings.number; i++)
				{
					if (theta[i] > settings.suplengthscale)
						theta[i] = settings.suplengthscale;
					if (theta[i] < settings.inflengthscale)
						theta[i] = settings.inflengthscale;
				}
				if (theta[0] > settings.sup_vc)
					theta[0] = settings.sup_vc;
				if (theta[0] < settings.inf_vc)
					theta[0] = settings.inf_vc;
				if (theta[1] > settings.sup_epsilon)
					theta[1] = settings.sup_epsilon;
				if (theta[1] < settings.inf_epsilon)
					theta[1] = settings.inf_epsilon;
				if (theta[2] > settings.sup_kappao)
					theta[2] = settings.sup_kappao;
				if (theta[2] < settings.inf_kappao)
					theta[2] = settings.inf_kappao;
			}

		}


		void train() override {
			Objective obj(this);
			LBFGSB solver;
			solver.solve(theta, obj);

		}

	};




}




#endif // !_MODEL_H

/**/
//struct Objective : public Problem {
//	Objective(Model* model) : Problem(model->settings.number), model(model) {}
//	void operator()(const TVector& theta) override {
//		/* Update Theta */
//		for (unsigned int i = settings.adjnum; i < settings.number; i++)
//		{
//			model->theta[i] = theta[i];
//		}
//		/* check the validation of hyperparameter */
//		{
//			for (unsigned int i = settings.adjnum; i < settings.number; i++)
//			{
//				if (model->theta[i] > settings.suplengthscale)
//					model->theta[i] = settings.suplengthscale;
//				if (model->theta[i] < settings.inflengthscale)
//					model->theta[i] = settings.inflengthscale;
//			}
//			if (model->theta[0] > settings.sup_vc)
//				model->theta[0] = settings.sup_vc;
//			if (model->theta[0] < settings.inf_vc)
//				model->theta[0] = settings.inf_vc;
//			if (model->theta[1] > settings.sup_epsilon)
//				model->theta[1] = settings.sup_epsilon;
//			if (model->theta[1] < settings.inf_epsilon)
//				model->theta[1] = settings.inf_epsilon;
//			if (model->theta[2] > settings.sup_kappao)
//				model->theta[2] = settings.sup_kappao;
//			if (model->theta[2] < settings.inf_kappao)
//				model->theta[2] = settings.inf_kappao;
//		}
//
//		// Begin : Regression_Evaluate_FuncGrad (bit_funcgrad.cpp)
//		double errorterm = 0.0, errorterma = 0.0, errortermb = 0.0, errortermc = 0.0, capnum = 0.0;
//		double sumlam = 0.0, sumloglam = 0.0, cbe = 0.0, scbe = 0.0, expncbe = 0.0, zd = 0.0, temp = 0.0;
//		double gkaio = 0.0, gkappa = 0.0;
//		double erfscbe;
//		//double* dsigmam = NULL;
//		TVector dsigmam;
//
//		//Alphas* alpha; Alphas* alphab;
//
//		/* Call USMO Routine to find MAP (bismo_regression : bismo_routine.cpp) */
//		{
//			auto take_step = [&](const unsigned int& widx1, const unsigned int& widx2) {
//				double tmp = 1;
//				double a1 = 0.0, a1a = 0.0, a2 = 0.0, a2a = 0.0;	//old alpha
//				double n1 = 0.0, n1a = 0.0, n2 = 0.0, n2a = 0.0;	//new alpha
//
//				SetName name1, name2;
//				unsigned int i1 = widx1; // alpha1
//				unsigned int i2 = widx2; // alpha2
//				if (i1 == i2) return 0;
//				/* ----------- */
//				a1 = n1 = ALPHAUP(i1);
//				a1a = n1a = ALPHADW(i1);
//				a2 = n2 = ALPHAUP(i2);
//				a2a = n2a = ALPHADW(i2);
//				double F1 = ALPHAF(i1);
//				double F2 = ALPHAF(i2);
//				double K11 = ALPHAKD(i1)
//				double K22 = ALPHAKD(i2)
//				double K12 = model->kernel(i1, i2);
//				double detH = K11 * K22 - K12 * K12;
//				if (!((ALPHASET(i1) == SetName::IA || ALPHASET(i1) == SetName::IB) && (ALPHASET(i2) == SetName::IA || ALPHASET(i2) == SetName::IB))) {
//					detH = (K11 * K22 - K12 * K12) / settings.tol;
//				}
//				/* ----------- */
//				double G1, G2, G1a, G2a;
//				bool case1 = false, case2 = false, case3 = false, case4 = false;
//				bool finish = false, update = false;
//				if (detH <= 0.0) return 0;
//				else {
//					// consider four quadrants together
//					while (!finish)
//					{
//						G1 = -F1 + (1 - settings.beta) * settings.epsilon;
//						G2 = -F2 + (1 - settings.beta) * settings.epsilon;
//						G1a = F1 + (1 - settings.beta) * settings.epsilon;
//						G2a = F2 + (1 - settings.beta) * settings.epsilon;
//
//						//this loop is passed at most three times.
//						if (!case1 && (a1 > 0.0 || (a1a == 0.0 && (-K22 * G1 + K12 * G2) > 0.0)) && (a2 > 0.0 || (a2a == 0.0 && (K12 * G1 - K11 * G2) > 0.0)))
//						{
//							case1 = true; // (a1, a2)
//							n1 = a1 + (-K22 * G1 + K12 * G2) / detH;
//							n2 = a2 + (K12 * G1 - K11 * G2) / detH;
//							//check constraints
//							if (n1 < 0.0) n1 = 0.0;
//							else if (n1 > settings.vc) n1 = settings.vc;
//							if (n2 < 0.0) n2 = 0.0;
//							else if (n2 > settings.vc) n2 = settings.vc;
//							n1a = 0.0;
//							n2a = 0.0;
//							//update if significant 
//							if (fabs(n1 - a1) + fabs(n2 - a2) > 0.0)
//							{
//								update = true;
//							}
//						}
//						else if (!case2 && (a1 > 0.0 || (a1a == 0.0 && (-K22 * G1 - K12 * G2a) > 0.0)) && (a2a > 0.0 || (a2 == 0.0 && (-K12 * G1 - K11 * G2a) > 0.0)))
//						{
//							case2 = true; // (a1, a2a)	
//							n1 = a1 + (-K22 * G1 - K12 * G2a) / detH;
//							n2a = a2a + (-K12 * G1 - K11 * G2a) / detH;
//							//check constraints
//							if (n1 < 0.0) n1 = 0.0;
//							else if (n1 > settings.vc) n1 = settings.vc;
//							if (n2a < 0.0) n2a = 0.0;
//							else if (n2a > settings.vc) n2a = settings.vc;
//							n1a = 0.0;
//							n2 = 0.0;
//							//update if significant 
//							if (fabs(n1 - a1) + fabs(n2a - a2a) > 0.0)
//							{
//								update = true;
//							}
//						}
//						else if (!case3 && (a1a > 0.0 || (a1 == 0.0 && (-K22 * G1a - K12 * G2) > 0.0)) && (a2 > 0.0 || (a2a == 0.0 && (-K12 * G1a - K11 * G2) > 0.0))) {
//							case3 = true; // (a1a, a2)
//							n1a = a1a + (-K22 * G1a - K12 * G2) / detH;
//							n2 = a2 + (-K12 * G1a - K11 * G2) / detH;
//							//check constraints
//							if (n1a < 0.0) n1a = 0.0;
//							else if (n1a > settings.vc) n1a = settings.vc;
//							if (n2 < 0.0) n2 = 0.0;
//							else if (n2 > settings.vc) n2 = settings.vc;
//							n1 = 0.0;
//							n2a = 0.0;
//							//update if significant 
//							if (fabs(n1a - a1a) + fabs(n2 - a2) > 0.0) {
//								update = true;
//							}
//						}
//						else if (!case4 && (a1a > 0.0 || (a1 == 0.0 && (-K22 * G1a + K12 * G2a) > 0.0)) && (a2a > 0.0 || (a2 == 0.0 && (K12 * G1a - K11 * G2a) > 0.0))) {
//							case4 = true; // (a1a, a2a)			
//							n1a = a1a + (-K22 * G1a + K12 * G2a) / detH;
//							n2a = a2a + (K12 * G1a - K11 * G2a) / detH;
//							//check constraints
//							if (n1a < 0.0) n1a = 0.0;
//							else if (n1a > settings.vc) n1a = settings.vc;
//							if (n2a < 0.0) n2a = 0.0;
//							else if (n2a > settings.vc) n2a = settings.vc;
//							n1 = 0.0;
//							n2 = 0.0;
//							//update if significant 
//							if (fabs(n1a - a1a) + fabs(n2a - a2a) > 0.0)
//							{
//								update = true;
//							}
//						}
//						else { finish = true; }
//						// update Fi cache if necessary
//						if (update)
//						{
//							//update F1 & F2
//							F1 = F1 + (a1 - a1a - n1 + n1a) * K11 + (a2 - a2a - n2 + n2a) * K12;
//							F2 = F2 + (a2 - a2a - n2 + n2a) * K22 + (a1 - a1a - n1 + n1a) * K12;
//							a1 = n1; a2 = n2; a1a = n1a; a2a = n2a;
//							update = false;
//						}
//					}
//					// end of while
//				}
//				/* ----------- */
//				// update Alpha List if necessary, then update Io_Cache, and vote settings.b_low & settings.b_up
//				if (fabs((n2 - n2a) - (ALPHAUP(i2) - ALPHADW(i2))) > 0.0 || fabs((n1 - n1a) - (ALPHAUP(i1) - ALPHADW(i1))) > 0.0) {
//					// store alphas in Alpha List
//					a1a = ALPHADW(i1);  a1 = ALPHAUP(i1);
//					a2a = ALPHADW(i2); a2 = ALPHAUP(i2);
//					ALPHADW(i1) = n1a; ALPHAUP(i1) = n1;
//					ALPHADW(i1) = n2a; ALPHAUP(i2) = n2;
//					/* ----------- */
//					// update Set & Cache_List  
//					name1 = set_name(n1, n1a); name2 = set_name(n2, n2a);
//					if (ALPHASET(i1) != name1)
//					{
//						if ((name1 == SetName::IA || name1 == SetName::IB) && (ALPHASET(i1) != SetName::IA && ALPHASET(i1) != SetName::IB)) {
//							// Add_Cache_Node( &Io_CACHE, alpha1 ) ; // insert into Io 
//							AlphaCache tmpalpha(ALPHA, i1);
//							model->io_cache.alphas.push_back(tmpalpha);
//						}
//						if ((ALPHASET(i1) == SetName::IA || ALPHASET(i1) == SetName::IB) && name1 != SetName::IA && name1 != SetName::IB) {
//							// Del_Cache_Node( &Io_CACHE, alpha1 ) ;
//							model->io_cache.alphas.erase(INDEXCACHE(i1));
//						}
//						ALPHASET(i1) = name1;
//					}
//					if (ALPHASET(i2) != name2)
//					{
//						if ((name2 == SetName::IA || name2 == SetName::IB) && (ALPHASET(i2) != SetName::IA && ALPHASET(i2) != SetName::IB)) {
//							// Add_Cache_Node( &Io_CACHE, alpha1 ) ; // insert into Io 
//							AlphaCache tmpalpha(ALPHA, i2);
//							model->io_cache.alphas.push_back(tmpalpha);
//						}
//						if ((ALPHASET(i2) == SetName::IA || ALPHASET(i2) == SetName::IB) && name2 != SetName::IA && name2 != SetName::IB) {
//							// Del_Cache_Node( &Io_CACHE, alpha2 ) ;
//							model->io_cache.alphas.erase(INDEXCACHE(i2));
//						}
//						ALPHASET(i2) = name2;
//					}
//					/* ----------- */
//					// update f-cache of i1 & i2 if not in Io_Cache
//					if (ALPHASET(i1) != SetName::IA && ALPHASET(i1) != SetName::IB) {
//						ALPHAF(i1) = ALPHAF(i1) - ((ALPHA.matrix(i1, 1) - ALPHA.matrix(i1, 0)) - (a1 - a1a)) * K11 - ((ALPHA.matrix(i2, 1) - ALPHA.matrix(i2, 0)) - (a2 - a2a)) * K12;
//					}
//					if (ALPHASET(i2) != SetName::IA && ALPHASET(i2) != SetName::IB) {
//						ALPHAF(i2) = ALPHAF(i2) - ((ALPHA.matrix(i1, 1) - ALPHA.matrix(i1, 0)) - (a1 - a1a)) * K12 - ((ALPHA.matrix(i2, 1) - ALPHA.matrix(i2, 0)) - (a2 - a2a)) * K22;
//					}
//					/* ----------- */
//					// update Fi in Io_Cache and vote settings.b_low & settings.b_up if possible
//					// cache = model->io_cache->front;
//					if (INDEXCACHE(0)->index != -1) { // if (NULL != cache)
//						INDEXCACHE(0)->f_cache -= ((ALPHAUP(i1) - ALPHADW(i1)) - (a1 - a1a)) * model->kernel(i1, INDEXCACHE(0)->index);
//						INDEXCACHE(0)->f_cache -= ((ALPHAUP(i2) - ALPHADW(i2)) - (a1 - a1a)) * model->kernel(i2, INDEXCACHE(0)->index);
//						settings.i_up = settings.i_low = INDEXCACHE(0)->index;
//						if (INDEXCACHE(0)->setname == SetName::IA) {
//							settings.b_up = settings.b_low = INDEXCACHE(0)->f_cache - settings.epsilon * (1.0 - settings.beta);
//						}
//						else if (INDEXCACHE(0)->setname == SetName::IB) {
//							settings.b_up = settings.b_low = INDEXCACHE(0)->f_cache + settings.epsilon * (1.0 - settings.beta);
//						}
//						else { printf("Error in Io_Cache List \n"); }
//						if (model->io_cache.alphas.size() > 1) { // while ( NULL != cache )
//							for (unsigned int i = 1; i < model->io_cache.alphas.size(); ++i) {
//								INDEXCACHE(i)->f_cache -= ((ALPHAUP(i1) - ALPHADW(i1)) - (a1 - a1a)) * model->kernel(i1, INDEXCACHE(i)->index);
//								INDEXCACHE(i)->f_cache -= ((ALPHAUP(i2) - ALPHADW(i2)) - (a1 - a1a)) * model->kernel(i2, INDEXCACHE(i)->index);
//
//								if (INDEXCACHE(i)->setname == SetName::IA) {
//									tmp = INDEXCACHE(i)->f_cache - settings.epsilon * (1.0 - settings.beta);
//								}
//								else if (INDEXCACHE(i)->setname == SetName::IB) {
//									tmp = INDEXCACHE(i)->f_cache + settings.epsilon * (1.0 - settings.beta);
//								}
//								else { printf("error in takeStep to create Io_Cache\n"); }
//
//								if (tmp < settings.b_up) {
//									settings.i_up = INDEXCACHE(i)->index;
//									settings.b_up = tmp;
//								}
//								else if (tmp > settings.b_low) {
//									settings.i_low = INDEXCACHE(i)->index;
//									settings.b_low = tmp;
//								}
//							}
//						}
//
//						/* check i1 & i2 here */
//						if ((name1 != SetName::IA) && (name1 != SetName::IB)) {
//							if (name1 == SetName::I1)
//							{
//								settings.i_up = (ALPHAF(i1) + (1.0 - settings.beta) * settings.epsilon < settings.b_up) ? i1 : settings.i_up;
//								settings.b_up = (ALPHAF(i1) + (1.0 - settings.beta) * settings.epsilon < settings.b_up) ? ALPHAF(i1) + (1.0 - settings.beta) * settings.epsilon : settings.b_up;
//								settings.i_low = (ALPHAF(i1) - (1.0 - settings.beta) * settings.epsilon > settings.b_low) ? i1 : settings.i_low;
//								settings.b_low = (ALPHAF(i1) - (1.0 - settings.beta) * settings.epsilon > settings.b_low) ? ALPHAF(i1) - (1.0 - settings.beta) * settings.epsilon : settings.b_low;
//							}
//							else if (name1 == SetName::I2)
//							{
//								settings.i_low = (ALPHAF(i1) + (1.0 - settings.beta) * settings.epsilon > settings.b_low) ? i1 : settings.i_low;
//								settings.b_low = (ALPHAF(i1) + (1.0 - settings.beta) * settings.epsilon > settings.b_low) ? ALPHAF(i1) + (1.0 - settings.beta) * settings.epsilon : settings.b_low;
//							}
//							else if (name1 == SetName::I3)
//							{
//								settings.i_up = (ALPHAF(i1) - (1.0 - settings.beta) * settings.epsilon < settings.b_up) ? i1 : settings.i_up;
//								settings.b_up = (ALPHAF(i1) - (1.0 - settings.beta) * settings.epsilon < settings.b_up) ? ALPHAF(i1) - (1.0 - settings.beta) * settings.epsilon : settings.b_up;
//							}
//							else { printf("error in takeStep fail to add i1 to Io_Cache\n"); }
//						}
//						if ((name2 != SetName::IA) && (name2 != SetName::IB)) {
//							if (name2 == SetName::I1)
//							{
//								settings.i_up = (ALPHAF(i2) + (1.0 - settings.beta) * settings.epsilon < settings.b_up) ? i2 : settings.i_up;
//								settings.b_up = (ALPHAF(i2) + (1.0 - settings.beta) * settings.epsilon < settings.b_up) ? ALPHAF(i2) + (1.0 - settings.beta) * settings.epsilon : settings.b_up;
//								settings.i_low = (ALPHAF(i2) - (1.0 - settings.beta) * settings.epsilon > settings.b_low) ? i2 : settings.i_low;
//								settings.b_low = (ALPHAF(i2) - (1.0 - settings.beta) * settings.epsilon > settings.b_low) ? ALPHAF(i2) - (1.0 - settings.beta) * settings.epsilon : settings.b_low;
//							}
//							else if (name2 == SetName::I2)
//							{
//								settings.i_low = (ALPHAF(i2) + (1.0 - settings.beta) * settings.epsilon > settings.b_low) ? i2 : settings.i_low;
//								settings.b_low = (ALPHAF(i2) + (1.0 - settings.beta) * settings.epsilon > settings.b_low) ? ALPHAF(i2) + (1.0 - settings.beta) * settings.epsilon : settings.b_low;
//							}
//							else if (name2 == SetName::I3)
//							{
//								settings.i_up = (ALPHAF(i2) - (1.0 - settings.beta) * settings.epsilon < settings.b_up) ? i2 : settings.i_up;
//								settings.b_up = (ALPHAF(i2) - (1.0 - settings.beta) * settings.epsilon < settings.b_up) ? ALPHAF(i2) - (1.0 - settings.beta) * settings.epsilon : settings.b_up;
//							}
//							else { printf("error in takeStep fail to add i1 to Io_Cache\n"); }
//						}
//					}
//					else {
//						if (name1 == SetName::I3 && name2 == SetName::I2)
//						{
//							settings.i_up = i1; settings.b_up = ALPHAF(i1) - (1.0 - settings.beta) * settings.epsilon;
//							settings.i_low = i2; settings.b_low = ALPHAF(i2) + (1.0 - settings.beta) * settings.epsilon;
//						}
//						else if (name1 == SetName::I2 && name2 == SetName::I3)
//						{
//							settings.i_up = i2;	settings.b_up = ALPHAF(i2) - (1.0 - settings.beta) * settings.epsilon;
//							settings.i_low = i1; settings.b_low = ALPHAF(i1) + (1.0 - settings.beta) * settings.epsilon;
//						}
//						else if (name1 == SetName::I1 && name2 == SetName::I2)
//						{
//							settings.i_up = i1; settings.b_up = ALPHAF(i1) + (1.0 - settings.beta) * settings.epsilon;
//							settings.i_low = (ALPHAF(i1) - (1.0 - settings.beta) * settings.epsilon > ALPHAF(i2) + (1.0 - settings.beta) * settings.epsilon) ? i1 : i2;
//							settings.b_low = (ALPHAF(i1) - (1.0 - settings.beta) * settings.epsilon > ALPHAF(i2) + (1.0 - settings.beta) * settings.epsilon) ? ALPHAF(i1) - (1.0 - settings.beta) * settings.epsilon : ALPHAF(i2) + (1.0 - settings.beta) * settings.epsilon;
//						}
//						else if (name1 == SetName::I1 && name2 == SetName::I3)
//						{
//							settings.i_low = i1; settings.b_low = ALPHAF(i1) - (1.0 - settings.beta) * settings.epsilon;
//							settings.i_up = (ALPHAF(i1) + (1.0 - settings.beta) * settings.epsilon < ALPHAF(i2) - (1.0 - settings.beta) * settings.epsilon) ? i1 : i2;
//							settings.b_up = (ALPHAF(i1) + (1.0 - settings.beta) * settings.epsilon < ALPHAF(i2) - (1.0 - settings.beta) * settings.epsilon) ? ALPHAF(i1) + (1.0 - settings.beta) * settings.epsilon : ALPHAF(i2) - (1.0 - settings.beta) * settings.epsilon;
//						}
//						else if (name1 == SetName::I2 && name2 == SetName::I1)
//						{
//							settings.i_up = i2; settings.b_up = ALPHAF(i2) + (1.0 - settings.beta) * settings.epsilon;
//							settings.i_low = (ALPHAF(i1) + (1.0 - settings.beta) * settings.epsilon > ALPHAF(i2) - (1.0 - settings.beta) * settings.epsilon) ? i1 : i2;
//							settings.b_low = (ALPHAF(i1) + (1.0 - settings.beta) * settings.epsilon > ALPHAF(i2) - (1.0 - settings.beta) * settings.epsilon) ? ALPHAF(i1) + (1.0 - settings.beta) * settings.epsilon : ALPHAF(i2) - (1.0 - settings.beta) * settings.epsilon;
//						}
//						else if (name1 == SetName::I3 && name2 == SetName::I1)
//						{
//							settings.i_low = i2; settings.b_low = ALPHAF(i2) - (1.0 - settings.beta) * settings.epsilon;
//							settings.i_up = (ALPHAF(i1) - (1.0 - settings.beta) * settings.epsilon < ALPHAF(i2) + (1.0 - settings.beta) * settings.epsilon) ? i1 : i2;
//							settings.b_up = (ALPHAF(i1) - (1.0 - settings.beta) * settings.epsilon < ALPHAF(i2) + (1.0 - settings.beta) * settings.epsilon) ? ALPHAF(i1) - (1.0 - settings.beta) * settings.epsilon : ALPHAF(i2) + (1.0 - settings.beta) * settings.epsilon;
//						}
//						else if (name1 == SetName::I1 && name2 == SetName::I1) {
//							if (ALPHAF(i1) < ALPHAF(i2)) {
//								settings.i_up = i1; settings.i_low = i2;
//								settings.b_up = ALPHAF(i1) + (1.0 - settings.beta) * settings.epsilon;
//								settings.b_low = ALPHAF(i2) - (1.0 - settings.beta) * settings.epsilon;
//							}
//							else {
//								settings.i_up = i2; settings.i_low = i1;
//								settings.b_up = ALPHAF(i2) + (1.0 - settings.beta) * settings.epsilon;
//								settings.b_low = ALPHAF(i1) - (1.0 - settings.beta) * settings.epsilon;
//							}
//						}
//						else if (name1 == SetName::I2 && name2 == SetName::I2) {
//							settings.i_low = (ALPHAF(i1) + (1.0 - settings.beta) * settings.epsilon > ALPHAF(i2) + (1.0 - settings.beta) * settings.epsilon) ? i1 : i2;
//							settings.b_low = (ALPHAF(i1) + (1.0 - settings.beta) * settings.epsilon > ALPHAF(i2) + (1.0 - settings.beta) * settings.epsilon) ? ALPHAF(i1) + (1.0 - settings.beta) * settings.epsilon : ALPHAF(i2) + (1.0 - settings.beta) * settings.epsilon;
//							settings.b_up = ALPHA.f_cache[settings.i_up] + (1.0 - settings.beta) * settings.epsilon;
//						}
//						else if (name1 == SetName::I3 && name2 == SetName::I3) {
//							settings.i_up = (ALPHAF(i1) - (1.0 - settings.beta) * settings.epsilon < ALPHAF(i2) - (1.0 - settings.beta) * settings.epsilon) ? i1 : i2;
//							settings.b_up = (ALPHAF(i1) - (1.0 - settings.beta) * settings.epsilon < ALPHAF(i2) - (1.0 - settings.beta) * settings.epsilon) ? \
//								ALPHAF(i1) - (1.0 - settings.beta) * settings.epsilon : ALPHAF(i2) - (1.0 - settings.beta) * settings.epsilon;
//							settings.b_low = ALPHA.f_cache[settings.i_low] - (1.0 - settings.beta) * settings.epsilon;
//						}
//						else { printf("error in takeStep, can not locate setname, %d and %d.\n", name1, name2); }
//					}
//					return 1;
//				} // end of update 
//				else { return 0; }
//			};
//			auto examine_example = [&](const unsigned int& widx) {
//				double F2 = 0.0, tmp = 0.0;
//				bool optimal = true;
//				SetName set = ALPHASET(widx);
//				unsigned int i1 = 0;
//				unsigned int i2 = widx;
//
//				if (settings.beta < 0.0) settings.beta = 0.0;
//				else if (settings.beta > 1.0) settings.beta = 1.0;
//
//				/* Calculate F2 */
//				if (set == SetName::IA || set == SetName::IB) {
//					F2 = ALPHAF(widx);
//				}
//				else {
//					F2 = calculate_fi(i2);
//					ALPHAF(widx) = F2;
//					switch (set) {
//					case SetName::I1: {
//						tmp = F2 + (1.0 - settings.beta) * settings.epsilon;
//						if (tmp < settings.b_up)
//						{
//							settings.b_up = tmp;
//							settings.i_up = i2;
//						}
//						tmp = F2 - (1.0 - settings.beta) * settings.epsilon;
//						if (tmp > settings.b_low)
//						{
//							settings.b_low = tmp;
//							settings.i_low = i2;
//						}
//						break;
//					}
//					case SetName::I2: {
//						tmp = F2 + (1.0 - settings.beta) * settings.epsilon;
//						if (tmp > settings.b_low)
//						{
//							settings.b_low = tmp;
//							settings.i_low = i2;
//						}
//						break;
//					}
//					case SetName::I3: {
//						tmp = F2 - (1.0 - settings.beta) * settings.epsilon;
//						if (tmp < settings.b_up)
//						{
//							settings.b_up = tmp;
//							settings.i_up = i2;
//						}
//						break;
//					}
//					default:
//						printf("The Sample has a wrong Set Name.");
//					}
//				}
//				/* ----------- */
//				switch (set) {
//				case SetName::IA: {
//					tmp = F2 - (1.0 - settings.beta) * settings.epsilon;
//					if (-tmp > settings.tol || tmp > settings.tol)
//					{
//						optimal = false;
//						if (tmp > -tmp) i1 = settings.i_up;
//						else i1 = settings.i_low;
//					}
//					break;
//				}
//				case SetName::IB: {
//					tmp = F2 + (1.0 - settings.beta) * settings.epsilon;
//					if (-tmp > settings.tol || tmp > settings.tol)
//					{
//						optimal = false;
//						if (tmp > -tmp) i1 = settings.i_up;
//						else i1 = settings.i_low;
//					}
//					break;
//				}
//				case SetName::I1: {
//					tmp = F2 + (1.0 - settings.beta) * settings.epsilon;
//					if (-tmp > settings.tol)
//					{
//						optimal = false;
//						i1 = settings.i_low;
//						if (tmp > -tmp) i1 = settings.i_up;
//					}
//					else
//					{
//						tmp = F2 - (1.0 - settings.beta) * settings.epsilon;
//						if (tmp > settings.tol)
//						{
//							optimal = false;
//							i1 = settings.i_up;
//							if (-tmp > tmp) i1 = settings.i_low;
//						}
//					}
//					break;
//				}
//				case SetName::I2: {
//					if ((F2 + (1.0 - settings.beta) * settings.epsilon) > settings.tol)
//					{
//						optimal = false;
//						i1 = settings.i_up;
//					}
//					break;
//				}
//				case SetName::I3: {
//					if (-(F2 - (1.0 - settings.beta) * settings.epsilon) > settings.tol)
//					{
//						optimal = false;
//						i1 = settings.i_low;
//					}
//					break;
//				}
//				default:
//					printf("The Sample has a wrong Set Name.");
//				}
//				/* ----------- */
//				if (optimal) return 0;
//				else if (take_step(i1, i2)) return 1;
//				else return 0;
//			};
//
//			bool examineAll = true;
//			unsigned int numChanged = 0;
//			//CacheNode* cache = NULL; Alphas* alpha = NULL;
//			unsigned int loop = 0, Io_Num = 0;
//			int* Io_Index = NULL;
//			double temp = 0.0;
//
//			//model->free_cache_list();
//			//model->io_cache.alpha
//			//model->free_alphas();
//
//			/* if ( FALSE == SMO_QUICKMODE ) */
//			{
//				// alpha = model->alpha + model->i_ymin;
//				unsigned int widx = model->i_ymin;
//				//settings.b_up = alpha->pair->target + settings.epsilon * (1.0 - settings.beta);
//				settings.b_up = model->outputs(widx, 0) + settings.epsilon * (1.0 - settings.beta);
//				//alpha->f_cache = model->alpha->output;
//				model->alpha.f_cache[widx] = model->outputs(widx, 0);
//				//settings.i_up = alpha->pair->index;
//				settings.i_up = widx;
//				//alpha = ALPHA + settings->pairs->i_ymax - 1;
//				widx = model->i_ymax;
//				//settings.b_low = alpha->pair->target - settings.epsilon * (1.0 - settings.beta);
//				settings.b_low = model->outputs(widx, 0) - settings.epsilon * (1.0 - settings.beta);
//				//alpha->f_cache = alpha->pair->target;
//				model->alpha.f_cache[widx] = model->outputs(widx, 0);
//				//settings.i_low = alpha->pair->index;
//				settings.i_low = widx;
//			}
//			/* Main Routine */
//			{
//				while (numChanged > 0 || examineAll) {
//					numChanged = 0;
//					if (examineAll) {
//						// loop over all pairs
//						for (unsigned int idx = 0; idx <= model->nrows; idx++) {
//							numChanged += examine_example(idx);
//						}
//						std::cout << "b_up - b_low = " << settings.b_up - settings.b_low << " | " << "Changed = " << numChanged << std::endl;
//					}
//					else {
//						/* Default Method SKTWO */
//						// check the worse pair	
//						numChanged = 1;
//						while (!(settings.b_up > -settings.tol && settings.b_low < settings.tol) && numChanged > 0.5) {
//							if (!take_step(settings.i_up, settings.i_low)) {
//								std::cout << "I_UP = " << settings.i_up << " | " << "I_LOW = " << settings.i_low << " Failed to update" << std::endl;
//								numChanged = 0;
//							}
//						}
//						numChanged = 0;
//						std::cout << "b_up - b_low = " << settings.b_up - settings.b_low << " | " << "Cache Size = " << model->io_cache.alphas.size() << std::endl;
//					}
//
//					if (examineAll) examineAll = false;
//					else if (numChanged == 0) examineAll = true;
//				}
//
//			}
//
//
//
//
//
//		}
//	}
//
//	Model* model;
//};

//			auto set_name = [&](double& a, double& b) {
//
//				double a_ = a;
//				double b_ = b;
//
//				if ((a_ * b_) != 0)
//				{
//					printf("\r\nFatal Error: alpha or VC in takeStep %f %f \r\n", a, b);
//					a = 0; b = 0;
//					return SetName::I1;
//				}
//
//				if (a_ > settings.vc)
//				{
//					printf("\r\nFatal Error: alpha or VC in takeStep %f %f \r\n", a, b);
//					a = settings.vc;
//					a_ = settings.vc;
//					return SetName::I3;
//				}
//
//				if (b_ > settings.vc)
//				{
//					printf("\r\nFatal Error: alpha or VC in takeStep %f %f \r\n", a, b);
//					b = settings.vc;
//					b_ = settings.vc;
//					return SetName::I2;
//				}
//
//
//				if (settings.vc == a_ && 0 == b_)
//					return SetName::I3;
//				else if (settings.vc == b_ && 0 == a_)
//					return SetName::I2;
//				else if (a_ > 0 && a_ < settings.vc && 0 == b_)
//					return SetName::IA;
//				else if (b_ > 0 && b_ < settings.vc && 0 == a_)
//					return SetName::IB;
//				else if (0 == a_ && 0 == b_)
//					return SetName::I1;
//				else
//				{
//					printf("\r\nFATAL ERROR : wrong alpha or VC in GetName. %f %f \r\n", a, b);
//					a = 0; b = 0;
//					return SetName::I1;
//				}
//
//
//			};
//			auto calculate_fi = [&](const unsigned int& widx) {
//				/* setandfi.cpp */
//				double Fi = 0;
//				for (unsigned j = 0; j < model->nrows; ++j) {
//					if (ALPHAUP(j) != 0 || ALPHADW(j) != 0)
//						Fi += (ALPHAUP(j) - ALPHADW(j)) * model->kernel(j, widx);
//				}
//				//for (unsigned i = 0; i < model->nrows; ++i) {
//				//	aj = model->alpha + i;
//				//	if (aj->alpha_up != 0 || aj->alpha_dw != 0)
//				//		Fi = Fi + (aj->alpha_up - aj->alpha_dw) * model->kernel(i, idx);
//				//}
//				Fi = model->outputs(widx) - Fi;
//				return Fi;
//			};



/* OLD CODE */
//void free_cache_list() {
//	CacheNode* temp = NULL;
//	while (io_cache->front != NULL)
//	{
//		temp = io_cache->front;
//		io_cache->front = temp->next;
//		io_cache->count--;
//		temp->alpha->cache = NULL;
//		delete temp;
//	}
//	io_cache->front = NULL;
//	io_cache->rear = NULL;
//}
//void free_alphas() {
//	for (unsigned int i = 0; i < nrows; ++i) {
//		alpha.get()[i].alpha_up = 0;
//		alpha.get()[i].alpha_dw = 0;
//		alpha.get()[i].f_cache = 0;
//		alpha.get()[i].kernel = kernel(i, i);
//		alpha.get()[i].setname = SetName::I1;
//		alpha.get()[i].cache = NULL;
//	}
//}
//void add_cache_node(Alphas* alpha_) {
//	CacheNode* node = new CacheNode;
//	node->alpha = alpha_;
//	node->previous = NULL;
//	node->next = NULL;
//	if (io_cache->front == 0)
//		io_cache->front = node;
//	else
//	{
//		node->next = io_cache->front;
//		io_cache->front->previous = node;
//		io_cache->front = node;
//	}
//	io_cache->count++;
//	// add pointer into alpha list
//	alpha_->cache = node;
//}
//void free_cache_node(Alphas* alpha_) {
//	CacheNode* node = alpha_->cache;
//	if (node->previous != NULL) {
//		node->previous->next = node->next;
//		if (NULL != node->next) // not at rear
//			node->next->previous = node->previous;
//		else					// at rear but not front
//		{
//			node->previous->next = NULL;
//			io_cache->rear = node->previous;
//		}
//	}
//	else if (node->next != NULL) {
//		io_cache->front = node->next;
//		node->next->previous = NULL;
//	}
//	else {
//		io_cache->front = NULL;
//		io_cache->rear = NULL;
//	}
//	io_cache->count--;
//	alpha->cache = NULL;
//	delete node;
//}