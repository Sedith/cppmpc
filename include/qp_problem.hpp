/*
 * Copyright (c) 2020 LAAS/CNRS
 * All rights reserved.
 *
 * Redistribution  and  use  in  source  and binary  forms,  with  or  without
 * modification, are permitted provided that the following conditions are met:
 *
 *   1. Redistributions of  source  code must retain the  above copyright
 *      notice and this list of conditions.
 *   2. Redistributions in binary form must reproduce the above copyright
 *      notice and  this list of  conditions in the  documentation and/or
 *      other materials provided with the distribution.
 *
 * THE SOFTWARE  IS PROVIDED "AS IS"  AND THE AUTHOR  DISCLAIMS ALL WARRANTIES
 * WITH  REGARD   TO  THIS  SOFTWARE  INCLUDING  ALL   IMPLIED  WARRANTIES  OF
 * MERCHANTABILITY AND  FITNESS.  IN NO EVENT  SHALL THE AUTHOR  BE LIABLE FOR
 * ANY  SPECIAL, DIRECT,  INDIRECT, OR  CONSEQUENTIAL DAMAGES  OR  ANY DAMAGES
 * WHATSOEVER  RESULTING FROM  LOSS OF  USE, DATA  OR PROFITS,  WHETHER  IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR  OTHER TORTIOUS ACTION, ARISING OUT OF OR
 * IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 *
 *                                                 Martin Jacquet - March 2020
 *                                               Based on CPPMPC by Yutao Chen
 */
#ifndef H_CPPMPC_QP_PROBLEM
#define H_CPPMPC_QP_PROBLEM

#include "mpc_common.hpp"

#include <eigen3/Eigen/Dense>

using namespace Eigen;

class qp_in {
    public:
        MatrixXd x;
        MatrixXd u;
        MatrixXd y;
        VectorXd yN;
        MatrixXd W;
        MatrixXd WN;
        MatrixXd p;
        VectorXd lbx;
        VectorXd ubx;
        VectorXd lbu;
        VectorXd ubu;
        VectorXd lbg;
        VectorXd ubg;
        VectorXd lbgN;
        VectorXd ubgN;
        double reg;

        void init(model_size& size);
};

class qp_data {
    public:
        MatrixXd Q;
        MatrixXd S;
        MatrixXd R;
        MatrixXd A;
        MatrixXd B;
        MatrixXd a;
        MatrixXd gx;
        MatrixXd gu;
        MatrixXd Cx;
        MatrixXd Cgx;
        MatrixXd CgN;
        MatrixXd Cgu;
        VectorXd lb_x;
        VectorXd ub_x;
        VectorXd lb_u;
        VectorXd ub_u;
        VectorXd lb_g;
        VectorXd ub_g;

        void init(model_size& size);
};

class qp_out {
    public:
        MatrixXd dx;
        MatrixXd du;
        MatrixXd lam;
        VectorXd mu_u;
        VectorXd mu_x;
        VectorXd mu_g;

        void init(model_size& size);
};

class qp_problem {
    public:
        qp_in in;
        qp_data data;
        qp_out out;
        model_size size;

        qp_problem(model_size& s);
        void generateQP();
        void expandSol(const VectorXd& x0);
        void info(double& OBJ);
};

#endif  /* H_CPPMPC_QP_PROBLEM */
