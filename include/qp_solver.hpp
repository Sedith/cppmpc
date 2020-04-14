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
#ifndef H_CPPMPC_QP_SOLVER
#define H_CPPMPC_QP_SOLVER

#include "mpc_common.hpp"
#include "qp_problem.hpp"
#include "full_condensing.hpp"

#include <qpOASES.hpp>

using namespace qpOASES;

class qp_solver{
    private:
        int nu;
        int nbx;
        int nbg;
        int nbgN;
        int N;

        SQProblem*  myQP;
        Options*    myOptions;

        full_condensing* fc;

    public:
        qp_solver(model_size& size);
        ~qp_solver();

        void generate();
        void solveQP(qp_problem& qp, const VectorXd& x0, uint64_t& sample);
        void compute_obj(double& OBJ);
};

#endif  /* H_CPPMPC_QP_SOLVER */
