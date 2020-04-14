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
#include "qp_solver.hpp"

qp_solver::qp_solver(model_size& size)
{
    nu = size.nu;
    nbx = size.nbx;
    nbg = size.nbg;
    nbgN = size.nbgN;
    N = size.N;

    fc = new full_condensing(size);
    myQP = new SQProblem(N*nu, N*nbx+N*nbg+nbgN);
    myOptions = new Options();
    myOptions->setToMPC();
    myOptions->printLevel = PL_NONE;
    myQP->setOptions(*myOptions);
}

void qp_solver::solveQP(qp_problem& qp, const VectorXd& x0, uint64_t& sample)
{
    fc->condense(qp, x0);

    int nWSR = 50;

    if (sample == 0)
        myQP->init(fc->Hc.data(), fc->gc.data(), fc->Cc.data(), qp.data.lb_u.data(), qp.data.ub_u.data(), fc->lcc.data(), fc->ucc.data(), nWSR, 0);
    else
        myQP->hotstart(fc->Hc.data(), fc->gc.data(), fc->Cc.data(), qp.data.lb_u.data(), qp.data.ub_u.data(), fc->lcc.data(), fc->ucc.data(), nWSR, 0);

    double yOpt[N*nu+N*nbx+N*nbg+nbgN];
    myQP->getPrimalSolution(qp.out.du.data());
    myQP->getDualSolution(yOpt);
    memcpy(qp.out.mu_u.data(), yOpt, N*nu*sizeof(double));
    memcpy(qp.out.mu_g.data(), yOpt+N*nu, (N*nbg+nbgN)*sizeof(double));
    memcpy(qp.out.mu_x.data(), yOpt+N*nu+N*nbg+nbgN, N*nbx*sizeof(double));
}

qp_solver::~qp_solver()
{
    delete myQP;
    delete myOptions;
    delete fc;
}
