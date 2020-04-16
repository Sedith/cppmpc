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
#include "full_condensing.hpp"
#include "constants.hpp"

full_condensing::full_condensing(uint16_t& N)
{
    this->N = N;
    Hc = Matrix<double, Dynamic, Dynamic, RowMajor>::Zero(N*nu,N*nu);
    Cc = Matrix<double, Dynamic, Dynamic, RowMajor>::Zero(N*nbx+N*nbg+nbgN,N*nu);
    gc = VectorXd::Zero(N*nu);
    lcc = VectorXd::Zero(N*nbx+N*nbg+nbgN);
    ucc = VectorXd::Zero(N*nbx+N*nbg+nbgN);

    G = MatrixXd::Zero(N*nx,N*nu);
    L = VectorXd::Zero((N+1)*nx);
    W = MatrixXd::Zero(nx,nu);
    w = VectorXd::Zero(nx);
}

void full_condensing::condense(qp_problem& qp, const VectorXd& x0)
{
    int idx = N*nbg+nbgN;

    int i, j;

    /* compute G */
    for (i=0; i<N; i++)
    {
        G.block(i*nx,i*nu,nx,nu) = qp.data.B.block(0,i*nu,nx,nu);
        for (j=i+1; j<N; j++)
            G.block(j*nx,i*nu,nx,nu) = qp.data.A.block(0,j*nx,nx,nx) * G.block((j-1)*nx,i*nu,nx,nu);
    }

    /* Compute Hc */
    for (i=0; i<N; i++)
    {
        W = qp.data.Q.block(0,N*nx,nx,nx) * G.block((N-1)*nx,i*nu,nx,nu);
        for (j=N-1; j>i; j--)
        {
            Hc.block(j*nu,i*nu,nu,nu) = qp.data.S.block(0,j*nu,nx,nu).transpose() * G.block((j-1)*nx,i*nu,nx,nu) + qp.data.B.block(0,j*nu,nx,nu).transpose() * W;
            W = qp.data.Q.block(0,j*nx,nx,nx) * G.block((j-1)*nx,i*nu,nx,nu) + qp.data.A.block(0,j*nx,nx,nx).transpose() * W;
            Hc.block(i*nu,j*nu,nu,nu) = Hc.block(j*nu,i*nu,nu,nu).transpose();
        }
        Hc.block(i*nu,i*nu,nu,nu) = qp.data.R.block(0,i*nu,nu,nu) + qp.data.B.block(0,i*nu,nx,nu).transpose() * W;
    }

    /* Compute Ccg */
    if (nbg > 0)
        for (i=0; i<N; i++)
        {
            Cc.block(i*nbg,i*nu,nbg,nu) = qp.data.Cgu.block(0,i*nu,nbg,nu);
            for (j=i+1; j<N; j++)
                Cc.block(j*nbg,i*nu,nbg,nu) = qp.data.Cgx.block(0,j*nx,nbg,nx)*G.block((j-1)*nx,i*nu,nx,nu);
        }

    /* Compute CcN */
    if (nbgN > 0)
        for (i=0; i<N; i++)
        {
            Cc.block(N*nbg,i*nu,nbgN,nu) = qp.data.CgN * G.block((N-1)*nx,i*nu,nx,nu);
        }

    /* Compute Ccx */
    if (nbx > 0)
        for (i=0; i<N; i++)
        {
            for(j=i+1;j<=N;j++)
                Cc.block(idx+(j-1)*nbx,i*nu,nbx,nu) = qp.data.Cx * G.block((j-1)*nx,i*nu,nx,nu);

        }

    /* compute L */
    L.head(nx) = x0 - qp.in.x.col(0);
    for (i=0; i<N; i++)
    {
        L.segment((i+1)*nx,nx) = qp.data.A.block(0,i*nx,nx,nx)*L.segment(i*nx,nx) + qp.data.a.col(i);
    }

    /* compute gc */
    w = qp.data.gx.col(N) + qp.data.Q.block(0,N*nx,nx,nx) * L.tail(nx);
    for (i=N-1; i>0; i--)
    {
        gc.segment(i*nu,nu) = qp.data.gu.col(i) + qp.data.S.block(0,i*nu,nx,nu).transpose()*L.segment(i*nx,nx) + qp.data.B.block(0,i*nu,nx,nu).transpose()*w;
        w = qp.data.gx.col(i) + qp.data.Q.block(0,i*nx,nx,nx)*L.segment(i*nx,nx) + qp.data.A.block(0,i*nx,nx,nx).transpose()*w;
    }
    gc.head(nu) = qp.data.gu.col(0) + qp.data.S.leftCols(nu).transpose()*L.head(nx) + qp.data.B.leftCols(nu).transpose()*w;

    /* Compute cc */
    if (nbg > 0)
        for (i=0; i<N; i++)
        {
            lcc.segment(i*nbg,nbg) = qp.data.Cgx.block(0,i*nx,nbg,nx)*L.segment(i*nx,nx);
            ucc.segment(i*nbg,nbg) = qp.data.ub_g.segment(i*nbg,nbg) - lcc.segment(i*nbg,nbg);
            lcc.segment(i*nbg,nbg) = qp.data.lb_g.segment(i*nbg,nbg) - lcc.segment(i*nbg,nbg);
        }

    /* Compute ccN */
    if (nbgN > 0)
    {
        lcc.segment(N*nbg,nbgN) = qp.data.CgN * L.tail(nx);
        ucc.segment(N*nbg,nbgN) = qp.data.ub_g.segment(N*nbg,nbgN) - lcc.segment(N*nbg,nbgN);
        lcc.segment(N*nbg,nbgN) = qp.data.lb_g.segment(N*nbg,nbgN) - lcc.segment(N*nbg,nbgN);
    }

    if (nbx > 0)
        for (i=0; i<N; i++)
        {
            lcc.segment(idx+i*nbx,nbx) = qp.data.Cx * L.segment((i+1)*nx,nx);
            ucc.segment(idx+i*nbx,nbx) = qp.data.ub_x.segment(i*nbx,nbx) - lcc.segment(idx+i*nbx,nbx);
            lcc.segment(idx+i*nbx,nbx) = qp.data.lb_x.segment(i*nbx,nbx) - lcc.segment(idx+i*nbx,nbx);
        }
}
