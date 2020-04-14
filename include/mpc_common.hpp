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
#ifndef H_CPPMPC_COMMON
#define H_CPPMPC_COMMON

struct model_size {
    int nx;        // No. of states
    int nu;        // No. of controls
    int ny;        // No. of cost terms
    int nyN;       // No. of cost terms in the terminal stage
    int np;        // No. of on-line parameters
    int nbx;       // No. of state box constraints
    int nbu;       // No. of control box constraints
    int nbg;       // No. of general constraints
    int nbgN;      // No. of general constraints in the terminal stage
    int N;         // No. of shooting points
    int* nbx_idx;  // index of state box constraints
    int* nbu_idx;  // index of control box constraints

    void set_nx(const int n) {this->nx = n;}
    void set_nu(const int n) {this->nu = n;}
    void set_ny(const int n) {this->ny = n;}
    void set_nyN(const int n) {this->nyN = n;}
    void set_np(const int n) {this->np = n;}
    void set_nbg(const int n) {this->nbg = n;}
    void set_nbgN(const int n) {this->nbgN = n;}
    void set_N(const int n) {this->N = n;}
    void set_nbx(const int n, const int* a)
    {
        this->nbx = n;
        this->nbx_idx = new int[n];
        for (int i=0; i<n; i++)
            this->nbx_idx[i] = a[i];
    }
    void set_nbu(const int n, const int* a)
    {
        this->nbu = n;
        this->nbu_idx = new int[n];
        for (int i=0; i<n; i++)
            this->nbu_idx[i] = a[i];
    }

    ~model_size() {
        delete [] this->nbx_idx;
        delete [] this->nbu_idx;
    }
};

void regularization(int n, double* A, double reg);

#endif  /* H_CPPMPC_COMMON */
