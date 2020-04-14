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
#ifndef H_CPPMPC_CASADI_WRAPPER
#define H_CPPMPC_CASADI_WRAPPER

void f_Fun(double** in, double** out);
void vde_Fun(double** in, double** out);
void impl_f_Fun(double** in, double** out);
void impl_jac_x_Fun(double** in, double** out);
void impl_jac_u_Fun(double** in, double** out);
void impl_jac_xdot_Fun(double** in, double** out);
void F_Fun(double** in, double** out);
void D_Fun(double** in, double** out);
void Hi_Fun(double** in, double** out);
void HN_Fun(double** in, double** out);
void gi_Fun(double** in, double** out);
void gN_Fun(double** in, double** out);
void Ci_Fun(double** in, double** out);
void CN_Fun(double** in, double** out);
void path_con_Fun(double** in, double** out);
void path_con_N_Fun(double** in, double** out);
void adj_Fun(double** in, double** out);
void adjN_Fun(double** in, double** out);
void obji_Fun(double** in, double** out);
void objN_Fun(double** in, double** out);

#endif  /* H_CPPMPC_CASADI_WRAPPER */
