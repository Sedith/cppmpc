#include <iostream>
#include <fstream>
#include "mpc_common.hpp"
#include "qp_problem.hpp"
#include "full_condensing.hpp"
#include "qp_solver.hpp"
#include "Timer.hpp"
#include "casadi_wrapper.hpp"
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

#define COUCOU(i) cout<<"coucou "<<i<<endl;

struct  mpc_workspace_s {
    qp_problem* qp;     // qp problem
    qp_solver*  solver; // encapsulating solver & full condenser
    Timer       timer;  // start/stop timer
    mpc_workspace_s(model_size& size) {
        this->solver = new qp_solver(size);
        this->qp = new qp_problem(size);
        this->qp->in.reg = 1e-8;
    }
};

int main()
{
    // define problem size
    model_size size;
    size.nx = 16;
    size.nu = 4;
    size.ny = 18;
    size.nyN = 18;
    size.np = 0;
    size.nbu = 4;
    size.nbx = 4;
    size.nbg = 0;
    size.nbgN = 0;
    size.N = 10;
    size.nbx_idx = new int[size.nbx];
    size.nbx_idx[0] = 12;
    size.nbx_idx[1] = 13;
    size.nbx_idx[2] = 14;
    size.nbx_idx[3] = 13;
    size.nbu_idx = new int[size.nbu];
    size.nbu_idx[0] = 0;
    size.nbu_idx[1] = 1;
    size.nbu_idx[2] = 2;
    size.nbu_idx[3] = 3;

    // Workspace
    mpc_workspace_s* ws = new mpc_workspace_s(size);
    ArrayXd x0 = ArrayXd::Zero(size.nx);
    uint64_t sample = 0;
    double OBJ;
    double CPT;

    // initial condition and parameters
    double hover_force = 1.230 * 9.81 / 4;
    double c_f = 6.5e-4;
    x0(2) = 1;  // z = 1
    x0(12) = hover_force;
    x0(13) = hover_force;
    x0(14) = hover_force;
    x0(15) = hover_force;
    for(int i=1;i<size.N+1;i++)
        ws->qp->in.x.col(i) = x0;

    ws->qp->in.y(2,0) = 1.3;
    for(int i=1;i<size.N;i++)
        ws->qp->in.y.col(i) = ws->qp->in.y.col(0);
    ws->qp->in.yN = ws->qp->in.y.col(0);

    ws->qp->in.W(0,0) = 30;
    ws->qp->in.W(1,0) = 30;
    ws->qp->in.W(2,0) = 10;
    ws->qp->in.W(3,0) = 1;
    ws->qp->in.W(4,0) = 1;
    ws->qp->in.W(5,0) = 1;
    ws->qp->in.W(6,0) = 1;
    ws->qp->in.W(7,0) = 1;
    ws->qp->in.W(8,0) = 1;
    ws->qp->in.W(9,0) = 1;
    ws->qp->in.W(10,0) = 1;
    ws->qp->in.W(11,0) = 1;
    ws->qp->in.W(12,0) = 1e-4;
    ws->qp->in.W(13,0) = 1e-4;
    ws->qp->in.W(14,0) = 1e-4;
    ws->qp->in.W(15,0) = 1e-4;
    ws->qp->in.W(16,0) = 1e-4;
    ws->qp->in.W(17,0) = 1e-4;
    for(int i=1;i<size.N;i++)
        ws->qp->in.W.col(i) = ws->qp->in.W.col(0);
    ws->qp->in.WN = ws->qp->in.W.col(0);

    for(int i=1;i<size.nbx;i++)
    {
        ws->qp->in.lbx(i) = c_f * pow(16,2);
        ws->qp->in.ubx(i) = c_f * pow(100,2);
    }
    for(int i=1;i<size.nbu;i++)
    {
        ws->qp->in.lbu(i) = -100;
        ws->qp->in.ubu(i) = 100;
    }

    ws->qp->in.reg = 1E-8;

    // prepare the closed-loop simulation
    double Tf=10, Ts=0.01, t=0;

    double *simu_in[3];
    double *simu_out[1];
    simu_in[2] = ws->qp->in.p.col(0).data();

    ofstream myfile;
    myfile.open ("data.txt");

    ws->qp->info(OBJ);
    myfile <<"Sample 0: " << x0.transpose() << " | - |OBJ=: "<<OBJ<<endl;
    // start the simulation
    while (t < Tf)
    {
        // call RTI solving routine
        ws->timer.start();
        ws->qp->generateQP();
        ws->solver->solveQP(*(ws->qp), x0, sample);
        ws->qp->expandSol(x0);
        ws->qp->info(OBJ);
        CPT = ws->timer.getElapsedTimeInMilliSec();
        sample++;

        // simulate feedback
        simu_in[0] = ws->qp->in.x.col(0).data();
        simu_in[1] = ws->qp->in.u.col(0).data();
        simu_out[0] = x0.data();
        F_Fun(simu_in, simu_out);

        // update time
        t += Ts;

        // store the closed-loop results
        myfile <<"Sample " << sample <<": " << x0.array().segment(0,3).transpose() << " " << x0.array().segment(12,4).transpose() << " | " << ws->qp->in.u.col(0).transpose() << " |OBJ=: "<<OBJ <<" |CPT=: " << CPT << "ms" <<endl;

        // shifting(optional)
        for (int i=0; i<size.N-1; i++)
        {
            ws->qp->in.x.col(i) = ws->qp->in.x.col(i+1);
            ws->qp->in.u.col(i) = ws->qp->in.u.col(i+1);
        }
        ws->qp->in.x.col(size.N-1) = ws->qp->in.x.col(size.N);
    }

    return 0;
}
