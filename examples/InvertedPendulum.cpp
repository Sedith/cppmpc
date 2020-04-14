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
    size.nx = 4;
    size.nu = 1;
    size.ny = 5;
    size.nyN = 4;
    size.np = 0;
    size.nbu = 1;
    size.nbx = 1;
    size.nbg = 0;
    size.nbgN = 0;
    size.N = 40;
    size.nbx_idx = new int[size.nbx];
    size.nbx_idx[0] = 0;
    size.nbu_idx = new int[size.nbu];
    size.nbu_idx[0] = 0;

    // Workspace
    mpc_workspace_s* ws = new mpc_workspace_s(size);
    ArrayXd x0 = ArrayXd::Zero(4);
    uint64_t sample = 0;
    double OBJ;
    double CPT;

    // initial condition and parameters
    x0(1) = M_PI;

    for(int i=0;i<size.N+1;i++)
        ws->qp->in.x(1,i) = M_PI;

    ws->qp->in.W(0,0) = 10;
    ws->qp->in.W(1,0) = 10;
    ws->qp->in.W(2,0) = 0.1;
    ws->qp->in.W(3,0) = 0.1;
    ws->qp->in.W(4,0) = 0.01;
    for(int i=1;i<size.N;i++)
        ws->qp->in.W.col(i) = ws->qp->in.W.col(0);

    ws->qp->in.WN(0) = 10;
    ws->qp->in.WN(1) = 10;
    ws->qp->in.WN(2) = 0.1;
    ws->qp->in.WN(3) = 0.1;

    ws->qp->in.lbu(0) = -20;
    ws->qp->in.ubu(0) = 20;
    ws->qp->in.lbx(0) = -2;
    ws->qp->in.ubx(0) = 2;

    ws->qp->in.reg = 1E-8;

    // prepare the closed-loop simulation
    double Tf=10, Ts=0.01, t=0;

    double *simu_in[3];
    double *simu_out[1];
    simu_in[2] = ws->qp->in.p.col(0).data();

    ofstream myfile;
    myfile.open ("data.txt");

    ws->qp->info(OBJ);
    myfile <<"Sample 0: " << x0.transpose() << " | - |OBJ=: "<<OBJ <<" |CPT=: " << "" <<endl;
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
        myfile <<"Sample " << sample <<": " << x0.transpose() << " | " << ws->qp->in.u.col(0) << " |OBJ=: "<<OBJ <<" |CPT=: " << CPT << "ms" <<endl;

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
