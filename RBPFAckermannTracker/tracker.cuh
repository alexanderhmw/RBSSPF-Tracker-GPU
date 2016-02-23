#ifndef TRACKER_CUH
#define TRACKER_CUH

#include<cuda.h>
#include<cuda_runtime.h>
#include<thrust/random/linear_congruential_engine.h>
#include<thrust/random/uniform_real_distribution.h>
#include<thrust/random/normal_distribution.h>
#include<thrust/generate.h>

#include<chrono>
#include<vector>
#include<random>
#include<time.h>
#include<algorithm>

//#define TCINFO

#define PI 3.14159265359
#define DEG2RAD(ang) (ang*PI/180)

#define MINSIGMA 1e-2
#define UNCERTAINTHRESH 0.4
#define MAXSIGMA 1e6

#define RQPN 2048
#define SPN 4
#define MAXPN (SPN*RQPN)
#define MAXBEAMNUM 2048

#define CUDAFREE(pointer) if(pointer!=NULL){cudaFree(pointer);pointer=NULL;}

#define THREAD_1D 256
#define THREAD_2D 16
#define GetKernelDim_1D(numBlocks, threadsPerBlock, dim) int numBlocks=(dim+THREAD_1D-1)/THREAD_1D; int threadsPerBlock=THREAD_1D;
#define GetKernelDim_2D(numBlocks, threadsPerBlock, xdim, ydim) dim3 numBlocks(int((xdim+THREAD_2D-1)/THREAD_2D), int((ydim+THREAD_2D-1)/THREAD_2D)); dim3 threadsPerBlock(THREAD_2D, THREAD_2D);
#define GetThreadID_1D(id) int id=blockDim.x*blockIdx.x+threadIdx.x;
#define GetThreadID_2D(xid,yid) int xid=blockDim.x*blockIdx.x+threadIdx.x;int yid=blockDim.y*blockIdx.y+threadIdx.y;

#define NEARESTRING 3.35
#define MINBEAM 2
#define MAXBEAM 100
#define MARGIN0 0.1
#define MARGIN1 0.1
#define MARGIN2 0.1

#define SIGMA 0.5
#define COST0 1
#define WEIGHT0 -2
#define COST1 2
#define WEIGHT1 -8
#define COST2 0
#define WEIGHT2 0
#define COST3 1.6
#define WEIGHT3 -5.12

#define SSPFFLAG 1
#define REJECTFLAG 0

struct ObjectState
{
    double weight;
    double x,y,theta;
    double dx,dy,dtheta;
    double wl,wr,lf,lb;
    double dwl,dwr,dlf,dlb;
    double a,v,k,omega;
    double da,dv,dk,domega;
    double count;

    double cx[4],cy[4],cl[4];
    int bid[4];
    int eid[2];

    double ox,oy;
};

struct StateConstrain
{
    double thetamin,thetamax;
    double wlmin=0,wlmax=3;
    double wrmin=0,wrmax=3;
    double lfmin=0,lfmax=5;
    double lbmin=0,lbmax=5;
    double amin=DEG2RAD(-60),amax=DEG2RAD(60);
    double vmin=-10,vmax=30;
    double kmin=-0.5,kmax=0.5;
    double omegamin=DEG2RAD(-90),omegamax=DEG2RAD(90);
};

struct LaserScan
{
    double timestamp;
    double x,y,theta;
    int beamnum;
    double length[MAXBEAMNUM];
};

struct EgoMotion
{
    bool validflag=0;
    double x,y,theta;
    double timestamp;
    double dx=0,dy=0,dtheta=0;
    double dt=0;
    double density;
    bool motionflag=0;
};

struct ObjectStateOffset
{
    double thetaoff=DEG2RAD(30),thetaprec=DEG2RAD(1),thetazoom=1;
    double wloff=1.5,wlprec=0.1,wlzoom=1;
    double wroff=1.5,wrprec=0.1,wrzoom=1;
    double lfoff=2.5,lfprec=0.1,lfzoom=1;
    double lboff=2.5,lbprec=0.1,lbzoom=1;
    double aoff=DEG2RAD(60),aprec=DEG2RAD(1),azoom=1;
    double voff=20,vprec=1,vzoom=1;
    double koff=0.6,kprec=0.001,kzoom=1;
    double omegaoff=DEG2RAD(90),omegaprec=DEG2RAD(1),omegazoom=1;
    double anneal=1;
    double annealratio=1;
};

extern "C" void startTracker();
extern "C" void stopTracker();
extern "C" void setLaserScan(LaserScan & scan);
extern "C" void initTracker(ObjectState * particles, int & pnum, ObjectState & average, int & beamnum, int * beamid);
extern "C" void initMotion(ObjectState * particles, int & pnum, ObjectState & average, int & beamnum, int * beamid);
extern "C" void updateTracker(ObjectState * particles, int & pnum, ObjectState & average, bool & pfflag, int & beamnum, int * beamid);


#endif // TRACKER_CUH

