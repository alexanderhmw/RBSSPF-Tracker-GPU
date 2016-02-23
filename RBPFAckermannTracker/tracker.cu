#include"tracker.cuh"

using namespace std::chrono;

LaserScan * d_scan=NULL;
LaserScan h_scan;
EgoMotion h_egomotion;

ObjectState * d_particle=NULL;
ObjectState h_particle[RQPN];
ObjectState * d_tmpparticle=NULL;
ObjectState h_tmpparticle[MAXPN];
bool h_flag[MAXPN];

int h_seed[MAXPN];
thrust::minstd_rand * d_rng=NULL;

#define PI 3.14159265359

//==============================================================================

__host__ __device__
void deviceBuildModel(ObjectState & state, double & density)
{
    double c=cos(state.theta);
    double s=sin(state.theta);

    state.ox=-c*state.x-s*state.y;
    state.oy=s*state.x-c*state.y;

    state.cx[0]=c*state.lf-s*state.wl+state.x; state.cy[0]=s*state.lf+c*state.wl+state.y;
    state.cx[1]=c*state.lf+s*state.wr+state.x; state.cy[1]=s*state.lf-c*state.wr+state.y;
    state.cx[2]=-c*state.lb+s*state.wr+state.x; state.cy[2]=-s*state.lb-c*state.wr+state.y;
    state.cx[3]=-c*state.lb-s*state.wl+state.x; state.cy[3]=-s*state.lb+c*state.wl+state.y;

    state.cl[0]=state.cl[2]=state.wl+state.wr;
    state.cl[1]=state.cl[3]=state.lf+state.lb;

    state.bid[0]=(atan2(state.cy[0],state.cx[0])+PI)/density;
    state.bid[1]=(atan2(state.cy[1],state.cx[1])+PI)/density;
    state.bid[2]=(atan2(state.cy[2],state.cx[2])+PI)/density;
    state.bid[3]=(atan2(state.cy[3],state.cx[3])+PI)/density;

    if(state.ox>state.lf)
    {
        if(state.oy>state.wl)
        {
            state.eid[0]=0;state.eid[1]=3;
        }
        else if(state.oy<-state.wr)
        {
            state.eid[0]=0;state.eid[1]=1;
        }
        else
        {
            state.eid[0]=0;state.eid[1]=-1;
        }
    }
    else if(state.ox<-state.lb)
    {
        if(state.oy>state.wl)
        {
            state.eid[0]=2;state.eid[1]=3;
        }
        else if(state.oy<-state.wr)
        {
            state.eid[0]=2;state.eid[1]=1;
        }
        else
        {
            state.eid[0]=2;state.eid[1]=-1;
        }
    }
    else
    {
        if(state.oy>state.wl)
        {
            state.eid[0]=3;state.eid[1]=-1;
        }
        else if(state.oy<-state.wr)
        {
            state.eid[0]=1;state.eid[1]=-1;
        }
        else
        {
            state.eid[0]=-1;state.eid[1]=-1;
        }
    }
    return;
}

__host__ __device__
void deviceMeasureEdge(ObjectState & state, int edgeid, LaserScan * scan, double anneal, int * beamnum, int * beamid, bool uncertainflag)
{
    if(state.eid[edgeid]<0)
    {
        return;
    }

    if(uncertainflag)
    {
        switch(state.eid[edgeid])
        {
        case 0:
            if(state.dlf>UNCERTAINTHRESH)
            {
                return;
            }
            break;
        case 1:
            if(state.dwr>UNCERTAINTHRESH)
            {
                return;
            }
            break;
        case 2:
            if(state.dlb>UNCERTAINTHRESH)
            {
                return;
            }
            break;
        case 3:
            if(state.dwl>UNCERTAINTHRESH)
            {
                return;
            }
            break;
        default:
            break;
        }
    }

    int starteid=state.eid[edgeid];
    int endeid=(state.eid[edgeid]+1)%4;

    int startbid=state.bid[starteid];
    int endbid=state.bid[endeid];
    if(startbid>endbid)
    {
        endbid+=scan->beamnum;
    }

    int totalbeam=(endbid-startbid)+1;
    if(totalbeam<=3)
    {
        state.eid[edgeid]=-1;
    }

    double dx1=state.cx[endeid]-state.cx[starteid];
    double dy1=state.cy[endeid]-state.cy[starteid];
    double dx2=-dy1/state.cl[starteid];
    double dy2=dx1/state.cl[starteid];

    double density=2*PI/scan->beamnum;
//    int midbid=(startbid+endbid)/2;
    for(int i=startbid;i<=endbid;i++)
    {
        double P[4]={MAXBEAM,MAXBEAM,MAXBEAM,MAXBEAM};
        int tmpid=i%scan->beamnum;
//        double weightsigma=abs(i-midbid)*2.0/totalbeam*0.09+0.01;
        double bear=tmpid*density-PI;
        double c=cos(bear);
        double s=sin(bear);
        double tmpx=c*dx1+s*dy1;
        double tmpy=s*dx1-c*dy1;
        if(tmpy!=0)
        {
            double beta=tmpx/tmpy*(c*state.cy[starteid]-s*state.cx[starteid])+(c*state.cx[starteid]+s*state.cy[starteid]);
            if(beta>=MINBEAM&&beta<=MAXBEAM)
            {
                P[2]=beta;
                double gamma0,gamma1,gamma2;
                if(beta<NEARESTRING)
                {
                    gamma0=fabs(beta-(tmpx/tmpy*(c*(state.cy[starteid]+dy2*beta)-s*(state.cx[starteid]+dx2*beta))+c*(state.cx[starteid]+dx2*beta)+s*(state.cy[starteid]+dy2*beta)));
                    gamma1=fabs(beta-(tmpx/tmpy*(c*(state.cy[starteid]+dy2*2)-s*(state.cx[starteid]+dx2*2))+c*(state.cx[starteid]+dx2*2)+s*(state.cy[starteid]+dy2*2)));
                    gamma2=fabs(beta-(tmpx/tmpy*(c*(state.cy[starteid]+dy2*beta)-s*(state.cx[starteid]+dx2*beta))+c*(state.cx[starteid]+dx2*beta)+s*(state.cy[starteid]+dy2*beta)));
                }
                else
                {
                    gamma0=fabs(beta-(tmpx/tmpy*(c*(state.cy[starteid]+dy2*MARGIN0)-s*(state.cx[starteid]+dx2*MARGIN0))+c*(state.cx[starteid]+dx2*MARGIN0)+s*(state.cy[starteid]+dy2*MARGIN0)));
                    gamma1=fabs(beta-(tmpx/tmpy*(c*(state.cy[starteid]+dy2*MARGIN1)-s*(state.cx[starteid]+dx2*MARGIN1))+c*(state.cx[starteid]+dx2*MARGIN1)+s*(state.cy[starteid]+dy2*MARGIN1)));
                    gamma2=fabs(beta-(tmpx/tmpy*(c*(state.cy[starteid]+dy2*MARGIN2)-s*(state.cx[starteid]+dx2*MARGIN2))+c*(state.cx[starteid]+dx2*MARGIN2)+s*(state.cy[starteid]+dy2*MARGIN2)));
                }
                P[1]=P[2]-gamma0>=MINBEAM?P[2]-gamma0:MINBEAM;
                P[3]=P[2]+gamma1<=MAXBEAM?P[2]+gamma1:MAXBEAM;
                P[0]=P[2]-gamma2>=MINBEAM?P[2]-gamma2:MINBEAM;
                double tmplogweight;
                if(scan->length[tmpid]<=P[0])
                {
                    tmplogweight=0;
//                    double delta=scan->length[tmpid]-P[0];
//                    double w1=WEIGHT0-WEIGHT0;
//                    double w2=WEIGHT1-WEIGHT0;
//                    tmplogweight=w1+(w2-w1)*exp(-delta*delta/0.01);
                }
                else if(scan->length[tmpid]<=P[1])
                {
                    tmplogweight=WEIGHT1-WEIGHT0;
//                    double delta=scan->length[tmpid]-P[1];
//                    double w1=WEIGHT1-WEIGHT0;
//                    double w2=WEIGHT2-WEIGHT0;
//                    tmplogweight=w1+(w2-w1)*exp(-delta*delta/0.01);
                }
                else if(scan->length[tmpid]<=P[3])
                {
                    if(beta>=NEARESTRING)
                    {
                        if(beamnum!=NULL&&beamid!=NULL&&totalbeam>3)
                        {
                            beamid[*beamnum]=tmpid;
                            (*beamnum)++;
                        }
                        state.count++;
                    }
//                    tmplogweight=WEIGHT2-WEIGHT0;
                    double delta=scan->length[tmpid]-P[2];
                    double w1=WEIGHT2-WEIGHT0;
                    double w2=2*w1;
                    tmplogweight=(w1+(w2-w1)*exp(-delta*delta/0.01));
                }
                else
                {
                    tmplogweight=WEIGHT3-WEIGHT0;
//                    double delta=scan->length[tmpid]-P[3];
//                    double w1=WEIGHT3-WEIGHT0;
//                    double w2=WEIGHT2-WEIGHT0;
//                    tmplogweight=w1+(w2-w1)*exp(-delta*delta/0.01);
                }
                state.weight+=tmplogweight/anneal;
            }
        }
    }
}

__host__ __device__
void deviceEgoMotion(ObjectState & state, EgoMotion & egomotion)
{
    double c=cos(egomotion.dtheta);
    double s=sin(egomotion.dtheta);
    double tmpx=c*state.x-s*state.y+egomotion.dx;
    double tmpy=s*state.x+c*state.y+egomotion.dy;
    state.x=tmpx;
    state.y=tmpy;
    state.theta+=egomotion.dtheta;
    return;
}

__host__ __device__
void deviceAckermannModel(ObjectState & state0, ObjectState & state1, EgoMotion & egomotion)
{
    state1=state0;
    if(state1.v==0)
    {
        state1.k=0;
        state1.a=0;
        deviceEgoMotion(state1,egomotion);
        return;
    }

    double c=cos(state1.theta);
    double s=sin(state1.theta);

    if(state1.k==0)
    {
        state1.x=state1.x+c*state1.v*egomotion.dt;
        state1.y=state1.y+s*state1.v*egomotion.dt;
        state1.a=0;
        deviceEgoMotion(state1,egomotion);
        return;
    }

    double c0=cos(state1.theta+state1.a);
    double s0=sin(state1.theta+state1.a);
    state1.omega=state1.v*state1.k;
    double dtheta=state1.omega*egomotion.dt;
    state1.theta+=dtheta;
    double c1=cos(state1.theta+state1.a);
    double s1=sin(state1.theta+state1.a);
    double R=1/state1.k;

    state1.x=state1.x+R*(-s0+s1);
    state1.y=state1.y+R*(c0-c1);
    deviceEgoMotion(state1,egomotion);
    return;
}

//==============================================================================

__global__
void kernelSetRandomSeed(int * seed, thrust::minstd_rand * rng, int tmppnum)
{
    GetThreadID_1D(id);
    if(id>=tmppnum)
    {
        return;
    }
    rng[id]=thrust::minstd_rand(seed[id]);
    return;
}

__global__
void kernelMeasureModel(LaserScan * scan, ObjectState * particle, ObjectState * tmpparticle, int tmppnum, thrust::minstd_rand * rng, ObjectStateOffset objectstateoffset, StateConstrain stateconstrain, EgoMotion egomotion)
{
    GetThreadID_1D(id);
    if(id>=tmppnum)
    {
        return;
    }
    int pid=id/SPN;

    tmpparticle[id]=particle[pid];

    if(objectstateoffset.thetaoff>objectstateoffset.thetaprec)
    {
        double thetamin=tmpparticle[id].theta-objectstateoffset.thetaoff;thetamin=thetamin>stateconstrain.thetamin?thetamin:stateconstrain.thetamin;
        double thetamax=tmpparticle[id].theta+objectstateoffset.thetaoff;thetamax=thetamax<stateconstrain.thetamax?thetamax:stateconstrain.thetamax;
        tmpparticle[id].theta=thrust::random::uniform_real_distribution<double>(thetamin,thetamax)(rng[id]);
    }

    double wlmin=tmpparticle[id].wl-objectstateoffset.wloff;wlmin=wlmin>stateconstrain.wlmin?wlmin:stateconstrain.wlmin;
    double wlmax=tmpparticle[id].wl+objectstateoffset.wloff;wlmax=wlmax<stateconstrain.wlmax?wlmax:stateconstrain.wlmax;
    tmpparticle[id].wl=thrust::random::uniform_real_distribution<double>(wlmin,wlmax)(rng[id]);

    double wrmin=tmpparticle[id].wr-objectstateoffset.wroff;wrmin=wrmin>stateconstrain.wrmin?wrmin:stateconstrain.wrmin;
    double wrmax=tmpparticle[id].wr+objectstateoffset.wroff;wrmax=wrmax<stateconstrain.wrmax?wrmax:stateconstrain.wrmax;
    tmpparticle[id].wr=thrust::random::uniform_real_distribution<double>(wrmin,wrmax)(rng[id]);

    double lfmin=tmpparticle[id].lf-objectstateoffset.lfoff;lfmin=lfmin>stateconstrain.lfmin?lfmin:stateconstrain.lfmin;
    double lfmax=tmpparticle[id].lf+objectstateoffset.lfoff;lfmax=lfmax<stateconstrain.lfmax?lfmax:stateconstrain.lfmax;
    tmpparticle[id].lf=thrust::random::uniform_real_distribution<double>(lfmin,lfmax)(rng[id]);

    double lbmin=tmpparticle[id].lb-objectstateoffset.lboff;lbmin=lbmin>stateconstrain.lbmin?lbmin:stateconstrain.lbmin;
    double lbmax=tmpparticle[id].lb+objectstateoffset.lboff;lbmax=lbmax<stateconstrain.lbmax?lbmax:stateconstrain.lbmax;
    tmpparticle[id].lb=thrust::random::uniform_real_distribution<double>(lbmin,lbmax)(rng[id]);

    deviceBuildModel(tmpparticle[id],egomotion.density);

    tmpparticle[id].weight=0;
    tmpparticle[id].count=0;
    deviceMeasureEdge(tmpparticle[id],0,scan,objectstateoffset.anneal,NULL,NULL,0);
    deviceMeasureEdge(tmpparticle[id],1,scan,objectstateoffset.anneal,NULL,NULL,0);

    return;
}

__global__
void kernelMotionModel(LaserScan * scan, ObjectState * particle, int pnum, ObjectState * tmpparticle, int tmppnum, thrust::minstd_rand * rng, ObjectStateOffset objectstateoffset, StateConstrain stateconstrain, EgoMotion egomotion)
{
    GetThreadID_1D(id);
    if(id>=tmppnum)
    {
        return;
    }
    double index=double(pnum)/double(tmppnum);
    int pid=int(id*index);

    tmpparticle[id]=particle[pid];

    if(egomotion.motionflag)
    {
        tmpparticle[id].v=thrust::random::normal_distribution<double>(tmpparticle[id].v,objectstateoffset.voff)(rng[id]);
        tmpparticle[id].v=tmpparticle[id].v>stateconstrain.vmin?tmpparticle[id].v:stateconstrain.vmin;
        tmpparticle[id].v=tmpparticle[id].v<stateconstrain.vmax?tmpparticle[id].v:stateconstrain.vmax;

        tmpparticle[id].omega=thrust::random::normal_distribution<double>(tmpparticle[id].omega,objectstateoffset.omegaoff)(rng[id]);
        tmpparticle[id].omega=tmpparticle[id].omega>stateconstrain.omegamin?tmpparticle[id].omega:stateconstrain.omegamin;
        tmpparticle[id].omega=tmpparticle[id].omega<stateconstrain.omegamax?tmpparticle[id].omega:stateconstrain.omegamax;
    }
    else
    {
        double vmin=tmpparticle[id].v-objectstateoffset.voff;vmin=vmin>stateconstrain.vmin?vmin:stateconstrain.vmin;
        double vmax=tmpparticle[id].v+objectstateoffset.voff;vmax=vmax<stateconstrain.vmax?vmax:stateconstrain.vmax;
        tmpparticle[id].v=thrust::random::uniform_real_distribution<double>(vmin,vmax)(rng[id]);

        double omegamin=tmpparticle[id].omega-objectstateoffset.omegaoff;omegamin=omegamin>stateconstrain.omegamin?omegamin:stateconstrain.omegamin;
        double omegamax=tmpparticle[id].omega+objectstateoffset.omegaoff;omegamax=omegamax<stateconstrain.omegamax?omegamax:stateconstrain.omegamax;
        tmpparticle[id].omega=thrust::random::uniform_real_distribution<double>(omegamin,omegamax)(rng[id]);
    }

    if(tmpparticle[id].v==0)
    {
        tmpparticle[id].k=(stateconstrain.kmin+stateconstrain.kmax)/2;
    }
    else
    {
        tmpparticle[id].k=tmpparticle[id].omega/tmpparticle[id].v;
        if(tmpparticle[id].k<stateconstrain.kmin)
        {
            tmpparticle[id].k=stateconstrain.kmin;
        }
        if(tmpparticle[id].k>stateconstrain.kmax)
        {
            tmpparticle[id].k=stateconstrain.kmax;
        }
    }
    tmpparticle[id].omega=tmpparticle[id].v*tmpparticle[id].k;

    double R,phi=stateconstrain.amax;
    if(tmpparticle[id].k!=0)
    {
        R=1/fabs(tmpparticle[id].k);
        phi=atan2(4.0,R);
    }

    if(tmpparticle[id].omega>0)
    {
        stateconstrain.amin=DEG2RAD(-20);
        stateconstrain.amax=phi;
        stateconstrain.amax=stateconstrain.amax>stateconstrain.amin?stateconstrain.amax:stateconstrain.amin;
    }
    else if(tmpparticle[id].omega<0)
    {
        stateconstrain.amax=DEG2RAD(20);
        stateconstrain.amin=-phi;
        stateconstrain.amin=stateconstrain.amin<stateconstrain.amax?stateconstrain.amin:stateconstrain.amax;
    }
    else if(tmpparticle[id].omega==0)
    {
        stateconstrain.amin=0;
        stateconstrain.amax=0;
    }

    if(egomotion.motionflag)
    {
        tmpparticle[id].a=thrust::random::normal_distribution<double>(tmpparticle[id].a,objectstateoffset.aoff)(rng[id]);
        tmpparticle[id].a=tmpparticle[id].a>stateconstrain.amin?tmpparticle[id].a:stateconstrain.amin;
        tmpparticle[id].a=tmpparticle[id].a<stateconstrain.amax?tmpparticle[id].a:stateconstrain.amax;
    }
    else
    {
        double amin=tmpparticle[id].a-objectstateoffset.aoff;amin=amin>stateconstrain.amin?amin:stateconstrain.amin;
        double amax=tmpparticle[id].a+objectstateoffset.aoff;amax=amax<stateconstrain.amax?amax:stateconstrain.amax;
        tmpparticle[id].a=thrust::random::uniform_real_distribution<double>(amin,amax)(rng[id]);
    }
//    tmpparticle[id].a=0;

    ObjectState movedparticle=tmpparticle[id];
    deviceAckermannModel(movedparticle,movedparticle,egomotion);
    deviceBuildModel(movedparticle,egomotion.density);

    movedparticle.weight=0;
    movedparticle.count=0;
    deviceMeasureEdge(movedparticle,0,scan,objectstateoffset.anneal,NULL,NULL,0);
    deviceMeasureEdge(movedparticle,1,scan,objectstateoffset.anneal,NULL,NULL,0);
    tmpparticle[id].weight=movedparticle.weight;
    tmpparticle[id].count=movedparticle.count;

    return;
}

__global__
void kernelMotionUpdate(ObjectState * particle, int pnum, EgoMotion egomotion)
{
    GetThreadID_1D(id);
    if(id>=pnum)
    {
        return;
    }
    deviceAckermannModel(particle[id],particle[id],egomotion);
    deviceBuildModel(particle[id],egomotion.density);
}

//==============================================================================

int sampleParticle(int tmppnum, ObjectState & average, ObjectStateOffset & objectstateoffset, bool rejectflag)
{
    cudaMemcpy(h_tmpparticle,d_tmpparticle,sizeof(ObjectState)*tmppnum,cudaMemcpyDeviceToHost);

    double maxlogweight=h_tmpparticle[0].weight;
    double minlogweight=h_tmpparticle[0].weight;
    for(int j=0;j<tmppnum;j++)
    {
        if(maxlogweight<h_tmpparticle[j].weight)
        {
            maxlogweight=h_tmpparticle[j].weight;
        }
        if(minlogweight>h_tmpparticle[j].weight)
        {
            minlogweight=h_tmpparticle[j].weight;
        }
        h_flag[j]=1;
    }

    double maxscale=maxlogweight<=30?1:30/maxlogweight;
    double minscale=minlogweight>=-30?1:-30/minlogweight;
    double scale=maxscale<minscale?maxscale:minscale;
    for(int j=0;j<tmppnum;j++)
    {
        h_tmpparticle[j].weight=exp(h_tmpparticle[j].weight*scale);
        if(j>0)
        {
            h_tmpparticle[j].weight+=h_tmpparticle[j-1].weight;
        }
    }

    int planpnum=tmppnum<RQPN?tmppnum:RQPN;
    double step=1.0/planpnum;
    int accuracy=1000000;
    double samplebase=(rand()%accuracy)*step/accuracy;
    double weightsum=h_tmpparticle[tmppnum-1].weight;
    int pnum=0;

    average.weight=0;
    average.x=0;average.y=0;average.theta=0;
    average.wl=0;average.wr=0;average.lf=0;average.lb=0;
    average.a=0;average.v=0;average.k=0;
    average.count=0;

    ObjectState minstate,maxstate;

    double weight=1.0/planpnum;
    for(int j=0, k=0;j<planpnum;j++)
    {
        double sample=samplebase+j*step;
        while(k<tmppnum)
        {
            if(sample>h_tmpparticle[k].weight/weightsum)
            {
                k++;
                continue;
            }
            if(h_flag[k])
            {
                h_flag[k]=0;
                if(rejectflag)
                {
                    bool flag=0;
                    for(int l=0;l<pnum;l++)
                    {
                        if(h_tmpparticle[k].wl>=h_particle[l].wl-objectstateoffset.wloff&&h_tmpparticle[k].wl<=h_particle[l].wl+objectstateoffset.wloff
                         &&h_tmpparticle[k].wr>=h_particle[l].wr-objectstateoffset.wroff&&h_tmpparticle[k].wr<=h_particle[l].wr+objectstateoffset.wroff
                         &&h_tmpparticle[k].lf>=h_particle[l].lf-objectstateoffset.lfoff&&h_tmpparticle[k].lf<=h_particle[l].lf+objectstateoffset.lfoff
                         &&h_tmpparticle[k].lb>=h_particle[l].lb-objectstateoffset.lboff&&h_tmpparticle[k].lb<=h_particle[l].lb+objectstateoffset.lboff
                         &&h_tmpparticle[k].a>=h_particle[l].a-objectstateoffset.aoff&&h_tmpparticle[k].a<=h_particle[l].a+objectstateoffset.aoff
                         &&h_tmpparticle[k].v>=h_particle[l].v-objectstateoffset.voff&&h_tmpparticle[k].v<=h_particle[l].v+objectstateoffset.voff
                         &&h_tmpparticle[k].k>=h_particle[l].k-objectstateoffset.koff&&h_tmpparticle[k].k<=h_particle[l].k+objectstateoffset.koff)
                        {
                            flag=1;
                            break;
                        }
                    }
                    if(flag)
                    {
                        break;
                    }
                }
                h_particle[pnum]=h_tmpparticle[k];
                h_particle[pnum].weight=weight;
                if(pnum==0)
                {
                    minstate.x=h_particle[pnum].x;maxstate.x=h_particle[pnum].x;
                    minstate.y=h_particle[pnum].y;maxstate.y=h_particle[pnum].y;
                    minstate.theta=h_particle[pnum].theta;maxstate.theta=h_particle[pnum].theta;
                    minstate.wl=h_particle[pnum].wl;maxstate.wl=h_particle[pnum].wl;
                    minstate.wr=h_particle[pnum].wr;maxstate.wr=h_particle[pnum].wr;
                    minstate.lf=h_particle[pnum].lf;maxstate.lf=h_particle[pnum].lf;
                    minstate.lb=h_particle[pnum].lb;maxstate.lb=h_particle[pnum].lb;
                    minstate.a=h_particle[pnum].a;maxstate.a=h_particle[pnum].a;
                    minstate.v=h_particle[pnum].v;maxstate.v=h_particle[pnum].v;
                    minstate.k=h_particle[pnum].k;maxstate.k=h_particle[pnum].k;
                    minstate.omega=h_particle[pnum].omega;maxstate.omega=h_particle[pnum].omega;

                }
                else
                {
                    minstate.x=minstate.x<h_particle[pnum].x?minstate.x:h_particle[pnum].x;
                    maxstate.x=maxstate.x>h_particle[pnum].x?maxstate.x:h_particle[pnum].x;
                    minstate.y=minstate.y<h_particle[pnum].y?minstate.y:h_particle[pnum].y;
                    maxstate.y=maxstate.y>h_particle[pnum].y?maxstate.y:h_particle[pnum].y;
                    minstate.theta=minstate.theta<h_particle[pnum].theta?minstate.theta:h_particle[pnum].theta;
                    maxstate.theta=maxstate.theta>h_particle[pnum].theta?maxstate.theta:h_particle[pnum].theta;
                    minstate.wl=minstate.wl<h_particle[pnum].wl?minstate.wl:h_particle[pnum].wl;
                    maxstate.wl=maxstate.wl>h_particle[pnum].wl?maxstate.wl:h_particle[pnum].wl;
                    minstate.wr=minstate.wr<h_particle[pnum].wr?minstate.wr:h_particle[pnum].wr;
                    maxstate.wr=maxstate.wr>h_particle[pnum].wr?maxstate.wr:h_particle[pnum].wr;
                    minstate.lf=minstate.lf<h_particle[pnum].lf?minstate.lf:h_particle[pnum].lf;
                    maxstate.lf=maxstate.lf>h_particle[pnum].lf?maxstate.lf:h_particle[pnum].lf;
                    minstate.lb=minstate.lb<h_particle[pnum].lb?minstate.lb:h_particle[pnum].lb;
                    maxstate.lb=maxstate.lb>h_particle[pnum].lb?maxstate.lb:h_particle[pnum].lb;
                    minstate.a=minstate.a<h_particle[pnum].a?minstate.a:h_particle[pnum].a;
                    maxstate.a=maxstate.a>h_particle[pnum].a?maxstate.a:h_particle[pnum].a;
                    minstate.v=minstate.v<h_particle[pnum].v?minstate.v:h_particle[pnum].v;
                    maxstate.v=maxstate.v>h_particle[pnum].v?maxstate.v:h_particle[pnum].v;
                    minstate.k=minstate.k<h_particle[pnum].k?minstate.k:h_particle[pnum].k;
                    maxstate.k=maxstate.k>h_particle[pnum].k?maxstate.k:h_particle[pnum].k;
                    minstate.omega=minstate.omega<h_particle[pnum].omega?minstate.omega:h_particle[pnum].omega;
                    maxstate.omega=maxstate.omega>h_particle[pnum].omega?maxstate.omega:h_particle[pnum].omega;

                }
                pnum++;
            }
            else
            {
                h_particle[pnum-1].weight+=weight;
            }
            average.weight+=weight;
            average.x+=h_particle[pnum-1].x*weight;
            average.y+=h_particle[pnum-1].y*weight;
            average.theta+=h_particle[pnum-1].theta*weight;
            average.wl+=h_particle[pnum-1].wl*weight;
            average.wr+=h_particle[pnum-1].wr*weight;
            average.lf+=h_particle[pnum-1].lf*weight;
            average.lb+=h_particle[pnum-1].lb*weight;
            average.a+=h_particle[pnum-1].a*weight;
            average.v+=h_particle[pnum-1].v*weight;
            average.k+=h_particle[pnum-1].k*weight;
            average.omega+=h_particle[pnum-1].omega*weight;
            average.count+=h_particle[pnum-1].count*weight;

            break;
        }
    }

    average.x/=average.weight;
    average.y/=average.weight;
    average.theta/=average.weight;
    average.wl/=average.weight;
    average.wr/=average.weight;
    average.lf/=average.weight;
    average.lb/=average.weight;
    average.a/=average.weight;
    average.v/=average.weight;
    average.k/=average.weight;
    average.omega/=average.weight;
    average.count/=average.weight;
    average.weight/=average.weight;

    average.dx=std::max(average.x-minstate.x,maxstate.x-average.x);
    average.dy=std::max(average.y-minstate.y,maxstate.y-average.y);
    average.dtheta=std::max(average.theta-minstate.theta,maxstate.theta-average.theta);
    average.dwl=std::max(average.wl-minstate.wl,maxstate.wl-average.wl);
    average.dwr=std::max(average.wr-minstate.wr,maxstate.wr-average.wr);
    average.dlf=std::max(average.lf-minstate.lf,maxstate.lf-average.lf);
    average.dlb=std::max(average.lb-minstate.lb,maxstate.lb-average.lb);
    average.da=std::max(average.a-minstate.a,maxstate.a-average.a);
    average.dv=std::max(average.v-minstate.v,maxstate.v-average.v);
    average.dk=std::max(average.k-minstate.k,maxstate.k-average.k);
    average.domega=std::max(average.omega-minstate.omega,maxstate.omega-average.omega);


    cudaMemcpy(d_particle,h_particle,sizeof(ObjectState)*pnum,cudaMemcpyHostToDevice);
    return pnum;
}

#define CALRATIO(ratio, vratio, maxratio, maxrange, minrange) \
    ratio=maxrange/minrange; vratio*=ratio; maxratio=ratio>maxratio?ratio:maxratio;
#define CALZOOM(zoom, maxrange, minrange, N) \
    zoom=log(maxrange/minrange)/log(2)/N;zoom=1/pow(2,zoom);

void SSPF_MeasureModel(ObjectState * particles, int & pnum, ObjectState & average, ObjectStateOffset & objectstateoffset)
{
    cudaMemcpy(d_particle,particles,sizeof(ObjectState)*pnum,cudaMemcpyHostToDevice);

    double ratio=1,vratio=1,maxratio=1;
    CALRATIO(ratio,vratio,maxratio,objectstateoffset.thetaoff,objectstateoffset.thetaprec);
    CALRATIO(ratio,vratio,maxratio,objectstateoffset.wloff,objectstateoffset.wlprec);
    CALRATIO(ratio,vratio,maxratio,objectstateoffset.wroff,objectstateoffset.wrprec);
    CALRATIO(ratio,vratio,maxratio,objectstateoffset.lfoff,objectstateoffset.lfprec);
    CALRATIO(ratio,vratio,maxratio,objectstateoffset.lboff,objectstateoffset.lbprec);
    objectstateoffset.anneal=maxratio*maxratio;
    double N=log(vratio)/log(2);

    CALZOOM(objectstateoffset.thetazoom,objectstateoffset.thetaoff,objectstateoffset.thetaprec,N);
    CALZOOM(objectstateoffset.wlzoom,objectstateoffset.wloff,objectstateoffset.wlprec,N);
    CALZOOM(objectstateoffset.wrzoom,objectstateoffset.wroff,objectstateoffset.wrprec,N);
    CALZOOM(objectstateoffset.lfzoom,objectstateoffset.lfoff,objectstateoffset.lfprec,N);
    CALZOOM(objectstateoffset.lbzoom,objectstateoffset.lboff,objectstateoffset.lbprec,N);
    objectstateoffset.annealratio=pow(objectstateoffset.anneal,-1/N);

    StateConstrain stateconstrain;
    stateconstrain.thetamin=particles[0].theta-objectstateoffset.thetaoff;
    stateconstrain.thetamax=particles[0].theta+objectstateoffset.thetaoff;

    int tmppnum;
    for(int i=1;i<=N;i++)
    {
        tmppnum=pnum*SPN;
        GetKernelDim_1D(blocknum,threadnum,tmppnum);
        kernelMeasureModel<<<blocknum,threadnum>>>(d_scan,d_particle,d_tmpparticle,tmppnum,d_rng,objectstateoffset,stateconstrain,h_egomotion);


        objectstateoffset.thetaoff*=objectstateoffset.thetazoom;
        objectstateoffset.wloff*=objectstateoffset.wlzoom;
        objectstateoffset.wroff*=objectstateoffset.wrzoom;
        objectstateoffset.lfoff*=objectstateoffset.lfzoom;
        objectstateoffset.lboff*=objectstateoffset.lbzoom;
        objectstateoffset.anneal*=objectstateoffset.annealratio;

        pnum=sampleParticle(tmppnum,average,objectstateoffset,REJECTFLAG);
    }
    {
        objectstateoffset.thetaoff=objectstateoffset.thetaprec;
        objectstateoffset.wloff=objectstateoffset.wlprec;
        objectstateoffset.wroff=objectstateoffset.wrprec;
        objectstateoffset.lfoff=objectstateoffset.lfprec;
        objectstateoffset.lboff=objectstateoffset.lbprec;
        objectstateoffset.anneal=1;
        tmppnum=pnum*SPN;
        GetKernelDim_1D(blocknum,threadnum,tmppnum);
        kernelMeasureModel<<<blocknum,threadnum>>>(d_scan,d_particle,d_tmpparticle,tmppnum,d_rng,objectstateoffset,stateconstrain,h_egomotion);
        pnum=sampleParticle(tmppnum,average,objectstateoffset,0);
    }
    {
        cudaMemcpy(particles,d_particle,sizeof(ObjectState)*pnum,cudaMemcpyDeviceToHost);
        deviceBuildModel(average,h_egomotion.density);
    }
}

void SSPF_MotionModel(ObjectState * particles, int & pnum, ObjectState & average, ObjectStateOffset & objectstateoffset)
{
    cudaMemcpy(d_particle,particles,sizeof(ObjectState)*pnum,cudaMemcpyHostToDevice);

    StateConstrain stateconstrain;
    int tmppnum;
    if(h_egomotion.motionflag||SSPFFLAG==0)
    {
        objectstateoffset.anneal=1;
        if(h_egomotion.motionflag)
        {
            tmppnum=MAXPN;
        }
        else
        {
            tmppnum=MAXPN;
        }
        GetKernelDim_1D(blocknum,threadnum,tmppnum);
        kernelMotionModel<<<blocknum,threadnum>>>(d_scan,d_particle,pnum,d_tmpparticle,tmppnum,d_rng,objectstateoffset,stateconstrain,h_egomotion);
        kernelMotionUpdate<<<blocknum,threadnum>>>(d_tmpparticle,tmppnum,h_egomotion);
        pnum=sampleParticle(tmppnum,average,objectstateoffset,0);
    }
    else
    {
        double ratio=1,vratio=1,maxratio=1;
        CALRATIO(ratio,vratio,maxratio,objectstateoffset.aoff,objectstateoffset.aprec);
        CALRATIO(ratio,vratio,maxratio,objectstateoffset.voff,objectstateoffset.vprec);
        CALRATIO(ratio,vratio,maxratio,objectstateoffset.omegaoff,objectstateoffset.omegaprec);
        objectstateoffset.anneal=maxratio*maxratio;
        double N=log(vratio)/log(2);

        CALZOOM(objectstateoffset.azoom,objectstateoffset.aoff,objectstateoffset.aprec,N);
        CALZOOM(objectstateoffset.vzoom,objectstateoffset.voff,objectstateoffset.vprec,N);
        CALZOOM(objectstateoffset.omegazoom,objectstateoffset.omegaoff,objectstateoffset.omegaprec,N);
        objectstateoffset.annealratio=pow(objectstateoffset.anneal,-1/N);

//        stateconstrain.amin=std::max(stateconstrain.amin,average.a-objectstateoffset.aoff);
//        stateconstrain.amax=std::min(stateconstrain.amax,average.a+objectstateoffset.aoff);
//        stateconstrain.vmin=std::max(stateconstrain.vmin,average.v-objectstateoffset.voff);
//        stateconstrain.vmax=std::min(stateconstrain.vmax,average.v+objectstateoffset.voff);
//        stateconstrain.kmin=std::max(stateconstrain.kmin,average.k-objectstateoffset.koff);
//        stateconstrain.kmax=std::min(stateconstrain.kmax,average.k+objectstateoffset.koff);
//        stateconstrain.omegamin=std::max(stateconstrain.omegamin,average.omega-objectstateoffset.omegaoff);
//        stateconstrain.omegamax=std::min(stateconstrain.omegamax,average.omega+objectstateoffset.omegaoff);

        int count=0;
        for(int i=1;i<=N;i++)
        {
            tmppnum=pnum*SPN;
            GetKernelDim_1D(blocknum,threadnum,tmppnum);
            kernelMotionModel<<<blocknum,threadnum>>>(d_scan,d_particle,pnum,d_tmpparticle,tmppnum,d_rng,objectstateoffset,stateconstrain,h_egomotion);            

            objectstateoffset.aoff*=objectstateoffset.azoom;
            objectstateoffset.voff*=objectstateoffset.vzoom;
            objectstateoffset.omegaoff*=objectstateoffset.omegazoom;
            objectstateoffset.anneal*=objectstateoffset.annealratio;

            pnum=sampleParticle(tmppnum,average,objectstateoffset,REJECTFLAG);
            count+=pnum;
        }
        {
            objectstateoffset.aoff=objectstateoffset.aprec;
            objectstateoffset.voff=objectstateoffset.vprec;
            objectstateoffset.omegaoff=objectstateoffset.omegaprec;
            objectstateoffset.anneal=1;
            tmppnum=pnum*SPN;
            GetKernelDim_1D(blocknum,threadnum,tmppnum);
            kernelMotionModel<<<blocknum,threadnum>>>(d_scan,d_particle,pnum,d_tmpparticle,tmppnum,d_rng,objectstateoffset,stateconstrain,h_egomotion);
            kernelMotionUpdate<<<blocknum,threadnum>>>(d_tmpparticle,tmppnum,h_egomotion);
            pnum=sampleParticle(tmppnum,average,objectstateoffset,0);
            count+=pnum;
        }
        std::cerr<<N<<"\n";
    }
    {
        cudaMemcpy(particles,d_particle,sizeof(ObjectState)*pnum,cudaMemcpyDeviceToHost);
        deviceBuildModel(average,h_egomotion.density);
    }
}

//==============================================================================

extern "C" void startTracker()
{
    stopTracker();
    cudaMalloc(&(d_scan),sizeof(LaserScan));
    cudaMalloc(&(d_particle),sizeof(ObjectState)*RQPN);
    cudaMalloc(&(d_tmpparticle),sizeof(ObjectState)*MAXPN);
    cudaMalloc(&(d_rng),sizeof(thrust::minstd_rand)*MAXPN);


    thrust::generate(h_seed,h_seed+MAXPN,rand);
    int * d_seed;
    cudaMalloc(&(d_seed),sizeof(int)*MAXPN);
    cudaMemcpy(d_seed,h_seed,sizeof(int)*MAXPN,cudaMemcpyHostToDevice);
    GetKernelDim_1D(blocks,threads,MAXPN);
    kernelSetRandomSeed<<<blocks,threads>>>(d_seed,d_rng,MAXPN);
    CUDAFREE(d_seed);
}

extern "C" void stopTracker()
{
    CUDAFREE(d_scan);
    CUDAFREE(d_particle);
    CUDAFREE(d_tmpparticle);
    CUDAFREE(d_rng);
}

extern "C" void setLaserScan(LaserScan & scan)
{
    cudaMemcpy(d_scan,&scan,sizeof(LaserScan),cudaMemcpyHostToDevice);
    h_scan=scan;
    if(h_egomotion.validflag)
    {
        double tmpdx=h_egomotion.x-scan.x;
        double tmpdy=h_egomotion.y-scan.y;
        double c=cos(scan.theta);
        double s=sin(scan.theta);
        h_egomotion.dx=c*tmpdx+s*tmpdy;
        h_egomotion.dy=-s*tmpdx+c*tmpdy;
        h_egomotion.dtheta=h_egomotion.theta-scan.theta;
        h_egomotion.dt=scan.timestamp-h_egomotion.timestamp;
    }
    h_egomotion.x=scan.x;
    h_egomotion.y=scan.y;
    h_egomotion.theta=scan.theta;
    h_egomotion.timestamp=scan.timestamp;
    h_egomotion.validflag=1;
    h_egomotion.density=2*PI/scan.beamnum;

//    std::cerr<<h_egomotion.timestamp<<"\t"<<h_egomotion.x<<"\t"<<h_egomotion.y<<"\t"<<h_egomotion.theta<<"\n";
}

extern "C" void initTracker(ObjectState * particles, int & pnum, ObjectState & average, int & beamnum, int * beamid)
{    
    ObjectStateOffset objectstateoffset;

    pnum=1;
//    particles[0]=average;
    particles[0].x=average.x;particles[0].y=average.y;particles[0].theta=average.theta;
    particles[0].wl=average.wl;particles[0].wr=average.wr;particles[0].lf=average.lf;particles[0].lb=average.lb;
    particles[0].a=average.a;particles[0].v=average.v;particles[0].k=average.k;particles[0].omega=average.v*average.k;

    SSPF_MeasureModel(particles,pnum,average,objectstateoffset);

    cudaDeviceSynchronize();

    average.dwl=average.dwl>MINSIGMA?average.dwl:MINSIGMA;
    average.dwr=average.dwr>MINSIGMA?average.dwr:MINSIGMA;
    average.dlf=average.dlf>MINSIGMA?average.dlf:MINSIGMA;
    average.dlb=average.dlb>MINSIGMA?average.dlb:MINSIGMA;

    average.dwl=average.dwl<UNCERTAINTHRESH?average.dwl:MAXSIGMA;
    average.dwr=average.dwr<UNCERTAINTHRESH?average.dwr:MAXSIGMA;
    average.dlf=average.dlf<UNCERTAINTHRESH?average.dlf:MAXSIGMA;
    average.dlb=average.dlb<UNCERTAINTHRESH?average.dlb:MAXSIGMA;

    if(average.dwl<UNCERTAINTHRESH&&average.dwr<UNCERTAINTHRESH)
    {
        double s=sin(average.theta);
        double c=cos(average.theta);
        double offset=(average.wl-average.wr)/2;
        average.x=average.x-s*offset;
        average.y=average.y+c*offset;
        average.wl=average.wr=(average.wl+average.wr)/2;
    }

    deviceBuildModel(average,h_egomotion.density);
    beamnum=0;
    deviceMeasureEdge(average,0,&h_scan,1,&beamnum,beamid,1);
    deviceMeasureEdge(average,1,&h_scan,1,&beamnum,beamid,1);
}

extern "C" void initMotion(ObjectState * particles, int & pnum, ObjectState & average, int & beamnum, int * beamid)
{
    ObjectState preaverage=average;
    ObjectState curaverage;
    ObjectStateOffset objectstateoffset;
    objectstateoffset.thetaoff=objectstateoffset.thetaprec;
    if(average.dwl<objectstateoffset.wlprec)
    {
        objectstateoffset.wloff=objectstateoffset.wlprec;
    }
    if(average.dwr<objectstateoffset.wrprec)
    {
        objectstateoffset.wroff=objectstateoffset.wrprec;
    }
    if(average.dlf<objectstateoffset.lfprec)
    {
        objectstateoffset.lfoff=objectstateoffset.lfprec;
    }
    if(average.dlb<objectstateoffset.lbprec)
    {
        objectstateoffset.lboff=objectstateoffset.lbprec;
    }

    h_egomotion.motionflag=0;

    pnum=1;
//    particles[0]=preaverage;
    particles[0].x=preaverage.x;particles[0].y=preaverage.y;particles[0].theta=preaverage.theta;
    particles[0].wl=preaverage.wl;particles[0].wr=preaverage.wr;particles[0].lf=preaverage.lf;particles[0].lb=preaverage.lb;
    particles[0].a=preaverage.a;particles[0].v=preaverage.v;particles[0].k=preaverage.k;particles[0].omega=preaverage.v*preaverage.k;

    SSPF_MotionModel(particles,pnum,curaverage,objectstateoffset);
    double dx=curaverage.dx;
    double dy=curaverage.dy;
    double dtheta=curaverage.dtheta;

    pnum=1;
//    particles[0]=curaverage;
    particles[0].x=curaverage.x;particles[0].y=curaverage.y;particles[0].theta=curaverage.theta;
    particles[0].wl=curaverage.wl;particles[0].wr=curaverage.wr;particles[0].lf=curaverage.lf;particles[0].lb=curaverage.lb;
    particles[0].a=curaverage.a;particles[0].v=curaverage.v;particles[0].k=curaverage.k;particles[0].omega=curaverage.v*curaverage.k;

    SSPF_MeasureModel(particles,pnum,curaverage,objectstateoffset);

    cudaDeviceSynchronize();
    average=curaverage;

    curaverage.dwl=curaverage.dwl>MINSIGMA?curaverage.dwl:MINSIGMA;
    curaverage.dwr=curaverage.dwr>MINSIGMA?curaverage.dwr:MINSIGMA;
    curaverage.dlf=curaverage.dlf>MINSIGMA?curaverage.dlf:MINSIGMA;
    curaverage.dlb=curaverage.dlb>MINSIGMA?curaverage.dlb:MINSIGMA;

    curaverage.dwl=curaverage.dwl<UNCERTAINTHRESH?curaverage.dwl:MAXSIGMA;
    curaverage.dwr=curaverage.dwr<UNCERTAINTHRESH?curaverage.dwr:MAXSIGMA;
    curaverage.dlf=curaverage.dlf<UNCERTAINTHRESH?curaverage.dlf:MAXSIGMA;
    curaverage.dlb=curaverage.dlb<UNCERTAINTHRESH?curaverage.dlb:MAXSIGMA;

    average.dx=dx;average.dy=dy;average.dtheta=dtheta;

    average.wl=(preaverage.wl*curaverage.dwl*curaverage.dwl+curaverage.wl*preaverage.dwl*preaverage.dwl)/(preaverage.dwl*preaverage.dwl+curaverage.dwl*curaverage.dwl);
    average.dwl=sqrt((preaverage.dwl*preaverage.dwl*curaverage.dwl*curaverage.dwl)/(preaverage.dwl*preaverage.dwl+curaverage.dwl*curaverage.dwl));
    average.dwl=average.dwl>MINSIGMA?average.dwl:MINSIGMA;

    average.wr=(preaverage.wr*curaverage.dwr*curaverage.dwr+curaverage.wr*preaverage.dwr*preaverage.dwr)/(preaverage.dwr*preaverage.dwr+curaverage.dwr*curaverage.dwr);
    average.dwr=sqrt((preaverage.dwr*preaverage.dwr*curaverage.dwr*curaverage.dwr)/(preaverage.dwr*preaverage.dwr+curaverage.dwr*curaverage.dwr));
    average.dwr=average.dwr>MINSIGMA?average.dwr:MINSIGMA;

    average.lf=(preaverage.lf*curaverage.dlf*curaverage.dlf+curaverage.lf*preaverage.dlf*preaverage.dlf)/(preaverage.dlf*preaverage.dlf+curaverage.dlf*curaverage.dlf);
    average.dlf=sqrt((preaverage.dlf*preaverage.dlf*curaverage.dlf*curaverage.dlf)/(preaverage.dlf*preaverage.dlf+curaverage.dlf*curaverage.dlf));
    average.dlf=average.dlf>MINSIGMA?average.dlf:MINSIGMA;

    average.lb=(preaverage.lb*curaverage.dlb*curaverage.dlb+curaverage.lb*preaverage.dlb*preaverage.dlb)/(preaverage.dlb*preaverage.dlb+curaverage.dlb*curaverage.dlb);
    average.dlb=sqrt((preaverage.dlb*preaverage.dlb*curaverage.dlb*curaverage.dlb)/(preaverage.dlb*preaverage.dlb+curaverage.dlb*curaverage.dlb));
    average.dlb=average.dlb>MINSIGMA?average.dlb:MINSIGMA;

    deviceBuildModel(average,h_egomotion.density);
    beamnum=0;
    deviceMeasureEdge(average,0,&h_scan,1,&beamnum,beamid,1);
    deviceMeasureEdge(average,1,&h_scan,1,&beamnum,beamid,1);
}

extern "C" void updateTracker(ObjectState * particles, int & pnum, ObjectState & average, bool & pfflag, int & beamnum, int * beamid)
{
    ObjectState preaverage=average;
    ObjectState curaverage;

    ObjectStateOffset objectstateoffset;

    objectstateoffset.thetaoff=objectstateoffset.thetaprec;
    if(average.dwl<objectstateoffset.wlprec)
    {
        objectstateoffset.wloff=objectstateoffset.wlprec;
    }
    if(average.dwr<objectstateoffset.wrprec)
    {
        objectstateoffset.wroff=objectstateoffset.wrprec;
    }
    if(average.dlf<objectstateoffset.lfprec)
    {
        objectstateoffset.lfoff=objectstateoffset.lfprec;
    }
    if(average.dlb<objectstateoffset.lbprec)
    {
        objectstateoffset.lboff=objectstateoffset.lbprec;
    }

    if(preaverage.dx<=0.5&&preaverage.dy<=0.5&&preaverage.dtheta<=DEG2RAD(10)&&preaverage.count>=3)
    {
        h_egomotion.motionflag=0;
        pnum=1;
        //particles[0]=preaverage;
        particles[0].x=preaverage.x;particles[0].y=preaverage.y;particles[0].theta=preaverage.theta;
        particles[0].wl=preaverage.wl;particles[0].wr=preaverage.wr;particles[0].lf=preaverage.lf;particles[0].lb=preaverage.lb;
        particles[0].a=preaverage.a;particles[0].v=preaverage.v;particles[0].k=preaverage.k;particles[0].omega=preaverage.v*preaverage.k;
    }
    else
    {
        h_egomotion.motionflag=1;
    }

    if(h_egomotion.motionflag)
    {
        objectstateoffset.aoff=DEG2RAD(10);
        objectstateoffset.voff=3;
        objectstateoffset.koff=0.05;
        objectstateoffset.omegaoff=DEG2RAD(10);
    }
    else
    {
        objectstateoffset.aoff=DEG2RAD(30);
        objectstateoffset.voff=10;
        objectstateoffset.koff=0.15;
        objectstateoffset.omegaoff=DEG2RAD(30);
    }

    pfflag=h_egomotion.motionflag;

//    std::cerr<<"Before Motion\n";
//    std::cerr<<preaverage.x<<"\t"<<preaverage.y<<"\t"<<preaverage.theta<<"\t"
//            <<preaverage.wl<<"\t"<<preaverage.wr<<"\t"<<preaverage.lf<<"\t"<<preaverage.lb<<"\t"
//           <<preaverage.a<<"\t"<<preaverage.v<<"\t"<<preaverage.k<<"\t"<<preaverage.v*preaverage.k<<"\t"
//          <<preaverage.count<<"\n";

    SSPF_MotionModel(particles,pnum,curaverage,objectstateoffset);

//    std::cerr<<"After Motion\n";
//    std::cerr<<curaverage.x<<"\t"<<curaverage.y<<"\t"<<curaverage.theta<<"\t"
//            <<curaverage.wl<<"\t"<<curaverage.wr<<"\t"<<curaverage.lf<<"\t"<<curaverage.lb<<"\t"
//           <<curaverage.a<<"\t"<<curaverage.v<<"\t"<<curaverage.k<<"\t"<<curaverage.v*curaverage.k<<"\t"
//          <<curaverage.count<<"\n";

    if(curaverage.count>=10||(curaverage.dx<=0.5&&curaverage.dy<=0.5&&curaverage.dtheta<=DEG2RAD(10)&&curaverage.count>=3))
    {
        double dx=curaverage.dx;
        double dy=curaverage.dy;
        double dtheta=curaverage.dtheta;

        pnum=1;
//        particles[0]=curaverage;
        particles[0].x=curaverage.x;particles[0].y=curaverage.y;particles[0].theta=curaverage.theta;
        particles[0].wl=curaverage.wl;particles[0].wr=curaverage.wr;particles[0].lf=curaverage.lf;particles[0].lb=curaverage.lb;
        particles[0].a=curaverage.a;particles[0].v=curaverage.v;particles[0].k=curaverage.k;particles[0].omega=curaverage.v*curaverage.k;

        SSPF_MeasureModel(particles,pnum,curaverage,objectstateoffset);

//        std::cerr<<"After Geometry\n";
//        std::cerr<<curaverage.x<<"\t"<<curaverage.y<<"\t"<<curaverage.theta<<"\t"
//                <<curaverage.wl<<"\t"<<curaverage.wr<<"\t"<<curaverage.lf<<"\t"<<curaverage.lb<<"\t"
//               <<curaverage.a<<"\t"<<curaverage.v<<"\t"<<curaverage.k<<"\t"<<curaverage.v*curaverage.k<<"\t"
//              <<curaverage.count<<"\n";

        cudaDeviceSynchronize();
        average=curaverage;

        curaverage.dwl=curaverage.dwl>MINSIGMA?curaverage.dwl:MINSIGMA;
        curaverage.dwr=curaverage.dwr>MINSIGMA?curaverage.dwr:MINSIGMA;
        curaverage.dlf=curaverage.dlf>MINSIGMA?curaverage.dlf:MINSIGMA;
        curaverage.dlb=curaverage.dlb>MINSIGMA?curaverage.dlb:MINSIGMA;

        curaverage.dwl=curaverage.dwl<UNCERTAINTHRESH?curaverage.dwl:MAXSIGMA;
        curaverage.dwr=curaverage.dwr<UNCERTAINTHRESH?curaverage.dwr:MAXSIGMA;
        curaverage.dlf=curaverage.dlf<UNCERTAINTHRESH?curaverage.dlf:MAXSIGMA;
        curaverage.dlb=curaverage.dlb<UNCERTAINTHRESH?curaverage.dlb:MAXSIGMA;

        average.dx=dx;average.dy=dy;average.dtheta=dtheta;

        average.wl=(preaverage.wl*curaverage.dwl*curaverage.dwl+curaverage.wl*preaverage.dwl*preaverage.dwl)/(preaverage.dwl*preaverage.dwl+curaverage.dwl*curaverage.dwl);
        average.dwl=sqrt((preaverage.dwl*preaverage.dwl*curaverage.dwl*curaverage.dwl)/(preaverage.dwl*preaverage.dwl+curaverage.dwl*curaverage.dwl));
        average.dwl=average.dwl>MINSIGMA?average.dwl:MINSIGMA;

        average.wr=(preaverage.wr*curaverage.dwr*curaverage.dwr+curaverage.wr*preaverage.dwr*preaverage.dwr)/(preaverage.dwr*preaverage.dwr+curaverage.dwr*curaverage.dwr);
        average.dwr=sqrt((preaverage.dwr*preaverage.dwr*curaverage.dwr*curaverage.dwr)/(preaverage.dwr*preaverage.dwr+curaverage.dwr*curaverage.dwr));
        average.dwr=average.dwr>MINSIGMA?average.dwr:MINSIGMA;

        average.lf=(preaverage.lf*curaverage.dlf*curaverage.dlf+curaverage.lf*preaverage.dlf*preaverage.dlf)/(preaverage.dlf*preaverage.dlf+curaverage.dlf*curaverage.dlf);
        average.dlf=sqrt((preaverage.dlf*preaverage.dlf*curaverage.dlf*curaverage.dlf)/(preaverage.dlf*preaverage.dlf+curaverage.dlf*curaverage.dlf));
        average.dlf=average.dlf>MINSIGMA?average.dlf:MINSIGMA;

        average.lb=(preaverage.lb*curaverage.dlb*curaverage.dlb+curaverage.lb*preaverage.dlb*preaverage.dlb)/(preaverage.dlb*preaverage.dlb+curaverage.dlb*curaverage.dlb);
        average.dlb=sqrt((preaverage.dlb*preaverage.dlb*curaverage.dlb*curaverage.dlb)/(preaverage.dlb*preaverage.dlb+curaverage.dlb*curaverage.dlb));
        average.dlb=average.dlb>MINSIGMA?average.dlb:MINSIGMA;
    }
    else
    {
        cudaDeviceSynchronize();
        average=curaverage;

        average.wl=preaverage.wl;average.dwl=preaverage.dwl;
        average.wr=preaverage.wr;average.dwr=preaverage.dwr;
        average.lf=preaverage.lf;average.dlf=preaverage.dlf;
        average.lb=preaverage.lb;average.dlb=preaverage.dlb;
    }
    deviceBuildModel(average,h_egomotion.density);
    beamnum=0;
    deviceMeasureEdge(average,0,&h_scan,1,&beamnum,beamid,1);
    deviceMeasureEdge(average,1,&h_scan,1,&beamnum,beamid,1);
}
