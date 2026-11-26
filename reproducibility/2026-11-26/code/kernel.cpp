// Unified M/M/k price optimizer. Patient supports enclose the lower-level graph.
// C++17, IEEE binary64, no fast-math. Interval operations enclose the input data.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <vector>
using std::array;
constexpr double inf=std::numeric_limits<double>::infinity();
double down(double x){return std::nextafter(x,-inf);}
double up(double x){return std::nextafter(x,inf);}
struct I {
    double l,h;
    I(double x=0):l(x),h(x){}
    I(double x,double y):l(x),h(y){}
    bool zero()const{return l==0 && h==0;}
    double mid()const{return (l+h)*.5;}
};
I primitive(double x){
    if(x==std::floor(x))return I(x);
    return I(down(down(x)),up(up(x)));
}
I operator+(I x,I y){
    if(x.zero())return y;
    if(y.zero())return x;
    return {down(x.l+y.l),up(x.h+y.h)};
}
I operator-(I x){return {-x.h,-x.l};}
I operator-(I x,I y){if(x.l==x.h&&y.l==y.h&&x.l==y.l)return 0;return x+-y;}
I operator*(I x,I y){
    if(x.zero()||y.zero())return 0;
    double a=x.l*y.l,b=x.l*y.h,c=x.h*y.l,d=x.h*y.h;
    return {down(std::min({a,b,c,d})),up(std::max({a,b,c,d}))};
}
I operator/(I x,I y){
    if(x.zero())return 0;
    if(y.l<=0&&y.h>=0)return {-inf,inf};
    return x*I(down(1/y.h),up(1/y.l));
}
I intersect(I a,I b){return {std::max(a.l,b.l),std::min(a.h,b.h)};}
bool empty(I x){return x.h<x.l;}
I hull(I a,I b){return {std::min(a.l,b.l),std::max(a.h,b.h)};}
double clamp(double x,double lo,double hi){return std::max(lo,std::min(hi,x));}
using Pair=array<I,2>;
using Vec=array<double,2>;
I dot(Pair a,Pair b){return a[0]*b[0]+a[1]*b[1];}
Pair subtract(Pair a,Pair b){return {a[0]-b[0],a[1]-b[1]};}

struct Pool {
    int k;
    I rate;
    double nu;
    std::vector<long double> coefficients;
    Pool(int staff,I mu):k(staff),rate(mu),nu(mu.mid()),coefficients(k){
        // P_k(z)/(k-1)!; Horner evaluation keeps all operations positive.
        long double fact=1;
        for(int i=0;i<k;i++){
            if(i)fact*=i;
            coefficients[i]=(k-i)/fact;
        }
    }
    double capacity()const{return k*nu;}
    double wait(double load)const{
        if(load<0)load=0;
        double gap=capacity()-load;
        if(gap<=0)return inf;
        double z=load/nu,tail=1.,sum=1.;
        for(int i=1;i<k;i++){
            tail*=z/i;
            sum+=tail;
        }
        tail*=z*nu/gap;
        return 1/nu+(tail/(sum+tail))/gap;
    }
    // For our k <= 18, a positive Horner polynomial and its monomial have
    // fewer than 100 roundings. At most 512*u is used multiplicatively here,
    // and endpoint loads and rates come from outward interval arithmetic.
    // The possibly ill-conditioned capacity subtraction is evaluated as an
    // interval before divisions; no relative error assumption is made on it.
    I at(double load,double speed)const{
        load=std::max(0.,load);
        I gap=I(k)*I(speed)-I(load);
        if(gap.l<=0){
            if(gap.h<=0)return {inf,inf};
            return {0,inf};
        }
        if(load==0)return I(1)/I(speed);
        long double z=(long double)load/speed,den=coefficients.back(),term=z;
        for(int i=k-2;i>=0;i--){den=den*z+coefficients[i];term*=z/(i+1);}
        long double odds=term/den;
        long double numerator=odds;
        long double eps=512.*std::numeric_limits<long double>::epsilon();
        double ratio_lo=down((double)(numerator*(1-eps)));
        double ratio_hi=up((double)(numerator*(1+eps)));
        I q=I(ratio_lo,ratio_hi)/gap;
        return I(1)/I(speed)+q;
    }
    I w(I load)const{
        return {at(load.l,rate.h).l,at(load.h,rate.l).h};
    }
    double derivative(double load)const{
        double gap=capacity()-load;
        if(load<=0)return k==1?1/(nu*nu):0;
        double z=load/nu, t=1, s=k, weighted=0;
        for(int i=1;i<k;i++){t*=z/i; s+=(k-i)*t; weighted+=i*(k-i)*t;}
        double q=wait(load)-1/nu;
        return q*((k-weighted/s)/load+1/gap);
    }
    I wprime(double load)const{
        if(load<=0)return k==1?I(1)/(rate*rate):I(0);
        I q=w(load)-I(1)/rate;
        q.l=std::max(0.,q.l);
        I z=I(load)/rate, term=1, pol=k, weighted=0;
        for(int i=1;i<k;i++){
            term=term*z/I(i);
            pol=pol+I(k-i)*term;
            weighted=weighted+I(i*(k-i))*term;
        }
        return q*((I(k)-weighted/pol)/I(load)+I(1)/(I(k)*rate-I(load)));
    }
    double inv(double target)const{
        if(target<=1/nu)return 0;
        double a=0,b=capacity();
        for(int i=0;i<38;i++){
            double m=(a+b)*.5;
            if(wait(m)<target)a=m;else b=m;
        }
        return (a+b)*.5;
    }
    I inverse(I target,double lower,double upper)const{
        if(w(lower).l>target.h || w(upper).h<target.l)return {1,0};
        double a=lower,b=upper;
        if(std::isfinite(target.l) && w(lower).h<target.l){
            double y=clamp(inv(target.l),lower,upper);
            double pad=1e-7;
            for(int s=0;s<8;s++,pad*=8){
                double m=std::max(a,y-pad);
                if(w(m).h <= target.l){a=m;break;}
            }
        }
        if(std::isfinite(target.h) && w(upper).l>target.h){
            double y=clamp(inv(target.h),lower,upper);
            double pad=1e-7;
            for(int s=0;s<8;s++,pad*=8){
                double m=std::min(b,y+pad);
                if(w(m).l >= target.h){b=m;break;}
            }
        }
        return {a,b};
    }
    double load_upper(double waitmax)const{
        double upper=capacity();
        double y=inv(waitmax)+1e-6;
        for(int j=0;j<10;j++){
            if(y>=upper)return upper;
            if(w(y).l>waitmax)return y;
            y=(y+upper)*.5;
        }
        return upper;
    }
};
// 0=Lambda,1=E0,2=a,3=b,4=d,5=theta,6=alpha,7=V,8=s,9=T,
// 10=cv,11=cf,12=r,13=nu_v,14=nu_f,15=K,16=path(N=0,O=1,S=2,R=3),
// 17=gamma,18=mandatory-receipt weight,19=eta,
// 20=lo_v,21=hi_v,22=lo_f,23=hi_f,24=reference offline tariff.
struct Model {
    I L,E,a,b,d,theta,alpha,V,s,T,r,receipt,eta,ref;
    I det;
    int K,path;
    Pair q[2],bill[2],e,be,c,val,P;
    Vec pv;
    explicit Model(const double*z){
        L=primitive(z[0]); E=primitive(z[1]);a=primitive(z[2]);b=primitive(z[3]);d=primitive(z[4]);
        theta=primitive(z[5]);alpha=primitive(z[6]);V=primitive(z[7]);s=primitive(z[8]);T=primitive(z[9]);
        c={primitive(z[10]),primitive(z[11])};r=primitive(z[12]);pv={z[13],z[14]};
        K=(int)z[15];path=(int)z[16];I g=primitive(z[17]);receipt=primitive(z[18]);
        eta=primitive(z[19]);ref=primitive(z[24]);
        P={I(z[20],z[21]),I(z[22],z[23])};
        if(path==0)b=0;
        if(path==3){
            I zz=I(1)-g*a*b,ze=I(1)-g*a*d;
            q[0]={I(1)+a*b/zz,a/zz};q[1]={b/zz,I(1)/zz};
            e={d/ze,I(1)/ze};
            val={V+a*s-T*a/zz,V+s-T/zz};
            det=I(1)/zz;
        }else{
            q[0]={I(1)+(path==2?a*b:I(0)),a};
            q[1]={b,1};e={d,1};val={V+a*(s-T),V+s-T};
            det=path==2?I(1):I(1)-a*b;
        }
        bill[0]={I(1)+theta*(q[0][0]-I(1)),q[0][1]};
        if(path<=1)bill[0][0]=1;
        bill[1]={theta*q[1][0],q[1][1]};
        be={theta*e[0],e[1]};
    }
    I exogenous(Pair p)const{
        if(eta.zero())return E;
        I a=eta*alpha*(dot(be,p)-ref);
        a.l=std::max(a.l,0.);a.h=std::max(a.h,0.);
        // Direct positive exponential series with a geometric remainder.
        auto exponential=[](double x){
            I term=1,z=1;
            for(int j=1;j<=32;j++){term=term*I(x)/I(j);z=z+term;}
            I t=I(x)/I(33);
            if(t.h>=1)return I(z.l,inf);
            return I(z.l,(z+term*t/(I(1)-t)).h);
        };
        return E/I(exponential(a.l).l,exponential(a.h).h);
    }
};

struct Eval {
    double J=-inf,G=0,x=0,y=0,v=0,f=0,lv=0,lf=0,wv=0,wf=0,E=0,res=0;
    double p=0,q=0,lo=-inf,hi=inf;
    int support=-1;
};
struct Bound {
    double high=-inf,low=inf;
    void add(I value){high=std::max(high,value.h);low=std::min(low,value.l);}
};

struct Problem {
    const Model&m;
    int k;
    array<Pool,2> pool;
    Pair W0, ell_max, qdiff, hdiff, beta, invone;
    I a00,a01,a10,a11,kap,rho;
    double flow_tol=1e-7;
    Problem(const Model &model,int split):
        m(model),k(split),pool{Pool(k,primitive(m.pv[0])),Pool(m.K-k,primitive(m.pv[1]))}{
        qdiff=subtract(m.q[0],m.q[1]);hdiff=subtract(m.bill[0],m.bill[1]);
        a00=m.q[1][1]/m.det; a01=-m.q[1][0]/m.det;
        a10=-m.q[0][1]/m.det; a11=m.q[0][0]/m.det;
        beta={a00*m.val[0]+a10*m.val[1],a01*m.val[0]+a11*m.val[1]};
        invone={a00+a10,a01+a11};
        kap=m.theta+(I(1)-m.theta)*a00;
        rho=-(I(1)-m.theta)*a01;
        Pair minimum={I(m.P[0].l),I(m.P[1].l)};
        for(int j=0;j<2;j++){
            double upper_wait=pool[j].w(m.E*m.e[j]).h;
            for(int i=0;i<2;i++)if(m.q[i][j].l>0){
                I w=(m.val[i]-m.alpha*dot(m.bill[i],minimum))/(m.c[j]*m.q[i][j]);
                upper_wait=std::max(upper_wait,w.h);
            }
            ell_max[j]=pool[j].load_upper(upper_wait);
            // At least the current mandatory load must be permitted.
            ell_max[j].h=std::max(ell_max[j].h,(m.E*m.e[j]).h);
        }
        Pair price=m.P;
        I E0=m.exogenous(price);
        Pair ext{E0*m.e[0],E0*m.e[1]};
        for(int j=0;j<2;j++){
            int o=1-j;
            double bound=0;
            for(int i=0;i<2;i++){
                if(m.q[i][j].zero())continue;
                if(m.q[i][o].l<=0){bound=inf;break;}
                bound=std::max(bound,(m.q[i][j]/m.q[i][o]).h);
            }
            if(std::isfinite(bound)){
                I upper=ext[j]+I(bound)*(I(ell_max[o].h)-ext[o]);
                ell_max[j].h=std::min(ell_max[j].h,upper.h);
            }
            double maxcoef=std::max(m.q[0][j].h,m.q[1][j].h);
            ell_max[j].h=std::min(ell_max[j].h,(ext[j]+m.L*I(maxcoef)).h);
        }
    }
    bool admissible()const{
        for(int j=0;j<2;j++){
            I ex=m.E*m.e[j],cap=I(pool[j].k)*pool[j].rate;
            if(ex.l>=cap.h || (ex.l==ex.h&&cap.l==cap.h&&ex.l==cap.l))return false;
            if(ex.h>=cap.l)throw std::runtime_error("indeterminate input stability");
        }
        return true;
    }
    Pair loads(Pair x,I E)const{
        return {m.e[0]*E+m.q[0][0]*x[0]+m.q[1][0]*x[1],
                m.e[1]*E+m.q[0][1]*x[0]+m.q[1][1]*x[1]};
    }
    Pair queues(Pair ell)const{return {pool[0].w(ell[0]),pool[1].w(ell[1])};}
    Pair cg(Pair ell)const{
        Pair w=queues(ell);
        return {m.c[0]*w[0],m.c[1]*w[1]};
    }
    Pair contribution(double t,Pair start,Pair direction,Pair coeff)const{
        return {coeff[0]*m.c[0]*pool[0].w(start[0]+direction[0]*I(t)),
                coeff[1]*m.c[1]*pool[1].w(start[1]+direction[1]*I(t))};
    }
    Vec utility(Vec x,Vec p,double E)const{
        double cost[2],u[2];
        for(int j=0;j<2;j++){
            double ell=x[0]*m.q[0][j].mid()+x[1]*m.q[1][j].mid()+E*m.e[j].mid();
            cost[j]=pool[j].wait(ell)*m.c[j].mid();
        }
        for(int i=0;i<2;i++){
            u[i]=m.val[i].mid()-m.alpha.mid()*(m.bill[i][0].mid()*p[0]+m.bill[i][1].mid()*p[1]);
            for(int j=0;j<2;j++)if(m.q[i][j].mid()!=0)u[i]-=m.q[i][j].mid()*cost[j];
        }
        return {u[0],u[1]};
    }
    double residual(Vec x,Vec u)const{
        double top=std::max({0.,u[0],u[1]}), r=0;
        for(int i=0;i<2;i++)if(x[i]>1e-9)r=std::max(r,top-u[i]);
        if(x[0]+x[1]<m.L.mid()-1e-8)r=std::max(r,top);
        return r;
    }
    Eval eval(Vec p)const{
        Eval ans;ans.p=p[0];ans.q=p[1];
        double E=m.exogenous({p[0],p[1]}).mid(),D=m.L.mid();
        Vec x{0,0},u=utility(x,p,E);
        auto finish=[&](int support,Vec x,Vec u){
            ans.support=support;
            ans.x=x[0];ans.y=x[1];ans.v=u[0];ans.f=u[1];ans.E=E;ans.res=residual(x,u);
            ans.lv=E*m.e[0].mid()+x[0]*m.q[0][0].mid()+x[1]*m.q[1][0].mid();
            ans.lf=E*m.e[1].mid()+x[0]*m.q[0][1].mid()+x[1]*m.q[1][1].mid();
            ans.wv=pool[0].wait(ans.lv);ans.wf=pool[1].wait(ans.lf);
            double mb=E*(p[0]*m.be[0].mid()+p[1]*m.be[1].mid());
            ans.G=mb;
            for(int i=0;i<2;i++)ans.G+=x[i]*(p[0]*m.bill[i][0].mid()+p[1]*m.bill[i][1].mid());
            ans.J=ans.G+(m.receipt.mid()-1)*mb-m.r.mid()*(D-x[0]-x[1]);
            if(!(std::isfinite(ans.wv)&&std::isfinite(ans.wf)&&ans.res<=2e-6))ans.J=-inf;
            return ans;
        };
        if(std::max(u[0],u[1])<=1e-11)return finish(0,x,u);
        Vec net;
        for(int i=0;i<2;i++)net[i]=m.val[i].mid()-m.alpha.mid()*(m.bill[i][0].mid()*p[0]+m.bill[i][1].mid()*p[1]);
        double d0=(a00.mid()*net[0]+a10.mid()*net[1])/m.c[0].mid();
        double d1=(a01.mid()*net[0]+a11.mid()*net[1])/m.c[1].mid();
        double ev=E*m.e[0].mid(),ef=E*m.e[1].mid();
        if(d0>=pool[0].wait(ev)&&d1>=pool[1].wait(ef)){
            double l=pool[0].inv(d0)-ev,f=pool[1].inv(d1)-ef;
            x={a00.mid()*l+a01.mid()*f,a10.mid()*l+a11.mid()*f};
            if(x[0]>=0&&x[1]>=0&&x[0]+x[1]<=D){
                ans=finish(6,x,utility(x,p,E));
                if(ans.J>-inf)return ans;
            }
        }
        for(int i=0;i<2;i++){
            if(u[i]<0)continue;
            double a=0,b=D;
            for(int j=0;j<2;j++)if(m.q[i][j].mid()>0)
                b=std::min(b,(pool[j].capacity()-E*m.e[j].mid()-1e-8)/m.q[i][j].mid());
            if(b<0)continue;
            Vec w{0,0};w[i]=b;
            Vec ub=utility(w,p,E);
            bool full=ub[i]>=0&&b==D;
            if(!full){
                if(ub[i]>0)continue;
                for(int j=0;j<44;j++){
                    double t=(a+b)*.5;w[i]=t;
                    if(utility(w,p,E)[i]>0)a=t;else b=t;
                }
                w[i]=(a+b)*.5;ub=utility(w,p,E);
            }
            if(ub[i]>=-1e-8&&ub[1-i]<=ub[i]+1e-8){
                ans=finish(full?1+i:3+i,w,ub);
                if(ans.J>-inf)return ans;
            }
        }
        double l=0,h=D;
        for(int j=0;j<2;j++){
            double q=qdiff[j].mid();
            double cap=pool[j].capacity()-1e-8-E*m.e[j].mid()-D*m.q[1][j].mid();
            if(q>0)h=std::min(h,cap/q);
            else if(q<0)l=std::max(l,cap/q);
            else if(cap<0)return Eval();
        }
        if(l>h)return Eval();
        for(int j=0;j<46;j++){
            double t=(l+h)*.5; Vec v=utility({t,D-t},p,E);
            if(v[0]>v[1])l=t;else h=t;
        }
        x={(l+h)*.5,D-(l+h)*.5};
        return finish(5,x,utility(x,p,E));
    }
    I objective(Pair x,Pair p,I E)const{
        I X=intersect(x[0]+x[1],I(0,m.L.h));
        return x[0]*dot(m.bill[0],p)+x[1]*dot(m.bill[1],p)
               +m.receipt*E*dot(m.be,p)+m.r*(X-m.L);
    }
    I arc_domain(Pair origin,Pair direction)const{
        I x(0,m.L.h);
        for(int j=0;j<2;j++){
            I spare=I(ell_max[j].h)-origin[j];
            if(direction[j].l>0)x.h=std::min(x.h,(spare/direction[j]).h);
            else if(direction[j].h<0)x.l=std::max(x.l,(spare/direction[j]).l);
            else if(direction[j].zero()&&spare.h<0)return {1,0};
        }
        return x;
    }
    I curve(Pair origin,Pair dir,Pair coeff,double x)const{
        I z=0;
        for(int j=0;j<2;j++)if(!coeff[j].zero())
            z=z+coeff[j]*m.c[j]*pool[j].w(origin[j]+dir[j]*I(x));
        return z;
    }
    double nominal_curve(Vec origin,Vec dir,Vec coeff,double x)const{
        double z=0;
        for(int j=0;j<2;j++)if(coeff[j]!=0)
            z+=coeff[j]*pool[j].wait(origin[j]+dir[j]*x);
        return z;
    }
    // For any true equilibrium on the segment T(t) is increasing. Interval
    // verification of the two endpoint signs is what shrinks the domain.
    I arc_root(Pair origin,Pair dir,Pair coeff,I x,I target)const{
        if(empty(x))return x;
        I A=curve(origin,dir,coeff,x.l),B=curve(origin,dir,coeff,x.h);
        if(A.l>target.h||B.h<target.l)return {1,0};
        Vec o{origin[0].mid(),origin[1].mid()},d{dir[0].mid(),dir[1].mid()};
        Vec c{(coeff[0]*m.c[0]).mid(),(coeff[1]*m.c[1]).mid()};
        auto guess=[&](double rhs){
            double l=x.l,h=x.h;
            for(int i=0;i<33;i++){
                double t=(l+h)*.5;
                if(nominal_curve(o,d,c,t)<rhs)l=t;else h=t;
            }
            return (l+h)*.5;
        };
        double l=x.l,h=x.h;
        if(A.h<target.l){
            double y=guess(target.l),pad=flow_tol;
            for(int j=0;j<9;j++,pad*=8){
                double t=std::max(x.l,y-pad);
                if(curve(origin,dir,coeff,t).h<=target.l){l=t;break;}
            }
        }
        if(B.l>target.h){
            double y=guess(target.h),pad=flow_tol;
            for(int j=0;j<9;j++,pad*=8){
                double t=std::min(x.h,y+pad);
                if(curve(origin,dir,coeff,t).l>=target.h){h=t;break;}
            }
        }
        return {l,h};
    }

    struct Row{Pair a;I b;};
    // Dual interval bound for 2D linear price constraints. Any nonnegative
    // multipliers give a valid bound. Primal vertices supply price candidates
    // only. No certificate depends on identifying all vertices in floating point.
    double linear_bound(Pair p,Pair obj,const std::vector<Row>&rows,std::vector<Vec>*candidates=nullptr)const{
        auto dual=[&](double u,double v,const Row&A,const Row&B){
            if(u<0||v<0||!std::isfinite(u)||!std::isfinite(v))return inf;
            Pair r{obj[0]-I(u)*A.a[0]-I(v)*B.a[0],obj[1]-I(u)*A.a[1]-I(v)*B.a[1]};
            return (I(u)*A.b+I(v)*B.b+dot(p,r)).h;
        };
        auto fathom=[&](double u,double v,const Row&A,const Row&B){
            if(u<0||v<0||!std::isfinite(u)||!std::isfinite(v))return false;
            Pair r{I(u)*A.a[0]+I(v)*B.a[0],I(u)*A.a[1]+I(v)*B.a[1]};
            return dot(r,p).l>(I(u)*A.b+I(v)*B.b).h;
        };
        double bound=dot(p,obj).h;
        for(const Row&A:rows){
            if(fathom(1,0,A,A))return -inf;
            for(int j=0;j<2;j++){
                double a=A.a[j].mid();
                if(std::abs(a)>1e-12)bound=std::min(bound,dual(obj[j].mid()/a,0,A,A));
            }
        }
        for(size_t i=0;i<rows.size();i++)for(size_t j=i+1;j<rows.size();j++){
            const Row&A=rows[i],&B=rows[j];
            double a=A.a[0].mid(),b=A.a[1].mid(),c=B.a[0].mid(),d=B.a[1].mid();
            double det=a*d-b*c;
            if(std::abs(det)>1e-13){
                bound=std::min(bound,dual((obj[0].mid()*d-obj[1].mid()*c)/det,
                                          (a*obj[1].mid()-b*obj[0].mid())/det,A,B));
                if(candidates){
                    Vec z{(A.b.mid()*d-b*B.b.mid())/det,(a*B.b.mid()-A.b.mid()*c)/det};
                    if(z[0]>=p[0].l-1e-7&&z[0]<=p[0].h+1e-7&&z[1]>=p[1].l-1e-7&&z[1]<=p[1].h+1e-7)
                        candidates->push_back({clamp(z[0],p[0].l,p[0].h),clamp(z[1],p[1].l,p[1].h)});
                }
            }
            if(fathom(1,1,A,B))return -inf;
            if(std::abs(c)>1e-12&&fathom(1,-a/c,A,B))return -inf;
            if(std::abs(d)>1e-12&&fathom(1,-b/d,A,B))return -inf;
        }
        return bound;
    }
    double axis_tangent(int i,I X,Pair ext,I E,Pair p)const{
        // Substitute participation equality into revenue, eliminating one
        // tariff. The route flow plus its mandatory receipt weight is positive.
        int j=m.bill[i][1].l>0?1:0,g=1-j;
        I ratio=m.receipt*E*m.be[j]/m.bill[i][j];
        I R=m.receipt*E*m.be[g]-ratio*m.bill[i][g];
        double t=X.mid();
        I delay=curve(ext,m.q[i],m.q[i],t), derivative=0;
        for(int h=0;h<2;h++)if(!m.q[i][h].zero()){
            I ell=ext[h]+m.q[i][h]*I(t);
            I v(pool[h].wprime(ell.l).l,pool[h].wprime(ell.h).h);
            derivative=derivative+m.c[h]*m.q[i][h]*m.q[i][h]*v;
        }
        if(!std::isfinite(delay.h)||!std::isfinite(derivative.h))return inf;
        I B=(m.val[i]-delay)/m.alpha;
        I value=(I(t)+ratio)*B+m.r*(I(t)-m.L);
        I slope=B+m.r-(I(t)+ratio)*derivative/m.alpha;
        return (value+slope*(X-I(t))+R*p[g]).h;
    }
    I mixed_envelope(Pair ell,Pair ext,I E)const{
        I ef=m.receipt*E*m.e[1],ev=m.receipt*E*(m.be[0]+rho*m.e[1])/kap;
        Pair gamma{ev-ext[0],ef-ext[1]};
        I constant=-m.r*(m.L+dot(invone,ext));
        Pair gradient;
        for(int j=0;j<2;j++){
            // A tangent may be evaluated at any nonnegative stable load. Choose
            // the approximate stationary load within the supplied interval.
            double lo=ell[j].l,hi=ell[j].h;
            double b=beta[j].mid(), r=m.r.mid()*invone[j].mid(), c=m.c[j].mid();
            double g=gamma[j].mid(),a=m.alpha.mid();
            for(int n=0;n<12;n++){
                double t=(lo+hi)*.5;
                double der=(b-c*(pool[j].wait(t)+(t+g)*pool[j].derivative(t)))/a+r;
                if(der>0)lo=t;else hi=t;
            }
            double t=std::max((lo+hi)*.5,ext[j].h);
            // The actual concave summand has nonnegative (load+gamma) on
            // the feasible interval; t need not satisfy any price condition.
            I val=(I(t)+gamma[j])*(beta[j]-m.c[j]*pool[j].w(t))/m.alpha
                  +m.r*invone[j]*I(t);
            I slope=(beta[j]-m.c[j]*(pool[j].w(t)+(I(t)+gamma[j])*pool[j].wprime(t)))/m.alpha
                    +m.r*invone[j];
            if(!std::isfinite(val.h)||!std::isfinite(slope.h)||!std::isfinite(slope.l))return {-inf,inf};
            constant=constant+val-slope*I(t);
            gradient[j]=slope;
        }
        std::vector<Row> constraints = {
            {{-a00,-a01},-dot({a00,a01},ext)},
            {{-a10,-a11},-dot({a10,a11},ext)},
            {invone,m.L+dot(invone,ext)}
        };
        return constant+I(linear_bound(ell,gradient,constraints));
    }
    double common_upper(Pair p,I E)const{
        I vf=dot(m.bill[0],p),ff=dot(m.bill[1],p);
        I mandatory=m.receipt*E*dot(m.be,p);
        double u=(mandatory+m.L*I(std::max(vf.h,ff.h))).h;
        // A participation screen using just the service-time floor. Prices
        // at their box ceilings give the common receipt bound for all entries.
        Pair high{p[0].h,p[1].h};
        I vb=dot(m.bill[0],high),fb=dot(m.bill[1],high);
        I min_delay=dot(m.q[0],{m.c[0]/pool[0].rate,m.c[1]/pool[1].rate});
        if(fb.l>0&&(m.val[0]-min_delay).h < (m.alpha*fb).l)
            u=std::min(u,(mandatory+m.L*fb).h);
        return u;
    }

    Bound bound(Pair p,bool point=false,std::vector<Vec>*pv=nullptr)const{
        Bound out;
        I E=m.exogenous(p);
        Pair ext{E*m.e[0],E*m.e[1]},net;
        for(int i=0;i<2;i++)net[i]=m.val[i]-m.alpha*dot(m.bill[i],p);
        Pair base=cg(ext);
        if((net[0]-dot(m.q[0],base)).l<=0&&(net[1]-dot(m.q[1],base)).l<=0){
            out.add(objective({0,0},p,E));
        }
        for(int i=0;i<2;i++){
            Pair x{0,0};x[i]=m.L;
            Pair ell=loads(x,E);
            if(ell[0].l<=ell_max[0].h && ell[1].l<=ell_max[1].h){
                Pair cost=cg(ell);
                I used=net[i]-dot(m.q[i],cost);
                I diff=net[i]-net[1-i]-dot(subtract(m.q[i],m.q[1-i]),cost);
                if(used.h>=0&&diff.h>=0){
                    I value=objective(x,p,E);
                    if(!point&&m.eta.zero()&&std::isfinite(cost[0].h)&&std::isfinite(cost[1].h)){
                        Pair rate{m.L*m.bill[i][0]+m.receipt*E*m.be[0],
                                  m.L*m.bill[i][1]+m.receipt*E*m.be[1]};
                        std::vector<Row> rows{
                            {{{1,0}},p[0].h},{{{-1,0}},-p[0].l},
                            {{{0,1}},p[1].h},{{{0,-1}},-p[1].l},
                            {{m.alpha*m.bill[i][0],m.alpha*m.bill[i][1]},m.val[i]-dot(m.q[i],cost)},
                            {{m.alpha*(m.bill[i][0]-m.bill[1-i][0]),m.alpha*(m.bill[i][1]-m.bill[1-i][1])},
                              m.val[i]-m.val[1-i]-dot(subtract(m.q[i],m.q[1-i]),cost)}
                        };
                        value.h=std::min(value.h,linear_bound(p,rate,rows,pv));
                    }
                    if(value.h>-inf)out.add(value);
                }
            }
            I im=arc_root(ext,m.q[i],m.q[i],arc_domain(ext,m.q[i]),net[i]);
            if(!empty(im)){
                x={0,0};x[i]=im;
                ell=loads(x,E);Pair cost=cg(ell);
                if((net[1-i]-dot(m.q[1-i],cost)).l<=0){
                    I value=objective(x,p,E);
                    if(!point)value.h=std::min(value.h,axis_tangent(i,im,ext,E,p));
                    out.add(value);
                }
            }
        }
        Pair origin{ext[0]+m.q[1][0]*m.L,ext[1]+m.q[1][1]*m.L};
        I t=arc_root(origin,qdiff,qdiff,arc_domain(origin,qdiff),net[0]-net[1]);
        if(!empty(t)){
            Pair ell{origin[0]+qdiff[0]*t,origin[1]+qdiff[1]*t};
            for(int j=0;j<2;j++)ell[j]=intersect(ell[j],I(ext[j].l,ell_max[j].h));
            Pair cost=cg(ell);
            if(!empty(ell[0])&&!empty(ell[1])&&(net[0]-dot(m.q[0],cost)).h>=0&&(net[1]-dot(m.q[1],cost)).h>=0){
                I y=intersect(m.L-t,I(0,m.L.h));
                I value=objective({t,y},p,E);
                if(!point&&!empty(ell[0])&&!empty(ell[1])){
                    value.h=std::min(value.h,mixed_envelope(ell,ext,E).h);
                    I D=dot(qdiff,cost);
                    I delta=(m.val[0]-m.val[1]-D)/m.alpha;
                    I fixed=m.L*dot(m.bill[1],p)+m.receipt*E*dot(m.be,p);
                    value.h=std::min(value.h,(fixed+t*delta).h);
                }
                out.add(value);
            }
        }
        I wv=(beta[0]-m.alpha*kap*p[0])/m.c[0];
        I wf=(beta[1]-m.alpha*(p[1]-rho*p[0]))/m.c[1];
        Pair ell{pool[0].inverse(wv,ext[0].l,ell_max[0].h),pool[1].inverse(wf,ext[1].l,ell_max[1].h)};
        if(!empty(ell[0])&&!empty(ell[1])){
            Pair diff=subtract(ell,ext);
            Pair x{a00*diff[0]+a01*diff[1],a10*diff[0]+a11*diff[1]};
            if(x[0].h>=0&&x[1].h>=0&&(x[0]+x[1]).l<=m.L.h){
                x[0]=intersect(x[0],I(0,m.L.h));x[1]=intersect(x[1],I(0,m.L.h));
                I value=objective(x,p,E);
                if(!point)value.h=std::min(value.h,mixed_envelope(ell,ext,E).h);
                out.add(value);
            }
        }
        if(!point)out.high=std::min(out.high,common_upper(p,E));
        return out;
    }
};

struct Cell {
    Pair p;
    int k,serial;
    double upper;
    bool operator<(const Cell&o)const{
        return upper!=o.upper?upper<o.upper:serial>o.serial;
    }
};
bool better(const Eval&a,const Eval&b){
    if(a.lo>b.lo+2e-8)return true;
    if(a.lo+2e-8<b.lo)return false;
    return a.p<b.p||(a.p==b.p && a.q<b.q);
}
void row(const Eval&a,double ub,double*o){
    double v[]={a.J,a.G,a.x,a.y,a.lv,a.lf,a.wv,a.wf,a.E,a.v,a.f,a.p,a.q,a.lo,a.hi,ub,a.res,(double)a.support};
    for(int i=0;i<18;i++)o[i]=v[i];
}

extern "C" int evaluate(const double*z,int k,double p,double q,double*out){
    try{
        Model model(z);Problem t(model,k);
        if(!t.admissible())return 0;
        Eval a=t.eval({p,q});
        if(a.support<0||!std::isfinite(a.J))return -1;
        Bound b=t.bound({I(p),I(q)},true);
        if(!(b.low<=b.high&&std::isfinite(b.low)&&std::isfinite(b.high)))return -2;
        a.lo=b.low;a.hi=b.high;
        row(a,a.hi,out);
        return 1;
    }catch(...){return -9;}
}

extern "C" int optimize(const double*z,int fixed,double tol,int limit,double*out,double *splits){
    try{
        Model m(z);
        std::vector<Problem> p;
        std::vector<Eval> best;
        std::vector<double> retained;
        std::vector<int> splits_examined;
        std::priority_queue<Cell> pending;
        int serial=0,n_eval=0,n_split=0,joint=-1;
        bool f0=m.P[0].l==m.P[0].h&&m.P[1].l==m.P[1].h;
        auto incumbent=[&](){return joint<0?-inf:best[joint].lo;};
        auto candidate=[&](int k,Vec policy){
            Eval val=p[k].eval(policy);
            n_eval++;
            if(!std::isfinite(val.J)||val.J+1e-5<best[k].lo)return;
            Bound b=p[k].bound({I(policy[0]),I(policy[1])},true);
            if(!(b.low<=b.high&&std::isfinite(b.low)&&std::isfinite(b.high)))return;
            val.lo=b.low;val.hi=b.high;
            if(best[k].support<0||better(val,best[k])){
                best[k]=val;
                if(joint<0||better(best[k],best[joint])||
                    (std::abs(best[k].lo-best[joint].lo)<2e-8&&val.p==best[joint].p
                     &&val.q==best[joint].q&&p[k].k<p[joint].k))joint=k;
            }
        };
        auto add_cell=[&](int k,Pair box,double ub){
            if(ub<=incumbent()+tol)
                retained[k]=std::max(retained[k],ub);
            else pending.push({box,k,serial++,ub});
        };
        for(int k=1;k<m.K;k++){
            if(fixed && k!=fixed)continue;
            Problem t(m,k);
            if(!t.admissible())continue;
            p.push_back(t);best.emplace_back();retained.push_back(-inf);splits_examined.push_back(0);
        }
        if(p.empty())return 0;
        for(size_t k=0;k<p.size();k++){
            Pair box=m.P;
            std::vector<Vec> starts;
            Bound upper=p[k].bound(box,f0,&starts);
            if(f0)candidate(k,{box[0].l,box[1].l});
            else{
                // Nominal grid and pattern search choose candidate prices only.
                // Every retained policy is checked by a point enclosure.
                Vec top{box[0].l,box[1].l};double score=-inf;
                auto nominal=[&](Vec q){
                    Eval e=p[k].eval(q);n_eval++;
                    if(e.J>score||(e.J==score&&q<top)){score=e.J;top=q;return true;}
                    return false;
                };
                int ni=box[0].l==box[0].h?0:8,nj=box[1].l==box[1].h?0:8;
                if(nj==0 && ni)ni=16;
                for(int i=0;i<=ni;i++)for(int j=0;j<=nj;j++)
                    nominal({box[0].l+(box[0].h-box[0].l)*i/std::max(1,ni),
                             box[1].l+(box[1].h-box[1].l)*j/std::max(1,nj)});
                for(const Vec&q:starts)nominal(q);
                double dv=(box[0].h-box[0].l)/std::max(1,ni);
                double df=(box[1].h-box[1].l)/std::max(1,nj);
                if(score>-inf){
                    for(int iter=0;iter<55&&std::max(dv,df)>1e-8;iter++){
                        Vec q=top;
                        bool changed=false;
                        for(int i=-1;i<=1;i++)for(int j=-1;j<=1;j++){
                            if((i==0&&j==0)||(i&&dv==0)||(j&&df==0))continue;
                            changed=nominal({clamp(q[0]+i*dv,box[0].l,box[0].h),
                                             clamp(q[1]+j*df,box[1].l,box[1].h)})||changed;
                        }
                        if(!changed){dv*=.5;df*=.5;}
                    }
                    candidate(k,top);
                }
                // Keep the outer-box cap too, particularly for exact full entry.
                candidate(k,{box[0].h,box[1].h});
            }
            add_cell(k,box,upper.high);
        }
        if(joint<0)return -1;
        while(!pending.empty()&&n_split<limit && pending.top().upper>incumbent()+tol){
            Cell c=pending.top();pending.pop();
            double widths[]={c.p[0].h-c.p[0].l,c.p[1].h-c.p[1].l};
            int j=widths[0]>=widths[1]?0:1;
            double mid=c.p[j].mid();
            if(mid==c.p[j].l||mid==c.p[j].h){
                retained[c.k]=std::max(retained[c.k],c.upper);continue;
            }
            n_split++;splits_examined[c.k]++;
            for(int s=0;s<2;s++){
                Pair child=c.p;
                if(s)child[j].l=mid;else child[j].h=mid;
                Bound b=p[c.k].bound(child);
                double upper=std::min(c.upper,b.high);
                if(upper>incumbent()+tol*.25)
                    candidate(c.k,{child[0].mid(),child[1].mid()});
                add_cell(c.k,child,upper);
            }
        }
        while(!pending.empty()){
            Cell c=pending.top();pending.pop();retained[c.k]=std::max(retained[c.k],c.upper);
        }
        double ub=incumbent();
        for(size_t k=0;k<p.size();k++){
            double top=std::max(retained[k],best[k].hi);
            if(!std::isfinite(best[k].lo))top=retained[k];
            ub=std::max(ub,top);
            row(best[k],top,splits+(p[k].k-1)*20);
            splits[(p[k].k-1)*20+18]=(double)splits_examined[k];
            splits[(p[k].k-1)*20+19]=1.;
        }
        row(best[joint],ub,out);
        out[18]=p[joint].k;out[19]=n_split;out[20]=n_eval;out[21]=(double)p.size();
        return 1;
    }catch(...){return -9;}
}
