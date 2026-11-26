// Paired DES, C++17 + OpenMP. Time unit: eight-hour clinic-day.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <omp.h>
struct Policy {int kv; double xv,xf,wv,wf,j,de;};
constexpr double L=572,E=30,d1=.35,d2=.25,dE=.25,theta=.5,alpha=.3,V=50,s=15,T=20,cv=112,cf=168,pv=4160./147,pf=40,nu=54,r=10;
constexpr double Bf=pf+theta*d2*pv,Bv=pv+d1*Bf,BE=pf+theta*dE*pv;
struct Patient {double entered,start,service[3],bill; int route,stage; bool observed,ref,follow;};
struct Event {double t; Patient p; bool operator<(const Event&b)const{return t>b.t;}};
struct Result {long long pot=0,mandatory=0,v=0,f=0,balk=0,completedE=0,events=0; double sumE=0,expectedbill=0,realbill=0,initialE=0,followE=0,busytime[2]={0,0}; long long entries[2]={0,0}; int maxN[2]={0,0};};
Result sim(Policy p,int mode,uint64_t seed,double warm,double obs){
 std::mt19937_64 rng(seed); auto u=[&](){return ((rng()>>11)+.5)*0x1.0p-53;}; auto exp=[&](double rate){return -std::log(u())/rate;};
 const int staff[2]={p.kv,18-p.kv}; int busy[2]={0,0}; std::deque<Patient> q[2]; std::priority_queue<Event> heap; Result R; long long open=0; double end=warm+obs;
 auto start=[&](Patient a,int j,double t){++busy[j];heap.push({t+a.service[a.stage],a});};
 auto enter=[&](Patient a,int j,double t){a.entered=t;if(busy[j]<staff[j])start(a,j,t);else q[j].push_back(a);if(t>=warm&&t<end){++R.entries[j];R.maxN[j]=std::max(R.maxN[j],busy[j]+int(q[j].size()));}};
 auto wait=[&](int j){return 1./nu+std::max(0,busy[j]+int(q[j].size())-staff[j]+1)/(staff[j]*nu);};
 double next=exp(L+E),prev=0;
 while(next<end||open>0||(!heap.empty()&&heap.top().t<end)){
  ++R.events;
  double now=heap.empty()?next:std::min(next,heap.top().t);double dt=std::max(0.,std::min(now,end)-std::max(prev,warm));for(int j=0;j<2;j++)R.busytime[j]+=busy[j]*dt;prev=now;
  if(heap.empty()||next<heap.top().t){
   double t=next;next=t+exp(L+E); bool m=u()<E/(L+E);double choice=u(); bool ref=u()<d1;double followdraw=u();
   Patient a{};a.start=t;a.service[0]=exp(nu);a.service[1]=exp(nu);a.service[2]=exp(nu);a.ref=ref;a.follow=followdraw<(m?dE:d2);a.observed=t>=warm&&t<end;
   int route=2;
   if(!m){if(mode==0){route=choice<p.xv/L?0:choice<(p.xv+p.xf)/L?1:3;}else{
     double uv=V+d1*(s-T)-alpha*Bv-cv*(wait(0)+d1*d2*p.wv)-d1*cf*p.wf;
     double uf=V+s-T-alpha*Bf-cf*wait(1)-d2*cv*p.wv;
     double best=std::max(uv,uf); if(best<0)route=3;else if(uv>uf)route=0;else if(uf>uv)route=1;else route=choice<p.xv/L?0:1;
    }}
   if(a.observed){if(m)++R.mandatory;else{++R.pot;if(route==0)++R.v;else if(route==1)++R.f;else ++R.balk;}if(route!=3)R.expectedbill+=(route==0?Bv:route==1?Bf:BE);}
   if(route==3)continue;a.route=route;a.stage=route==0?0:1;if(a.observed)++open;enter(a,route==0?0:1,t);
  }else{
   Event ev=heap.top();heap.pop();Patient a=ev.p;int j=a.stage==1?1:0;--busy[j];if(!q[j].empty()){Patient b=q[j].front();q[j].pop_front();start(b,j,ev.t);}
   a.bill+=a.stage==0?pv:a.stage==1?pf:theta*pv;
   if(a.observed&&a.route==2){if(a.stage==1)R.initialE+=ev.t-a.entered;else R.followE+=ev.t-a.entered;}
   if(a.stage==0&&a.ref){a.stage=1;enter(a,1,ev.t);}else if(a.stage==1&&a.follow){a.stage=2;enter(a,0,ev.t);}else if(a.observed){--open;R.realbill+=a.bill;if(a.route==2){++R.completedE;R.sumE+=ev.t-a.start;}}
  }
  if(R.events>100000000)throw std::runtime_error("event limit");
 }
 if(open||R.mandatory!=R.completedE||R.pot!=R.v+R.f+R.balk)throw std::runtime_error("cohort mismatch");return R;
}
int main(int argc,char**argv){if(argc!=6&&argc!=7){std::cerr<<"usage: simulate source.csv out.csv reps warmup observation [kv]\n";return 1;}std::ifstream in(argv[1]);std::string line;std::getline(in,line);std::vector<Policy> P;while(std::getline(in,line)){std::stringstream ss(line);std::vector<std::string> a;std::string v;while(std::getline(ss,v,','))a.push_back(v);int k=std::stoi(a[0]);if(k>=3&&k<=13&&(argc==6||k==std::stoi(argv[6])))P.push_back({k,std::stod(a[5]),std::stod(a[6]),std::stod(a[7]),std::stod(a[8]),std::stod(a[2]),std::stod(a[11])});}if(P.size()!=(argc==6?11:1))return 2;
 int N=std::stoi(argv[3]);double warm=std::stod(argv[4]),obs=std::stod(argv[5]);std::vector<std::string> rows(P.size()*N*2); double t0=omp_get_wtime(); int done=0;
 #pragma omp parallel for schedule(dynamic)
 for(int idx=0;idx<int(P.size())*N;idx++){int i=idx/N,rep=idx%N;uint64_t seed=202611240000ULL+uint64_t(rep)*1000003ULL;for(int mode=0;mode<2;mode++){auto R=sim(P[i],mode,seed,warm,obs);std::ostringstream out;out<<std::setprecision(15)<<P[i].kv<<','<<rep+1<<','<<seed<<','<<mode<<','<<R.pot<<','<<R.v<<','<<R.f<<','<<R.balk<<','<<R.mandatory<<','<<R.completedE<<','<<R.expectedbill<<','<<R.realbill<<','<<R.sumE<<','<<R.initialE<<','<<R.followE<<','<<R.events<<','<<(R.expectedbill-r*R.balk)/obs<<','<<(R.realbill-r*R.balk)/obs<<','<<100.*(R.v+R.f)/R.pot<<','<<100.*R.v/R.pot<<','<<480.*R.sumE/R.completedE<<','<<R.entries[0]<<','<<R.entries[1]<<','<<R.busytime[0]/(P[i].kv*obs)<<','<<R.busytime[1]/((18-P[i].kv)*obs)<<','<<R.maxN[0]<<','<<R.maxN[1];rows[2*idx+mode]=out.str();}
 #pragma omp critical
 {++done;if(done%100==0)std::cerr<<done<<"/"<<P.size()*N<<" pairs; elapsed "<<omp_get_wtime()-t0<<" s\n";}}
 std::ofstream out(argv[2]);out<<"kv,rep,seed,mode,potential,virtual,inperson,balk,mandatory,completed_mandatory,expected_bill,realized_bill,mandatory_time,mandatory_initial_time,mandatory_followup_time,events,J,J_realized,access_pp,virtual_share_pp,DE_min,entries_v,entries_f,util_v,util_f,max_N_v,max_N_f\n";for(auto&s:rows)out<<s<<'\n';std::cerr<<"elapsed_seconds="<<omp_get_wtime()-t0<<"\n";
}
