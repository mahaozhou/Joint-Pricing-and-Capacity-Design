// Continuous-time FCFS, class-specific three-stage or recurrent pathways.
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <queue>
#include <random>
#include <vector>

struct Patient {
    int route,stage,assessments;
    double start,entered,service,vtime=0,ftime=0,bill=0;
    bool episode_observed,stage_observed;
};
struct Event {
    double t;
    Patient patient;
    bool arrival;
    bool operator<(const Event& other) const {return t>other.t;}
};
struct Totals {
    double n=0,w=0,q=0;
    void add(double wait,double queue){n++;w+=wait;q+=queue;}
};
struct EpisodeTotals {
    double n=0,online=0,offline=0,bills=0;
};

extern "C" int simulate(const double *d,uint64_t seed,double*out){
    // Lambda, x_v,x_f,E, a,b,d,gamma,path,theta,pv,pf,ratev,ratef,kv,kf,
    // warmup,observation,penalty,mandatory-receipt weight.
    double demand=d[0],xv=d[1],xf=d[2],E=d[3],a=d[4],b=d[5],follow_required=d[6],gamma=d[7];
    int path=(int)d[8];
    double theta=d[9],pricev=d[10],pricef=d[11],warmup=d[16],observation=d[17];
    int staff[]={int(d[14]),int(d[15])},busy[]={0,0};
    double rate[]={d[12],d[13]};
    if(path==0)b=0;
    if(path!=3)gamma=0;
    std::mt19937_64 rng(seed);
    auto u=[&](){return std::generate_canonical<double,53>(rng);};
    auto expon=[&](double mu){return -std::log1p(-u())/mu;};
    std::priority_queue<Event> events;
    std::array<std::deque<Patient>,2> queue;
    std::array<Totals,2> pools;
    std::array<Totals,12> by_stage;
    std::array<EpisodeTotals,3> episodes;
    double horizon=warmup+observation,balks=0;
    uint64_t open_episode=0,open_stage=0,iter=0;
    auto begin=[&](Patient patient,int pool,double t){
        busy[pool]++;
        patient.service=t;
        events.push({t+expon(rate[pool]),patient,false});
    };
    auto arrive=[&](Patient patient,int stage,double t){
        int pool=(stage==0||stage==3)?0:1;
        patient.stage=stage;patient.entered=t;
        patient.stage_observed=t>=warmup&&t<horizon;
        if(patient.stage_observed)open_stage++;
        if(busy[pool]<staff[pool])begin(patient,pool,t);
        else queue[pool].push_back(patient);
    };
    events.push({expon(demand+E),{},true});
    while(!events.empty()){
        Event ev=events.top();events.pop();
        if(ev.t>=horizon&&open_stage==0&&open_episode==0)break;
        if(++iter>50000000)return -1;
        if(ev.arrival){
            events.push({ev.t+expon(demand+E),{},true});
            double draw=u()*(demand+E);
            int route=(draw<xv?0:draw<xv+xf?1:draw<demand?3:2);
            if(route==3){
                if(ev.t>=warmup&&ev.t<horizon)balks++;
                continue;
            }
            Patient p{};
            p.route=route;p.start=ev.t;p.episode_observed=ev.t>=warmup&&ev.t<horizon;
            if(p.episode_observed)open_episode++;
            arrive(p,route==0?0:1,ev.t);
            continue;
        }
        Patient p=ev.patient;
        bool online=p.stage==0||p.stage==3;
        int pool=online?0:1;
        busy[pool]--;
        if(!queue[pool].empty()){
            Patient q=queue[pool].front();queue[pool].pop_front();begin(q,pool,ev.t);
        }
        double sojourn=ev.t-p.entered,waiting=p.service-p.entered;
        if(p.stage_observed){
            open_stage--;
            pools[pool].add(sojourn,waiting);
            by_stage[4*p.route+p.stage].add(sojourn,waiting);
        }
        if(online){p.vtime+=sojourn;p.bill+=(p.stage==0?1:theta)*pricev;}
        else {p.ftime+=sojourn;p.bill+=pricef;p.assessments++;}
        int next=-1;
        if(p.stage==0){
            if(u()<a)next=2;
        }else if(p.stage==3){
            if(gamma>0&&u()<gamma*a)next=2;
        }else{
            double probability=p.route==2?follow_required:b;
            if(path==1&&p.route==0&&p.assessments==1)probability=0;
            if(probability>0&&u()<probability)next=3;
        }
        if(next>=0){arrive(p,next,ev.t);continue;}
        if(p.episode_observed){
            open_episode--;
            EpisodeTotals &t=episodes[p.route];
            t.n++;t.online+=p.vtime;t.offline+=p.ftime;t.bills+=p.bill;
        }
    }
    for(const Totals&t:pools){*out++=t.n;*out++=t.w;*out++=t.q;}
    for(const Totals&t:by_stage){*out++=t.n;*out++=t.w;*out++=t.q;}
    for(const EpisodeTotals&t:episodes){*out++=t.n;*out++=t.online;*out++=t.offline;*out++=t.bills;}
    *out++=balks;*out++=(double)iter;
    return 1;
}
