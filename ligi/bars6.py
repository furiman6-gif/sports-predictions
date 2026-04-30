import pandas as pd
import math
import os
import glob
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from trueskill import Rating, rate_1vs1, global_env
from openskill.models import (
    PlackettLuce,
    BradleyTerryFull,
    BradleyTerryPart,
    ThurstoneMostellerFull,
    ThurstoneMostellerPart
)
import lightgbm as lgb

HOME_COL = "HomeTeam"
AWAY_COL = "AwayTeam"
DATE_COL = "Date"
CHECKPOINT_EVERY = 200

class WengLinRating:
    def __init__(self, mu=1500.0, sigma=500.0):
        self.mu = mu; self.sigma = sigma
    @property
    def ordinal(self): return self.mu - 3 * self.sigma

class WengLin:
    def __init__(self, mu=1500.0, sigma=500.0, beta=200.0, kappa=0.0001, gamma=5.0):
        self.default_mu=mu; self.default_sigma=sigma; self.beta=beta; self.kappa=kappa; self.gamma=gamma
    def rating(self, mu=None, sigma=None):
        return WengLinRating(mu=mu if mu is not None else self.default_mu, sigma=sigma if sigma is not None else self.default_sigma)
    def _c(self, a, b): return math.sqrt(2*self.beta**2+a.sigma**2+b.sigma**2)
    def _p_win(self, a, b): return 1.0/(1.0+math.exp(-(a.mu-b.mu)/self._c(a,b)))
    def rate_1v1(self, w, l):
        c=self._c(w,l); p=self._p_win(w,l); unc=p*(1-p)
        nwm=w.mu+(w.sigma**2/c)*(1-p); nlm=l.mu-(l.sigma**2/c)*p
        nws=math.sqrt(max((w.sigma*math.sqrt(max(1-(w.sigma**2/c**2)*unc,self.kappa)))**2+self.gamma**2,self.kappa))
        nls=math.sqrt(max((l.sigma*math.sqrt(max(1-(l.sigma**2/c**2)*unc,self.kappa)))**2+self.gamma**2,self.kappa))
        return WengLinRating(nwm,nws), WengLinRating(nlm,nls)
    def rate_draw(self, a, b):
        c=self._c(a,b); p=self._p_win(a,b); unc=p*(1-p)
        nam=a.mu+(a.sigma**2/c)*(0.5-p); nbm=b.mu+(b.sigma**2/c)*(p-0.5)
        nas=math.sqrt(max((a.sigma*math.sqrt(max(1-(a.sigma**2/c**2)*unc*0.5,self.kappa)))**2+self.gamma**2,self.kappa))
        nbs=math.sqrt(max((b.sigma*math.sqrt(max(1-(b.sigma**2/c**2)*unc*0.5,self.kappa)))**2+self.gamma**2,self.kappa))
        return WengLinRating(nam,nas), WengLinRating(nbm,nbs)
    def predict_win(self, a, b): return self._p_win(a,b)
    def predict_draw(self, a, b): p=self._p_win(a,b); return 0.25*(1-abs(2*p-1))

class FootballPageRank:
    def __init__(self, damping=0.85, iterations=100, decay=0.995):
        self.damping=damping; self.iterations=iterations; self.decay=decay
    def _build_matrix(self, matches, mode="standard"):
        teams=sorted(set([m[0] for m in matches]+[m[1] for m in matches]))
        ti={t:i for i,t in enumerate(teams)}; n=len(teams); A=np.zeros((n,n)); tm=len(matches)
        for mn,(h,a,hg,ag,_) in enumerate(matches):
            hi,ai=ti[h],ti[a]
            if mode=="decay": w=self.decay**(tm-mn)
            elif mode=="goal_weighted": w=math.log(abs(hg-ag)+1)+1 if hg!=ag else 0.5
            else: w=1.0
            if hg>ag: A[hi][ai]+=w
            elif ag>hg: A[ai][hi]+=w
            else: A[hi][ai]+=w*0.5; A[ai][hi]+=w*0.5
        cs=A.sum(axis=0); cs[cs==0]=1; return A/cs, teams, ti
    def compute(self, matches, mode="standard"):
        if not matches: return {}
        A,teams,_=self._build_matrix(matches,mode); n=len(teams); r=np.ones(n)/n
        for _ in range(self.iterations):
            rn=(1-self.damping)/n+self.damping*A@r
            if np.abs(rn-r).sum()<1e-10: break
            r=rn
        if r.max()>0: r=(r/r.max())*100
        return {teams[i]:round(r[i],4) for i in range(n)}

class MasseyRating:
    def compute(self, matches):
        if len(matches)<2: return {}
        teams=sorted(set([m[0] for m in matches]+[m[1] for m in matches]))
        ti={t:i for i,t in enumerate(teams)}; n=len(teams)
        if n<2: return {}
        mc=len(matches); M=np.zeros((mc+1,n)); b=np.zeros(mc+1)
        for i,(h,a,hg,ag,_) in enumerate(matches): M[i][ti[h]]=1; M[i][ti[a]]=-1; b[i]=hg-ag
        M[mc]=np.ones(n); b[mc]=0
        try: r,_,_,_=np.linalg.lstsq(M,b,rcond=None); return {teams[i]:round(r[i],4) for i in range(n)}
        except: return {t:0.0 for t in teams}

class ColleyRating:
    def compute(self, matches):
        if len(matches)<2: return {}
        teams=sorted(set([m[0] for m in matches]+[m[1] for m in matches]))
        ti={t:i for i,t in enumerate(teams)}; n=len(teams)
        if n<2: return {}
        C=np.eye(n)*2; bv=np.ones(n); w,l,g=np.zeros(n),np.zeros(n),np.zeros(n)
        for h,a,hg,ag,_ in matches:
            hi,ai=ti[h],ti[a]; C[hi][ai]-=1; C[ai][hi]-=1; g[hi]+=1; g[ai]+=1
            if hg>ag: w[hi]+=1; l[ai]+=1
            elif ag>hg: w[ai]+=1; l[hi]+=1
            else: w[hi]+=0.5; w[ai]+=0.5; l[hi]+=0.5; l[ai]+=0.5
        for i in range(n): C[i][i]+=g[i]; bv[i]+=(w[i]-l[i])/2
        try:
            r=np.linalg.solve(C,bv)
            if r.max()-r.min()>0: r=((r-r.min())/(r.max()-r.min()))*100
            return {teams[i]:round(r[i],4) for i in range(n)}
        except: return {t:50.0 for t in teams}

class KeenerRating:
    def __init__(self, mode="standard", epsilon=1e-6): self.mode=mode; self.epsilon=epsilon
    def _skew(self, x): return 0.5+0.5*math.copysign(math.sqrt(abs(2*x-1)),2*x-1)
    def compute(self, matches):
        if len(matches)<2: return {}
        teams=sorted(set([m[0] for m in matches]+[m[1] for m in matches]))
        ti={t:i for i,t in enumerate(teams)}; n=len(teams)
        if n<2: return {}
        sf,sa,gm=np.zeros((n,n)),np.zeros((n,n)),np.zeros((n,n))
        for h,a,hg,ag,_ in matches:
            hi,ai=ti[h],ti[a]; sf[hi][ai]+=hg; sa[hi][ai]+=ag; sf[ai][hi]+=ag; sa[ai][hi]+=hg; gm[hi][ai]+=1; gm[ai][hi]+=1
        A=np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if i==j: continue
                if gm[i][j]==0: A[i][j]=self.epsilon; continue
                t=sf[i][j]+sa[i][j]; r=(sf[i][j]+1)/(t+2) if t>0 else 0.5
                if self.mode=="nonlinear": r=self._skew(r)
                A[i][j]=r
        A+=self.epsilon*np.ones((n,n)); cs=A.sum(axis=0); cs[cs==0]=1; A=A/cs
        rk=np.ones(n)/n
        for _ in range(1000):
            rn=A@rk; nm=np.linalg.norm(rn,1)
            if nm>0: rn/=nm
            if np.abs(rn-rk).sum()<1e-12: break
            rk=rn
        if rk.max()>0: rk=(rk/rk.max())*100
        return {teams[i]:round(rk[i],4) for i in range(n)}

class MultiDimElo:
    def __init__(self, k_dims=2, base_rating=1500.0, k_factor=32.0, c_lr=0.1, reg=0.001):
        self.k_dims=k_dims; self.base_rating=base_rating; self.k_factor=k_factor; self.c_lr=c_lr; self.reg=reg
        self.ratings={}; self.cvectors={}
    def _get_rating(self, t):
        if t not in self.ratings: self.ratings[t]=self.base_rating; self.cvectors[t]=np.random.randn(self.k_dims)*0.1
        return self.ratings[t], self.cvectors[t]
    def _sigmoid(self, x): return 1.0/(1.0+math.exp(-np.clip(x,-500,500)))
    def _intransitivity_score(self, ca, cb):
        s=0.0
        for i in range(self.k_dims):
            for j in range(i+1,self.k_dims): s+=ca[i]*cb[j]-ca[j]*cb[i]
        return s
    def predict_win(self, ta, tb):
        ra,ca=self._get_rating(ta); rb,cb=self._get_rating(tb)
        return self._sigmoid((ra-rb)/400.0+self._intransitivity_score(ca,cb))
    def update(self, ta, tb, outcome):
        ra,ca=self._get_rating(ta); rb,cb=self._get_rating(tb)
        pa=self._sigmoid((ra-rb)/400.0+self._intransitivity_score(ca,cb)); e=outcome-pa
        self.ratings[ta]=ra+self.k_factor*e; self.ratings[tb]=rb-self.k_factor*e
        ga,gb=np.zeros(self.k_dims),np.zeros(self.k_dims)
        for i in range(self.k_dims):
            for j in range(i+1,self.k_dims): ga[i]+=cb[j]; ga[j]-=cb[i]; gb[j]+=ca[i]; gb[i]-=ca[j]
        sd=pa*(1-pa)
        self.cvectors[ta]=ca+self.c_lr*e*sd*ga-self.reg*ca; self.cvectors[tb]=cb-self.c_lr*e*sd*gb-self.reg*cb

class BARSRating:
    def __init__(self, mu=1500.0, sigma=350.0, k_base=32.0, k_min=8.0, k_max=64.0,
                 sigma_decay=1.005, sigma_min=50.0, sigma_max=500.0, surprise_window=20, streak_threshold=3):
        self.default_mu=mu; self.default_sigma=sigma; self.k_base=k_base; self.k_min=k_min; self.k_max=k_max
        self.sigma_decay=sigma_decay; self.sigma_min=sigma_min; self.sigma_max=sigma_max
        self.surprise_window=surprise_window; self.streak_threshold=streak_threshold
        self.ratings={}; self.sigmas={}; self.k_factors={}; self.games_played={}
        self.last_match={}; self.streaks={}; self.surprises={}; self.results_history={}
    def _init_team(self, t):
        if t not in self.ratings:
            self.ratings[t]=self.default_mu; self.sigmas[t]=self.default_sigma; self.k_factors[t]=self.k_base
            self.games_played[t]=0; self.last_match[t]=0; self.streaks[t]=0; self.surprises[t]=[]; self.results_history[t]=[]
    def _sigmoid(self, x): return 1.0/(1.0+math.exp(-np.clip(x,-500,500)))
    def _compute_adaptive_k(self, t):
        g=self.games_played[t]; nov=max(1.0,4.0/(1.0+g/30.0)); s=self.surprises[t]
        sm=(1.0+sum(s[-self.surprise_window:])/len(s[-self.surprise_window:])*2.0) if len(s)>=5 else 1.5
        st=abs(self.streaks[t]); stm=1.0+(st-self.streak_threshold)*0.15 if st>=self.streak_threshold else 1.0
        return max(self.k_min,min(self.k_max,self.k_base*nov*sm*stm))
    def predict_win(self, ta, tb):
        self._init_team(ta); self._init_team(tb); q=math.log(10)/400
        g=1.0/math.sqrt(1+3*q**2*(self.sigmas[ta]**2+self.sigmas[tb]**2)/math.pi**2)
        return self._sigmoid(g*(self.ratings[ta]-self.ratings[tb])/400.0)
    def predict_draw(self, ta, tb): p=self.predict_win(ta,tb); return 0.25*(1-abs(2*p-1))
    def update(self, ta, tb, outcome, match_idx=0):
        self._init_team(ta); self._init_team(tb)
        for t in [ta,tb]:
            gap=match_idx-self.last_match[t]
            if gap>1: self.sigmas[t]=min(self.sigmas[t]*self.sigma_decay**gap,self.sigma_max)
        pa=self.predict_win(ta,tb); e=outcome-pa; is_s=1.0 if abs(e)>0.35 else 0.0
        for t in [ta,tb]:
            self.surprises[t].append(is_s)
            if len(self.surprises[t])>self.surprise_window*2: self.surprises[t]=self.surprises[t][-self.surprise_window:]
        if outcome>0.6:
            self.streaks[ta]=max(1,self.streaks[ta]+1) if self.streaks[ta]>=0 else 1
            self.streaks[tb]=min(-1,self.streaks[tb]-1) if self.streaks[tb]<=0 else -1
        elif outcome<0.4:
            self.streaks[ta]=min(-1,self.streaks[ta]-1) if self.streaks[ta]<=0 else -1
            self.streaks[tb]=max(1,self.streaks[tb]+1) if self.streaks[tb]>=0 else 1
        else: self.streaks[ta]=0; self.streaks[tb]=0
        ka=self._compute_adaptive_k(ta); kb=self._compute_adaptive_k(tb)
        self.ratings[ta]+=ka*(self.sigmas[ta]/self.default_sigma)*e
        self.ratings[tb]-=kb*(self.sigmas[tb]/self.default_sigma)*e
        ie=abs(e)
        self.sigmas[ta]=max(self.sigma_min,self.sigmas[ta]*(0.995 if ie<0.21 else 1.002))
        self.sigmas[tb]=max(self.sigma_min,self.sigmas[tb]*(0.995 if ie<0.21 else 1.002))
        self.games_played[ta]+=1; self.games_played[tb]+=1
        self.last_match[ta]=match_idx; self.last_match[tb]=match_idx
        self.results_history[ta].append(outcome); self.results_history[tb].append(1.0-outcome)
        if len(self.results_history[ta])>50: self.results_history[ta]=self.results_history[ta][-50:]
        if len(self.results_history[tb])>50: self.results_history[tb]=self.results_history[tb][-50:]
        self.k_factors[ta]=ka; self.k_factors[tb]=kb
    def get_posterior_skew(self, t):
        self._init_team(t); h=self.results_history[t]
        if len(h)<5: return 0.0
        a=np.array(h[-20:])
        if a.std()==0: return 0.0
        return float(np.clip(((a-a.mean())**3).mean()/(a.std()**3),-3,3))
    def get_posterior_kurtosis(self, t):
        self._init_team(t); h=self.results_history[t]
        if len(h)<5: return 0.0
        a=np.array(h[-20:])
        if a.std()==0: return 0.0
        return float(np.clip(((a-a.mean())**4).mean()/(a.std()**4)-3,-3,6))

class EloDavidson:
    def __init__(self, base_rating=1500.0, k_factor=32.0, nu=0.3, auto_nu=False, home_advantage=0.0):
        self.base_rating=base_rating; self.k_factor=k_factor; self.nu=nu; self.auto_nu=auto_nu
        self.home_adv=home_advantage; self.ratings={}; self.total_matches=0; self.total_draws=0
    def _get_rating(self, t):
        if t not in self.ratings: self.ratings[t]=self.base_rating
        return self.ratings[t]
    def _gamma(self, r): return 10.0**(r/400.0)
    def predict_probs(self, ta, tb, ih=False):
        ra=self._get_rating(ta)+(self.home_adv if ih else 0); rb=self._get_rating(tb)
        ga,gb=self._gamma(ra),self._gamma(rb); sg=math.sqrt(ga*gb); d=ga+gb+self.nu*sg
        if d==0: return (1/3,1/3,1/3)
        return (ga/d, self.nu*sg/d, gb/d)
    def predict_win(self, ta, tb, ih=False): return self.predict_probs(ta,tb,ih)[0]
    def predict_draw(self, ta, tb, ih=False): return self.predict_probs(ta,tb,ih)[1]
    def update(self, ta, tb, outcome, is_home_a=False):
        ra=self._get_rating(ta); rb=self._get_rating(tb)
        pw,pd,pl=self.predict_probs(ta,tb,is_home_a)
        self.ratings[ta]=ra+self.k_factor*(outcome-(pw+0.5*pd))
        self.ratings[tb]=rb+self.k_factor*((1.0-outcome)-(pl+0.5*pd))
        if self.auto_nu:
            self.total_matches+=1
            if abs(outcome-0.5)<0.01: self.total_draws+=1
            if self.total_matches>=50:
                odr=self.total_draws/self.total_matches
                if odr<0.95: self.nu=max(0.01,min(2.0,self.nu*0.995+(2.0*odr/(1.0-odr))*0.005))

class TeamEmbeddingNet(nn.Module):
    def __init__(self, num_teams, embed_dim=16, hidden_dim=32):
        super().__init__(); self.embed_dim=embed_dim
        self.team_embedding=nn.Embedding(num_teams, embed_dim)
        nn.init.xavier_uniform_(self.team_embedding.weight)
        self.classifier=nn.Sequential(nn.Linear(embed_dim*2,hidden_dim),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(hidden_dim,hidden_dim//2),nn.ReLU(),nn.Dropout(0.1),nn.Linear(hidden_dim//2,3))
    def forward(self, home_ids, away_ids):
        eh=self.team_embedding(home_ids); ea=self.team_embedding(away_ids)
        return self.classifier(torch.cat([torch.abs(eh-ea),eh*ea],dim=1))
    def get_embedding(self, team_id):
        with torch.no_grad(): return self.team_embedding(torch.tensor([team_id])).squeeze().numpy()

class SiameseEmbeddingSystem:
    def __init__(self, embed_dim=16, hidden_dim=32, retrain_every=200, train_window=2000, epochs=30, lr=0.005, batch_size=64):
        self.embed_dim=embed_dim; self.hidden_dim=hidden_dim; self.retrain_every=retrain_every
        self.train_window=train_window; self.epochs=epochs; self.lr=lr; self.batch_size=batch_size
        self.team_to_idx={}; self.model=None; self.match_history=[]; self.current_embeddings={}
    def _get_team_idx(self, t):
        if t not in self.team_to_idx: self.team_to_idx[t]=len(self.team_to_idx)
        return self.team_to_idx[t]
    def add_match(self, h, a, target): self.match_history.append((self._get_team_idx(h),self._get_team_idx(a),target))
    def _train(self):
        nt=len(self.team_to_idx)
        if nt<4: return
        self.model=TeamEmbeddingNet(nt,self.embed_dim,self.hidden_dim)
        opt=optim.Adam(self.model.parameters(),lr=self.lr,weight_decay=1e-4); crit=nn.CrossEntropyLoss()
        data=self.match_history[-self.train_window:]
        if len(data)<20: return
        hi=torch.tensor([d[0] for d in data],dtype=torch.long)
        ai=torch.tensor([d[1] for d in data],dtype=torch.long)
        tg=torch.tensor([d[2] for d in data],dtype=torch.long)
        ds=torch.utils.data.TensorDataset(hi,ai,tg); ld=torch.utils.data.DataLoader(ds,batch_size=self.batch_size,shuffle=True)
        self.model.train()
        for _ in range(self.epochs):
            for bh,ba,bt in ld: opt.zero_grad(); loss=crit(self.model(bh,ba),bt); loss.backward(); opt.step()
        self.model.eval(); itm={v:k for k,v in self.team_to_idx.items()}
        for i in range(nt): self.current_embeddings[itm[i]]=self.model.get_embedding(i)
    def should_retrain(self): return len(self.match_history)%self.retrain_every==0 and len(self.match_history)>=50
    def get_embedding_diff(self, h, a):
        d=np.zeros(self.embed_dim); return self.current_embeddings.get(h,d)-self.current_embeddings.get(a,d)
    def get_cosine_similarity(self, h, a):
        d=np.zeros(self.embed_dim); eh=self.current_embeddings.get(h,d); ea=self.current_embeddings.get(a,d)
        nh,na=np.linalg.norm(eh),np.linalg.norm(ea)
        return float(np.dot(eh,ea)/(nh*na)) if nh>0 and na>0 else 0.0
    def get_euclidean_distance(self, h, a):
        d=np.zeros(self.embed_dim); return float(np.linalg.norm(self.current_embeddings.get(h,d)-self.current_embeddings.get(a,d)))

class LambdaMARTRanking:
    def __init__(self, retrain_every=200, history_window=200, num_leaves=63, n_estimators=200, learning_rate=0.05):
        self.retrain_every=retrain_every; self.history_window=history_window
        self.num_leaves=num_leaves; self.n_estimators=n_estimators; self.learning_rate=learning_rate
        self.team_history={}; self.current_rankings={}; self.model=None; self.match_count=0
    def _init_team(self, t):
        if t not in self.team_history: self.team_history[t]=[]
    def _build_team_features(self, team):
        hist=self.team_history.get(team,[])
        if len(hist)==0: return np.zeros(12)
        recent=hist[-self.history_window:]; total=len(recent)
        wins=sum(1 for _,_,_,_,r in recent if r==1.0); draws=sum(1 for _,_,_,_,r in recent if r==0.5)
        gf=sum(g for _,g,_,_,_ in recent); ga=sum(g for _,_,g,_,_ in recent)
        hg=[(g,ga_,r) for _,g,ga_,hf,r in recent if hf==1]; ag=[(g,ga_,r) for _,g,ga_,hf,r in recent if hf==0]
        hwr=sum(1 for _,_,r in hg if r==1.0)/max(1,len(hg)); awr=sum(1 for _,_,r in ag if r==1.0)/max(1,len(ag))
        streak=0
        for _,_,_,_,r in reversed(recent):
            if r==1.0: streak+=1
            elif r==0.0: streak-=1
            else: break
            if abs(streak)>=5: break
        last5=recent[-5:] if len(recent)>=5 else recent; form=sum(r for _,_,_,_,r in last5)/max(1,len(last5))
        owr=[]
        for opp,_,_,_,_ in recent[-20:]:
            oh=self.team_history.get(opp,[])
            if len(oh)>0: owr.append(sum(1 for _,_,_,_,r in oh if r==1.0)/len(oh))
        sos=np.mean(owr) if owr else 0.5
        return np.array([wins/max(1,total),draws/max(1,total),gf/max(1,total),ga/max(1,total),
            (gf-ga)/max(1,total),hwr,awr,streak,form,sos,total,math.log(total+1)])
    def _train(self):
        teams=[t for t,h in self.team_history.items() if len(h)>=5]
        if len(teams)<6: return
        features=[]; labels=[]
        for team in teams:
            feat=self._build_team_features(team); hist=self.team_history[team]
            wr=sum(1 for _,_,_,_,r in hist if r==1.0)/len(hist)
            gd=sum(gf-ga for _,gf,ga,_,_ in hist)/len(hist)
            relevance=wr*70+max(0,gd)*10+10; features.append(feat); labels.append(relevance)
        X=np.array(features); y=np.array(labels)
        ymin=y.min(); ymax=y.max()
        if ymax-ymin>0: y=((y-ymin)/(ymax-ymin)*30).astype(int)
        else: y=np.full_like(y,15,dtype=int)
        y=np.clip(y,0,30); group=[len(teams)]
        train_data=lgb.Dataset(X, label=y, group=group)
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "num_leaves": self.num_leaves,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": -1,
            "min_data_in_leaf": 2,
            "min_sum_hessian_in_leaf": 1e-6,
            "min_gain_to_split": 0.0,
            "min_data_per_group": 1,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "lambda_l2": 1e-3,
            "extra_trees": True,
            "feature_pre_filter": False,
            "force_col_wise": True,
            "verbosity": -1,
        }
        self.model = lgb.train(params, train_data)
        self.current_rankings={}
        scores=self.model.predict(X)
        for t,s in zip(teams,scores): self.current_rankings[t]=float(s)
    def add_match(self, home, away, hg, ag, outcome):
        self._init_team(home); self._init_team(away)
        self.team_history[home].append((away,hg,ag,1,outcome)); self.team_history[away].append((home,ag,hg,0,1.0-outcome if outcome!=0.5 else 0.5))
        for t in [home,away]:
            if len(self.team_history[t])>600: self.team_history[t]=self.team_history[t][-600:]
        self.match_count+=1
    def should_retrain(self): return self.match_count%self.retrain_every==0 and self.match_count>=50
    def get_ranking_diff(self, h, a): return self.current_rankings.get(h,0.0)-self.current_rankings.get(a,0.0)
    def get_team_features_diff(self, h, a): return self._build_team_features(h)-self._build_team_features(a)

class RankNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.scorer=nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(hidden_dim,hidden_dim//2),nn.ReLU(),nn.Dropout(0.1),nn.Linear(hidden_dim//2,1))
    def forward(self, x): return self.scorer(x).squeeze(-1)
    def predict_pair(self, xa, xb): return torch.sigmoid(self.forward(xa)-self.forward(xb))

class RankNetSystem:
    def __init__(self, hidden_dim=64, retrain_every=200, train_window=2000, epochs=50, lr=0.003, batch_size=64):
        self.hidden_dim=hidden_dim; self.retrain_every=retrain_every; self.train_window=train_window
        self.epochs=epochs; self.lr=lr; self.batch_size=batch_size; self.n_features=15
        self.team_history={}; self.match_pairs=[]; self.model=None; self.current_scores={}; self.match_count=0
    def _init_team(self, t):
        if t not in self.team_history: self.team_history[t]=[]
    def _build_team_features(self, team):
        hist=self.team_history.get(team,[])
        if len(hist)==0: return np.zeros(self.n_features,dtype=np.float32)
        recent=hist[-200:]; total=len(recent)
        wins=sum(1 for _,_,_,_,r in recent if r==1.0); draws=sum(1 for _,_,_,_,r in recent if r==0.5)
        losses=sum(1 for _,_,_,_,r in recent if r==0.0)
        gf=sum(g for _,g,_,_,_ in recent); ga=sum(g for _,_,g,_,_ in recent)
        hg=[(g,ga_,r) for _,g,ga_,hf,r in recent if hf==1]; ag=[(g,ga_,r) for _,g,ga_,hf,r in recent if hf==0]
        hwr=sum(1 for _,_,r in hg if r==1.0)/max(1,len(hg)); awr=sum(1 for _,_,r in ag if r==1.0)/max(1,len(ag))
        streak=0
        for _,_,_,_,r in reversed(recent):
            if r==1.0: streak+=1
            elif r==0.0: streak-=1
            else: break
            if abs(streak)>=5: break
        last5=recent[-5:] if len(recent)>=5 else recent; last10=recent[-10:] if len(recent)>=10 else recent
        form5=sum(r for _,_,_,_,r in last5)/max(1,len(last5)); form10=sum(r for _,_,_,_,r in last10)/max(1,len(last10))
        owr=[]
        for opp,_,_,_,_ in recent[-20:]:
            oh=self.team_history.get(opp,[])
            if len(oh)>0: owr.append(sum(1 for _,_,_,_,r in oh if r==1.0)/len(oh))
        sos=np.mean(owr) if owr else 0.5
        cs=sum(1 for _,_,ga_,_,_ in recent if ga_==0)/max(1,total)
        return np.array([wins/max(1,total),draws/max(1,total),losses/max(1,total),gf/max(1,total),ga/max(1,total),
            (gf-ga)/max(1,total),hwr,awr,streak/5.0,form5,form10,sos,min(total,200)/200.0,
            math.log(total+1)/6.0,cs],dtype=np.float32)
    def _train(self):
        if len(self.match_pairs)<50: return
        pairs=self.match_pairs[-self.train_window:]
        fh_l,fa_l,labels=[],[],[]
        for h,a,o in pairs: fh_l.append(self._build_team_features(h)); fa_l.append(self._build_team_features(a)); labels.append(o)
        Xh=torch.tensor(np.array(fh_l),dtype=torch.float32); Xa=torch.tensor(np.array(fa_l),dtype=torch.float32)
        Y=torch.tensor(labels,dtype=torch.float32)
        self.model=RankNetModel(self.n_features,self.hidden_dim)
        opt=optim.Adam(self.model.parameters(),lr=self.lr,weight_decay=1e-4)
        ds=torch.utils.data.TensorDataset(Xh,Xa,Y); ld=torch.utils.data.DataLoader(ds,batch_size=self.batch_size,shuffle=True)
        self.model.train()
        for _ in range(self.epochs):
            for bh,ba,bt in ld:
                opt.zero_grad(); p=self.model.predict_pair(bh,ba)
                loss=nn.functional.binary_cross_entropy(p,bt); loss.backward(); opt.step()
        self.model.eval()
        with torch.no_grad():
            for team in self.team_history:
                f=torch.tensor(self._build_team_features(team),dtype=torch.float32).unsqueeze(0)
                self.current_scores[team]=self.model(f).item()
    def add_match(self, home, away, hg, ag, outcome):
        self._init_team(home); self._init_team(away)
        oa=1.0-outcome if outcome!=0.5 else 0.5
        self.team_history[home].append((away,hg,ag,1,outcome)); self.team_history[away].append((home,ag,hg,0,oa))
        for t in [home,away]:
            if len(self.team_history[t])>600: self.team_history[t]=self.team_history[t][-600:]
        self.match_pairs.append((home,away,outcome))
        if len(self.match_pairs)>self.train_window*2: self.match_pairs=self.match_pairs[-self.train_window:]
        self.match_count+=1
    def should_retrain(self): return self.match_count%self.retrain_every==0 and self.match_count>=50
    def get_score_diff(self, h, a): return self.current_scores.get(h,0.0)-self.current_scores.get(a,0.0)
    def get_p_home_win(self, h, a):
        if self.model is None: return 0.5
        self.model.eval()
        with torch.no_grad():
            fh=torch.tensor(self._build_team_features(h),dtype=torch.float32).unsqueeze(0)
            fa=torch.tensor(self._build_team_features(a),dtype=torch.float32).unsqueeze(0)
            return self.model.predict_pair(fh,fa).item()
    def get_features_diff(self, h, a): return self._build_team_features(h)-self._build_team_features(a)

class ListNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )
    def forward(self, x):
        return self.scorer(x).squeeze(-1)

class ListMLEModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )
    def forward(self, x):
        return self.scorer(x).squeeze(-1)

class ListwiseRankingSystem:
    def __init__(self, mode="listnet", hidden_dim=64, retrain_every=200,
                 train_window=2000, epochs=80, lr=0.003, batch_size=1):
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.retrain_every = retrain_every
        self.train_window = train_window
        self.epochs = epochs
        self.lr = lr
        self.n_features = 16
        self.team_history = {}
        self.match_log = []
        self.model = None
        self.current_scores = {}
        self.match_count = 0
    def _init_team(self, team):
        if team not in self.team_history:
            self.team_history[team] = []
    def _build_team_features(self, team):
        hist = self.team_history.get(team, [])
        if len(hist) == 0:
            return np.zeros(self.n_features, dtype=np.float32)
        recent = hist[-200:]
        total = len(recent)
        wins = sum(1 for _, _, _, _, r in recent if r == 1.0)
        draws = sum(1 for _, _, _, _, r in recent if r == 0.5)
        losses = sum(1 for _, _, _, _, r in recent if r == 0.0)
        gf = sum(g for _, g, _, _, _ in recent)
        ga = sum(g for _, _, g, _, _ in recent)
        home_games = [(g, ga_, r) for _, g, ga_, hf, r in recent if hf == 1]
        away_games = [(g, ga_, r) for _, g, ga_, hf, r in recent if hf == 0]
        home_wr = sum(1 for _, _, r in home_games if r == 1.0) / max(1, len(home_games))
        away_wr = sum(1 for _, _, r in away_games if r == 1.0) / max(1, len(away_games))
        streak = 0
        for _, _, _, _, r in reversed(recent):
            if r == 1.0: streak += 1
            elif r == 0.0: streak -= 1
            else: break
            if abs(streak) >= 5: break
        last5 = recent[-5:] if len(recent) >= 5 else recent
        last10 = recent[-10:] if len(recent) >= 10 else recent
        form5 = sum(r for _, _, _, _, r in last5) / max(1, len(last5))
        form10 = sum(r for _, _, _, _, r in last10) / max(1, len(last10))
        owr = []
        for opp, _, _, _, _ in recent[-20:]:
            oh = self.team_history.get(opp, [])
            if len(oh) > 0:
                owr.append(sum(1 for _, _, _, _, r in oh if r == 1.0) / len(oh))
        sos = np.mean(owr) if owr else 0.5
        cs = sum(1 for _, _, ga_, _, _ in recent if ga_ == 0) / max(1, total)
        ppg = (wins * 3 + draws * 1) / max(1, total)
        return np.array([
            wins / max(1, total),
            draws / max(1, total),
            losses / max(1, total),
            gf / max(1, total),
            ga / max(1, total),
            (gf - ga) / max(1, total),
            home_wr,
            away_wr,
            streak / 5.0,
            form5,
            form10,
            sos,
            min(total, 200) / 200.0,
            math.log(total + 1) / 6.0,
            cs,
            ppg / 3.0,
        ], dtype=np.float32)
    def _compute_relevance(self, team):
        hist = self.team_history.get(team, [])
        if len(hist) == 0:
            return 0.0
        total = len(hist)
        wins = sum(1 for _, _, _, _, r in hist if r == 1.0)
        draws = sum(1 for _, _, _, _, r in hist if r == 0.5)
        gf = sum(g for _, g, _, _, _ in hist)
        ga = sum(g for _, _, g, _, _ in hist)
        ppg = (wins * 3 + draws * 1) / max(1, total)
        gd = (gf - ga) / max(1, total)
        return ppg + gd * 0.1
    def _listnet_loss(self, predicted_scores, true_relevances):
        p_true = torch.softmax(true_relevances, dim=0)
        log_p_pred = torch.log_softmax(predicted_scores, dim=0)
        return -torch.sum(p_true * log_p_pred)
    def _listmle_loss(self, predicted_scores, true_relevances):
        sorted_indices = torch.argsort(true_relevances, descending=True)
        sorted_scores = predicted_scores[sorted_indices]
        n = len(sorted_scores)
        loss = 0.0
        for i in range(n):
            remaining = sorted_scores[i:]
            log_sum_exp = torch.logsumexp(remaining, dim=0)
            loss += log_sum_exp - sorted_scores[i]
        return loss / n
    def _train(self):
        teams = [t for t, h in self.team_history.items() if len(h) >= 5]
        if len(teams) < 6:
            return
        features_list = []
        relevances_list = []
        for team in teams:
            feat = self._build_team_features(team)
            rel = self._compute_relevance(team)
            features_list.append(feat)
            relevances_list.append(rel)
        X = torch.tensor(np.array(features_list), dtype=torch.float32)
        Y = torch.tensor(relevances_list, dtype=torch.float32)
        if self.mode == "listmle":
            self.model = ListMLEModel(self.n_features, self.hidden_dim)
        else:
            self.model = ListNetModel(self.n_features, self.hidden_dim)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            if len(teams) > 20:
                indices = np.random.choice(len(teams), size=min(20, len(teams)), replace=False)
                X_batch = X[indices]; Y_batch = Y[indices]
            else:
                X_batch = X; Y_batch = Y
            predicted_scores = self.model(X_batch)
            loss = self._listmle_loss(predicted_scores, Y_batch) if self.mode=="listmle" else self._listnet_loss(predicted_scores, Y_batch)
            loss.backward(); optimizer.step()
        self.model.eval()
        with torch.no_grad():
            all_scores = self.model(X)
            for team, score in zip(teams, all_scores.numpy()):
                self.current_scores[team] = float(score)
    def add_match(self, home, away, hg, ag, outcome):
        self._init_team(home); self._init_team(away)
        outcome_away = 1.0 - outcome if outcome != 0.5 else 0.5
        self.team_history[home].append((away, hg, ag, 1, outcome))
        self.team_history[away].append((home, ag, hg, 0, outcome_away))
        if len(self.team_history[home])>600: self.team_history[home]=self.team_history[home][-600:]
        if len(self.team_history[away])>600: self.team_history[away]=self.team_history[away][-600:]
        self.match_count += 1
    def should_retrain(self):
        return self.match_count % self.retrain_every == 0 and self.match_count >= 50
    def get_score_diff(self, home, away):
        return self.current_scores.get(home, 0.0) - self.current_scores.get(away, 0.0)
    def get_rank_position_diff(self, home, away):
        if not self.current_scores: return 0.0
        sorted_teams = sorted(self.current_scores.items(), key=lambda x: x[1], reverse=True)
        positions = {t: i + 1 for i, (t, _) in enumerate(sorted_teams)}
        pos_h = positions.get(home, len(sorted_teams) // 2)
        pos_a = positions.get(away, len(sorted_teams) // 2)
        return float(pos_a - pos_h)
    def get_percentile_diff(self, home, away):
        if not self.current_scores: return 0.0
        scores = list(self.current_scores.values())
        if len(scores) < 2: return 0.0
        s_h = self.current_scores.get(home, np.median(scores))
        s_a = self.current_scores.get(away, np.median(scores))
        pct_h = (sum(1 for s in scores if s <= s_h) / len(scores)) * 100
        pct_a = (sum(1 for s in scores if s <= s_a) / len(scores)) * 100
        return pct_h - pct_a
    def get_features_diff(self, home, away):
        return self._build_team_features(home) - self._build_team_features(away)

LM_FEAT_NAMES = ["wr","dr","gpg","gcpg","gdpg","home_wr","away_wr","streak","form","sos","games","log_games"]
RN_FEAT_NAMES = ["wr","dr","lr","gpg","gcpg","gdpg","home_wr","away_wr","streak","form5","form10","sos","games","log_games","cs"]
LW_FEAT_NAMES = ["wr","dr","lr","gpg","gcpg","gdpg","home_wr","away_wr","streak","form5","form10","sos","games","log_games","cs","ppg"]

class RLHFRewardModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

class RLHFSystem:
    def __init__(self, feature_dim=15, hidden_dim=64,
                 retrain_every=200, train_window=2000,
                 epochs=60, lr=0.003):
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.retrain_every = retrain_every
        self.train_window = train_window
        self.epochs = epochs
        self.lr = lr
        self.team_history = {}
        self.pairs = []
        self.model = None
        self.current_rewards = {}
        self.match_count = 0
    def _init_team(self, team):
        if team not in self.team_history:
            self.team_history[team] = []
    def _build_features(self, team):
        hist = self.team_history.get(team, [])
        if len(hist) == 0:
            return np.zeros(self.feature_dim, dtype=np.float32)
        recent = hist[-200:]
        total = len(recent)
        wins = sum(1 for _,_,_,_,r in recent if r==1.0)
        draws = sum(1 for _,_,_,_,r in recent if r==0.5)
        losses = sum(1 for _,_,_,_,r in recent if r==0.0)
        gf = sum(g for _,g,_,_,_ in recent)
        ga = sum(g for _,_,g,_,_ in recent)
        return np.array([
            wins/total,
            draws/total,
            losses/total,
            gf/total,
            ga/total,
            (gf-ga)/total,
            math.log(total+1),
            total/200.0,
            wins*3 + draws,
            wins,
            losses,
            gf,
            ga,
            wins/(losses+1),
            (wins*3 + draws)/(total+1)
        ], dtype=np.float32)
    def _train(self):
        if len(self.pairs) < 50: return
        data = self.pairs[-self.train_window:]
        X_a = []; X_b = []; Y = []
        for home, away, outcome in data:
            X_a.append(self._build_features(home))
            X_b.append(self._build_features(away))
            Y.append(outcome)
        X_a = torch.tensor(np.array(X_a), dtype=torch.float32)
        X_b = torch.tensor(np.array(X_b), dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)
        self.model = RLHFRewardModel(self.feature_dim, self.hidden_dim)
        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            R_a = self.model(X_a)
            R_b = self.model(X_b)
            p = torch.sigmoid(R_a - R_b)
            loss = nn.functional.binary_cross_entropy(p, Y)
            loss.backward()
            opt.step()
        self.model.eval()
        with torch.no_grad():
            for team in self.team_history:
                f = torch.tensor(self._build_features(team), dtype=torch.float32).unsqueeze(0)
                self.current_rewards[team] = self.model(f).item()
    def add_match(self, home, away, hg, ag, outcome):
        self._init_team(home); self._init_team(away)
        outcome_away = 1.0 - outcome if outcome!=0.5 else 0.5
        self.team_history[home].append((away,hg,ag,1,outcome))
        self.team_history[away].append((home,ag,hg,0,outcome_away))
        self.pairs.append((home,away,outcome))
        self.match_count += 1
        if self.match_count % self.retrain_every == 0:
            print("   [RLHF] retraining...")
            self._train()
    def get_reward_diff(self, home, away):
        return self.current_rewards.get(home,0.0) - self.current_rewards.get(away,0.0)
    def get_p_home_win(self, home, away):
        diff = self.get_reward_diff(home, away)
        return 1.0 / (1.0 + math.exp(-diff))
    def get_percentile_diff(self, home, away):
        scores = list(self.current_rewards.values())
        if len(scores) < 5:
            return 0.0
        s_h = self.current_rewards.get(home,0.0)
        s_a = self.current_rewards.get(away,0.0)
        pct_h = sum(1 for s in scores if s <= s_h) / len(scores)
        pct_a = sum(1 for s in scores if s <= s_a) / len(scores)
        return pct_h - pct_a

def _state_file_path(file_path):
    prefix_name = os.path.basename(file_path).replace("_Teams_Full_Calendar.csv", "").replace('.csv','')
    dir_path = os.path.dirname(os.path.abspath(file_path))
    return os.path.join(dir_path, f"{prefix_name}_bars6_state.pkl")


def _save_state(state_path, state_obj):
    tmp = state_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(state_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, state_path)


def process_file(file_path):
    print(f"\n{'='*60}")
    print(f"  ROZPOCZYNAM PRZETWARZANIE: {file_path}")
    print(f"{'='*60}")
    prefix_name = os.path.basename(file_path).replace("_Teams_Full_Calendar.csv", "").replace('.csv','')
    dir_path = os.path.dirname(os.path.abspath(file_path))
    output_file = os.path.join(dir_path, f"{prefix_name}_bars6_Cechy_Rankingowe.csv")
    state_file = _state_file_path(file_path)
    df = pd.read_csv(file_path, low_memory=False, on_bad_lines="warn")

    # Upewnij się, że mamy kolumny z datą i nazwami drużyn (różne źródła mogą mieć inne nazwy)
    if DATE_COL not in df.columns:
        date_cands = [c for c in df.columns if 'date' in c.lower()]
        if date_cands:
            df.rename(columns={date_cands[0]: DATE_COL}, inplace=True)
        else:
            print("  BLAD: brakuje kolumny daty."); return
    if HOME_COL not in df.columns or AWAY_COL not in df.columns:
        home_cand = next((c for c in df.columns if 'home' in c.lower() and 'team' in c.lower()), None)
        away_cand = next((c for c in df.columns if 'away' in c.lower() and 'team' in c.lower()), None)
        if home_cand and away_cand:
            df.rename(columns={home_cand: HOME_COL, away_cand: AWAY_COL}, inplace=True)
        else:
            home_cand = next((c for c in df.columns if c.lower() == 'home'), None)
            away_cand = next((c for c in df.columns if c.lower() == 'away'), None)
            if home_cand and away_cand:
                df.rename(columns={home_cand: HOME_COL, away_cand: AWAY_COL}, inplace=True)
            else:
                print("  BLAD: brakuje kolumn z nazwami drużyn."); return

    # Wyszukaj kolumny z golami (obsługa różnych nazw: FTHG/FTAG, score_home/score_away, itp.)
    cols_lower = [c.lower() for c in df.columns]
    score_home_col = None
    score_away_col = None
    pairs = [('fthg','ftag'), ('home_score','away_score'), ('score_home','score_away'), ('homegoals','awaygoals'), ('hg','ag')]
    for a,b in pairs:
        if a in cols_lower and b in cols_lower:
            score_home_col = df.columns[cols_lower.index(a)]
            score_away_col = df.columns[cols_lower.index(b)]
            break
    if score_home_col is None:
        for c in df.columns:
            lc=c.lower()
            if 'fthg' in lc or 'fulltime_home' in lc or 'home_goals' in lc:
                score_home_col=c; break
    if score_away_col is None:
        for c in df.columns:
            lc=c.lower()
            if 'ftag' in lc or 'fulltime_away' in lc or 'away_goals' in lc:
                score_away_col=c; break
    if not score_home_col or not score_away_col:
        print(f"  BLAD: brakuje kolumn z golami. Dostępne kolumny: {list(df.columns)}"); return
    def get_target(row):
        h=pd.to_numeric(row[score_home_col],errors='coerce'); a=pd.to_numeric(row[score_away_col],errors='coerce')
        if pd.isna(h) or pd.isna(a): return None
        if h>a: return 0
        elif h<a: return 2
        else: return 1
    df = df.copy(); df['target'] = df.apply(get_target, axis=1)
    def get_result(row):
        h=pd.to_numeric(row[score_home_col],errors='coerce'); a=pd.to_numeric(row[score_away_col],errors='coerce')
        if pd.isna(h) or pd.isna(a): return ""
        if h>a: return "W"
        elif h<a: return "L"
        else: return "D"
    df['result'] = df.apply(get_result, axis=1)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="%d/%m/%Y", errors="coerce")
    # Przyszłe mecze (bez wyniku) pozostają – trafią na koniec po sortowaniu
    df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    n_future = df['target'].isna().sum()
    if n_future > 0:
        print(f"  Znaleziono {n_future} przyszlych meczow (bez wyniku) - rankingi pre-match")
    source_mtime_ns = os.stat(file_path).st_mtime_ns
    source_rows = len(df)
    ts_env=global_env(); ts_ratings={}
    # ODCHUDZONE: tylko Siamese - reszta ratingow wylaczona (formy/bramki/xg liczone w gbm4)
    os_models={}
    os_ratings={}
    wl_models={}
    wl_ratings={}
    pr=FootballPageRank(); massey=MasseyRating(); colley=ColleyRating()
    keener_models={}
    melo_models={}
    bars_models={}
    ed_models={}
    siamese_models={}
    lm_models={}
    rn_models={}
    lw_models = {}
    rlhf_models = {}
    GRAPH_RECALC_EVERY=50; GRAPH_WINDOW=500
    pr_sc,pr_gc,pr_dc,mas_c,col_c={},{},{},{},{}
    keen_c={p:{} for p in keener_models}; all_graph=[]
    fd={}
    SLIM_MODE = True  # tylko Siamese, bez TS/PR/Massey/Colley/etc.
    if not SLIM_MODE:
        for s in ["mu_diff","sigma_diff","ordinal_diff","certainty_diff","home_win_prob"]: fd[f"TS_{s}"]=[]
    for p in os_models:
        for s in ["mu_diff","sigma_diff","ordinal_diff","certainty_diff"]: fd[f"{p}_{s}"]=[]
    for p in wl_models:
        for s in ["mu_diff","sigma_diff","ordinal_diff","certainty_diff","p_home_win","p_draw"]: fd[f"{p}_{s}"]=[]
    for s in ["PR_diff","PR_GOL_diff","PR_DEC_diff","MASSEY_diff","COLLEY_diff"]: fd[s]=[]
    for p in keener_models: fd[f"{p}_diff"]=[]
    for p in melo_models:
        for s in ["rating_diff","p_home_win","intrans","c_norm_diff"]: fd[f"{p}_{s}"]=[]
    for p in bars_models:
        for s in ["mu_diff","sigma_diff","ordinal_diff","k_factor_diff","p_home_win","p_draw",
                   "streak_diff","surprise_diff","skew_diff","kurtosis_diff"]: fd[f"{p}_{s}"]=[]
    for p in ed_models:
        for s in ["rating_diff","p_home_win","p_draw","p_away_win","draw_tendency","expected_score_diff"]: fd[f"{p}_{s}"]=[]
    for prefix,siam in siamese_models.items():
        for d in range(siam.embed_dim): fd[f"{prefix}_emb_diff_{d}"]=[]
        fd[f"{prefix}_cosine_sim"]=[]; fd[f"{prefix}_euclidean_dist"]=[]
    for prefix in lm_models:
        fd[f"{prefix}_rank_diff"]=[]
        for fn in LM_FEAT_NAMES: fd[f"{prefix}_{fn}_diff"]=[]
    for prefix in rn_models:
        fd[f"{prefix}_score_diff"]=[]; fd[f"{prefix}_p_home_win"]=[]
        for fn in RN_FEAT_NAMES: fd[f"{prefix}_{fn}_diff"]=[]
    for prefix in lw_models:
        fd[f"{prefix}_score_diff"] = []
        fd[f"{prefix}_rank_pos_diff"] = []
        fd[f"{prefix}_percentile_diff"] = []
        for fn in LW_FEAT_NAMES:
            fd[f"{prefix}_{fn}_diff"] = []
    for prefix in rlhf_models:
        fd[f"{prefix}_reward_diff"] = []
        fd[f"{prefix}_p_home_win"] = []
        fd[f"{prefix}_reward_abs"] = []
        fd[f"{prefix}_percentile_diff"] = []
    start_idx = 0
    resumed = False
    # CHECKPOINT WLACZONY - pomija przeliczone pliki, wznawia od checkpointu
    if os.path.exists(state_file):
        try:
            with open(state_file, "rb") as f:
                st = pickle.load(f)
            if (
                st.get("source_file") == os.path.abspath(file_path)
                and st.get("source_mtime_ns") == source_mtime_ns
                and st.get("source_rows") == source_rows
                and st.get("completed") is True
                and os.path.exists(output_file)
            ):
                print("  Ten plik juz przeliczony dla aktualnego CSV - pomijam.")
                return
            if (
                st.get("source_file") == os.path.abspath(file_path)
                and st.get("source_rows", 0) <= source_rows
                and (
                    st.get("source_mtime_ns") == source_mtime_ns
                    or st.get("source_rows", 0) < source_rows
                )
                and st.get("next_idx", 0) > 0
            ):
                start_idx = min(int(st.get("next_idx", 0)), source_rows)
                ts_ratings = st.get("ts_ratings", ts_ratings)
                os_ratings = st.get("os_ratings", os_ratings)
                wl_ratings = st.get("wl_ratings", wl_ratings)
                pr_sc = st.get("pr_sc", pr_sc); pr_gc = st.get("pr_gc", pr_gc); pr_dc = st.get("pr_dc", pr_dc)
                mas_c = st.get("mas_c", mas_c); col_c = st.get("col_c", col_c); keen_c = st.get("keen_c", keen_c)
                all_graph = st.get("all_graph", all_graph)
                melo_models = st.get("melo_models", melo_models)
                bars_models = st.get("bars_models", bars_models)
                ed_models = st.get("ed_models", ed_models)
                siamese_models = st.get("siamese_models", siamese_models)
                lm_models = st.get("lm_models", lm_models)
                rn_models = st.get("rn_models", rn_models)
                lw_models = st.get("lw_models", lw_models)
                rlhf_models = st.get("rlhf_models", rlhf_models)
                fd_prev = st.get("fd")
                if isinstance(fd_prev, dict):
                    fd = fd_prev
                resumed = start_idx > 0
        except Exception as e:
            print(f"  OSTRZEZENIE: nie udalo sie wczytac checkpointu: {e}")

    def save_checkpoint(next_idx, completed=False):
        state_obj = {
            "source_file": os.path.abspath(file_path),
            "source_mtime_ns": source_mtime_ns,
            "source_rows": source_rows,
            "next_idx": int(next_idx),
            "completed": bool(completed),
            "ts_ratings": ts_ratings,
            "os_ratings": os_ratings,
            "wl_ratings": wl_ratings,
            "pr_sc": pr_sc, "pr_gc": pr_gc, "pr_dc": pr_dc,
            "mas_c": mas_c, "col_c": col_c, "keen_c": keen_c,
            "all_graph": all_graph,
            "melo_models": melo_models,
            "bars_models": bars_models,
            "ed_models": ed_models,
            "siamese_models": siamese_models,
            "lm_models": lm_models,
            "rn_models": rn_models,
            "lw_models": lw_models,
            "rlhf_models": rlhf_models,
            "fd": fd,
        }
        _save_state(state_file, state_obj)

    total=len(df)
    n_hist=df['target'].notna().sum()
    mc=1+len(os_models)+len(wl_models)+5+len(keener_models)+len(melo_models)+len(bars_models)+len(ed_models)+len(siamese_models)+len(lm_models)+len(rn_models)+len(lw_models)+len(rlhf_models)
    if resumed:
        print(f"  Wznowienie od meczu {start_idx}/{total} ...")
    print(f"  Wyliczanie ({n_hist} hist + {total-n_hist} przyszlych x {mc} modeli)...")
    t0 = time.time()
    for idx, row in df.iloc[start_idx:].iterrows():
        ht=str(row.get(HOME_COL,f"UH_{idx}")).strip(); at=str(row.get(AWAY_COL,f"UA_{idx}")).strip()
        is_future = pd.isna(row['target'])
        if not is_future:
            target=int(row['target'])
            hg=int(pd.to_numeric(row[score_home_col],errors='coerce'))
            ag=int(pd.to_numeric(row[score_away_col],errors='coerce'))
            outcome=1.0 if target==0 else(0.0 if target==2 else 0.5)
        else:
            target=None; hg=0; ag=0; outcome=0.5
        mn=len(all_graph)
        if not SLIM_MODE:
            th=ts_ratings.get(ht,Rating()); ta_r=ts_ratings.get(at,Rating())
            fd["TS_mu_diff"].append(round(th.mu-ta_r.mu,2))
            fd["TS_sigma_diff"].append(round(th.sigma-ta_r.sigma,2))
            fd["TS_ordinal_diff"].append(round((th.mu-3*th.sigma)-(ta_r.mu-3*ta_r.sigma),2))
            fd["TS_certainty_diff"].append(round((1/th.sigma)-(1/ta_r.sigma),6))
            td=math.sqrt(2*(ts_env.beta**2)+th.sigma**2+ta_r.sigma**2)
            fd["TS_home_win_prob"].append(round(ts_env.cdf((th.mu-ta_r.mu)/td),4))
            if not is_future:
                if target==0: th,ta_r=rate_1vs1(th,ta_r)
                elif target==2: ta_r,th=rate_1vs1(ta_r,th)
                else: th,ta_r=rate_1vs1(th,ta_r,drawn=True)
                ts_ratings[ht]=th; ts_ratings[at]=ta_r
        for p,m in os_models.items():
            h=os_ratings[p].get(ht,m.rating()); a=os_ratings[p].get(at,m.rating())
            fd[f"{p}_mu_diff"].append(round(h.mu-a.mu,2)); fd[f"{p}_sigma_diff"].append(round(h.sigma-a.sigma,2))
            fd[f"{p}_ordinal_diff"].append(round((h.mu-3*h.sigma)-(a.mu-3*a.sigma),2))
            fd[f"{p}_certainty_diff"].append(round((1/h.sigma)-(1/a.sigma),6))
            if not is_future:
                if target==0: [[h],[a]]=m.rate([[h],[a]])
                elif target==2: [[a],[h]]=m.rate([[a],[h]])
                else: [[h],[a]]=m.rate([[h],[a]],scores=[1,1])
                os_ratings[p][ht]=h; os_ratings[p][at]=a
        for p,wm in wl_models.items():
            h=wl_ratings[p].get(ht,wm.rating()); a=wl_ratings[p].get(at,wm.rating())
            fd[f"{p}_mu_diff"].append(round(h.mu-a.mu,2)); fd[f"{p}_sigma_diff"].append(round(h.sigma-a.sigma,2))
            fd[f"{p}_ordinal_diff"].append(round(h.ordinal-a.ordinal,2))
            fd[f"{p}_certainty_diff"].append(round((1/h.sigma)-(1/a.sigma),6))
            fd[f"{p}_p_home_win"].append(round(wm.predict_win(h,a),4))
            fd[f"{p}_p_draw"].append(round(wm.predict_draw(h,a),4))
            if not is_future:
                if target==0: h,a=wm.rate_1v1(h,a)
                elif target==2: a,h=wm.rate_1v1(a,h)
                else: h,a=wm.rate_draw(h,a)
                wl_ratings[p][ht]=h; wl_ratings[p][at]=a
        if not is_future and mn%GRAPH_RECALC_EVERY==0 and mn>0:
            w=all_graph[-GRAPH_WINDOW:]
            pr_sc=pr.compute(w,"standard"); pr_gc=pr.compute(w,"goal_weighted"); pr_dc=pr.compute(w,"decay")
            mas_c=massey.compute(w); col_c=colley.compute(w)
            for kp,km in keener_models.items(): keen_c[kp]=km.compute(w)
        fd["PR_diff"].append(round(pr_sc.get(ht,50)-pr_sc.get(at,50),4))
        fd["PR_GOL_diff"].append(round(pr_gc.get(ht,50)-pr_gc.get(at,50),4))
        fd["PR_DEC_diff"].append(round(pr_dc.get(ht,50)-pr_dc.get(at,50),4))
        fd["MASSEY_diff"].append(round(mas_c.get(ht,0)-mas_c.get(at,0),4))
        fd["COLLEY_diff"].append(round(col_c.get(ht,50)-col_c.get(at,50),4))
        for kp in keener_models: fd[f"{kp}_diff"].append(round(keen_c[kp].get(ht,50)-keen_c[kp].get(at,50),4))
        if not is_future: all_graph.append((ht,at,hg,ag,mn))
        for p,me in melo_models.items():
            rh,ch=me._get_rating(ht); ra,ca=me._get_rating(at)
            fd[f"{p}_rating_diff"].append(round(rh-ra,2))
            fd[f"{p}_p_home_win"].append(round(me.predict_win(ht,at),4))
            fd[f"{p}_intrans"].append(round(me._intransitivity_score(ch,ca),4))
            fd[f"{p}_c_norm_diff"].append(round(float(np.linalg.norm(ch)-np.linalg.norm(ca)),4))
            if not is_future: me.update(ht,at,outcome)
        for p,b in bars_models.items():
            b._init_team(ht); b._init_team(at)
            fd[f"{p}_mu_diff"].append(round(b.ratings[ht]-b.ratings[at],2))
            fd[f"{p}_sigma_diff"].append(round(b.sigmas[ht]-b.sigmas[at],2))
            fd[f"{p}_ordinal_diff"].append(round((b.ratings[ht]-3*b.sigmas[ht])-(b.ratings[at]-3*b.sigmas[at]),2))
            fd[f"{p}_k_factor_diff"].append(round(b.k_factors[ht]-b.k_factors[at],2))
            fd[f"{p}_p_home_win"].append(round(b.predict_win(ht,at),4))
            fd[f"{p}_p_draw"].append(round(b.predict_draw(ht,at),4))
            fd[f"{p}_streak_diff"].append(b.streaks[ht]-b.streaks[at])
            suh=sum(b.surprises[ht][-b.surprise_window:])/max(1,len(b.surprises[ht][-b.surprise_window:]))
            sua=sum(b.surprises[at][-b.surprise_window:])/max(1,len(b.surprises[at][-b.surprise_window:]))
            fd[f"{p}_surprise_diff"].append(round(suh-sua,4))
            fd[f"{p}_skew_diff"].append(round(b.get_posterior_skew(ht)-b.get_posterior_skew(at),4))
            fd[f"{p}_kurtosis_diff"].append(round(b.get_posterior_kurtosis(ht)-b.get_posterior_kurtosis(at),4))
            if not is_future: b.update(ht,at,outcome,match_idx=mn)
        for p,ed in ed_models.items():
            rh=ed._get_rating(ht); ra=ed._get_rating(at); ih=(p=="ED_HOME")
            pw,pdv,pl=ed.predict_probs(ht,at,ih)
            fd[f"{p}_rating_diff"].append(round(rh-ra,2)); fd[f"{p}_p_home_win"].append(round(pw,4))
            fd[f"{p}_p_draw"].append(round(pdv,4)); fd[f"{p}_p_away_win"].append(round(pl,4))
            fd[f"{p}_draw_tendency"].append(round(ed.nu,4))
            fd[f"{p}_expected_score_diff"].append(round((pw+0.5*pdv)-(pl+0.5*pdv),4))
            if not is_future: ed.update(ht,at,outcome,is_home_a=ih)
        for prefix,siam in siamese_models.items():
            if not is_future and siam.should_retrain():
                print(f"   Retrenowanie {prefix} (mecz #{mn})...")
                siam._train()
            ed_arr=siam.get_embedding_diff(ht,at)
            for d in range(siam.embed_dim): fd[f"{prefix}_emb_diff_{d}"].append(round(float(ed_arr[d]),4))
            fd[f"{prefix}_cosine_sim"].append(round(siam.get_cosine_similarity(ht,at),4))
            fd[f"{prefix}_euclidean_dist"].append(round(siam.get_euclidean_distance(ht,at),4))
            if not is_future: siam.add_match(ht,at,target)
        for prefix,lm in lm_models.items():
            if not is_future and lm.should_retrain():
                print(f"   Retrenowanie {prefix} (mecz #{mn})...")
                lm._train()
            fd[f"{prefix}_rank_diff"].append(round(lm.get_ranking_diff(ht,at),4))
            feat_diff=lm.get_team_features_diff(ht,at)
            for i,fn in enumerate(LM_FEAT_NAMES): fd[f"{prefix}_{fn}_diff"].append(round(float(feat_diff[i]),4))
            if not is_future: lm.add_match(ht,at,hg,ag,outcome)
        for prefix,rn in rn_models.items():
            if not is_future and rn.should_retrain():
                print(f"   Retrenowanie {prefix} (mecz #{mn})...")
                rn._train()
            fd[f"{prefix}_score_diff"].append(round(rn.get_score_diff(ht,at),4))
            fd[f"{prefix}_p_home_win"].append(round(rn.get_p_home_win(ht,at),4))
            feat_diff=rn.get_features_diff(ht,at)
            for i,fn in enumerate(RN_FEAT_NAMES): fd[f"{prefix}_{fn}_diff"].append(round(float(feat_diff[i]),4))
            if not is_future: rn.add_match(ht,at,hg,ag,outcome)
        for prefix, lw in lw_models.items():
            if not is_future and lw.should_retrain():
                print(f"   Retrenowanie {prefix} (mecz #{mn})...")
                lw._train()
            fd[f"{prefix}_score_diff"].append(round(lw.get_score_diff(ht, at), 4))
            fd[f"{prefix}_rank_pos_diff"].append(round(lw.get_rank_position_diff(ht, at), 4))
            fd[f"{prefix}_percentile_diff"].append(round(lw.get_percentile_diff(ht, at), 4))
            feat_diff = lw.get_features_diff(ht, at)
            for i, fn in enumerate(LW_FEAT_NAMES):
                fd[f"{prefix}_{fn}_diff"].append(round(float(feat_diff[i]), 4))
            if not is_future: lw.add_match(ht, at, hg, ag, outcome)
        for prefix, rlhf in rlhf_models.items():
            fd[f"{prefix}_reward_diff"].append(round(rlhf.get_reward_diff(ht, at), 4))
            fd[f"{prefix}_p_home_win"].append(round(rlhf.get_p_home_win(ht, at), 4))
            fd[f"{prefix}_reward_abs"].append(round(abs(rlhf.get_reward_diff(ht, at)), 4))
            fd[f"{prefix}_percentile_diff"].append(round(rlhf.get_percentile_diff(ht, at), 4))
            if not is_future: rlhf.add_match(ht, at, hg, ag, outcome)
        done = idx + 1
        if done % CHECKPOINT_EVERY == 0:
            elapsed = max(time.time() - t0, 1e-9)
            speed = done / elapsed
            print(f"  Postep: {done}/{total} ({done/total*100:.1f}%), ~{speed:.1f} mecz/s")
            save_checkpoint(done, completed=False)
    output_data = {DATE_COL: df[DATE_COL], HOME_COL: df[HOME_COL], AWAY_COL: df[AWAY_COL], "result": df["result"]}
    if 'id' in df.columns: output_data['id'] = df['id']
    if 'match_id' in df.columns: output_data['match_id'] = df['match_id']
    output_data.update(fd)
    df_out = pd.DataFrame(output_data)
    added_cols = 0
    def add_copy(new_name, base_cols):
        nonlocal added_cols
        for c in base_cols:
            if c in df_out.columns:
                df_out[new_name] = pd.to_numeric(df_out[c], errors="coerce")
                added_cols += 1
                return
        df_out[new_name] = 0.0
    add_copy("momentum_TS_ordinal", ["TS_ordinal_diff"])
    add_copy("momentum_MASSEY", ["MASSEY_diff"])
    add_copy("momentum_BARS_mu", ["BARS_mu_diff"])
    add_copy("momentum_ED_rating", ["ED_rating_diff", "ED_HOME_rating_diff"])
    add_copy("last_form5_diff", ["RN_form5_diff", "LNET_form5_diff", "LMLE_form5_diff"])
    add_copy("last_form10_diff", ["RN_form10_diff", "LNET_form10_diff", "LMLE_form10_diff"])
    add_copy("last_sos_diff", ["RN_sos_diff", "LNET_sos_diff", "LMLE_sos_diff"])
    protect = {DATE_COL, HOME_COL, AWAY_COL, "result", "id", "match_id"}
    num_cols = [c for c in df_out.columns if c not in protect]
    for c in num_cols:
        df_out[c] = pd.to_numeric(df_out[c], errors="coerce")
    df_out[num_cols] = df_out[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    print(f"  Dodano kolumn 'momentum_'/'last_': {added_cols}")
    print(f"\n  Zapisywanie: {output_file}...")
    df_out.to_csv(output_file, index=False)
    save_checkpoint(len(df), completed=True)
    ts_c=[c for c in df_out.columns if c.startswith("TS_")]
    os_c=[c for c in df_out.columns if any(c.startswith(p+"_") for p in os_models)]
    wl_c=[c for c in df_out.columns if any(c.startswith(p+"_") for p in wl_models)]
    gr_c=[c for c in df_out.columns if any(c.startswith(p) for p in ["PR_","MASSEY_","COLLEY_"])]
    kn_c=[c for c in df_out.columns if c.startswith("KEEN")]
    me_c=[c for c in df_out.columns if c.startswith("mELO")]
    ba_c=[c for c in df_out.columns if c.startswith("BARS")]
    ed_c=[c for c in df_out.columns if c.startswith("ED")]
    si_c=[c for c in df_out.columns if c.startswith("SIAM")]
    lm_c=[c for c in df_out.columns if c.startswith("LM")]
    rn_c=[c for c in df_out.columns if c.startswith("RN")]
    lw_c=[c for c in df_out.columns if c.startswith("LNET") or c.startswith("LMLE")]
    rlhf_c=[c for c in df_out.columns if c.startswith("RLHF")]
    rc=df_out["result"].value_counts()
    print(f"\n  GOTOWE! Plik: {output_file}")
    print(f"   Lacznie kolumn:     {len(df_out.columns)}")
    print(f"   TrueSkill:          {len(ts_c)} cech")
    print(f"   OpenSkill:          {len(os_c)} cech ({len(os_models)} modeli)")
    print(f"   Weng-Lin:           {len(wl_c)} cech ({len(wl_models)} modeli)")
    print(f"   PageRank+Massey+Col:{len(gr_c)} cech")
    print(f"   Keener:             {len(kn_c)} cech")
    print(f"   mELO:               {len(me_c)} cech")
    print(f"   BARS:               {len(ba_c)} cech")
    print(f"   Elo-Davidson:       {len(ed_c)} cech")
    print(f"   Siamese:            {len(si_c)} cech")
    print(f"   LambdaMART:         {len(lm_c)} cech")
    print(f"   RankNet:            {len(rn_c)} cech")
    print(f"   ListNet/ListMLE:    {len(lw_c)} cech ({len(lw_models)} wariantow)")
    print(f"   RLHF:               {len(rlhf_c)} cech ({len(rlhf_models)} warianty)")
    print(f"   Meczow:             {len(df_out)}")
    print(f"   Wyniki:             W={rc.get('W',0)}  D={rc.get('D',0)}  L={rc.get('L',0)}")

    # === AUTO-MERGE: dopisz kolumny Siamese do wszystkie_sezony.csv ===
    siam_cols = [c for c in df_out.columns if c.startswith("SIAM")]
    if siam_cols:
        source_csv = file_path
        try:
            main_df = pd.read_csv(source_csv, low_memory=False, on_bad_lines="warn")
            key_cols = ["Date", "HomeTeam", "AwayTeam"]
            merge_df = df_out[key_cols + siam_cols].copy()
            # zapisz oryginalna kolumne Date z main_df zeby zachowac jej format
            orig_date = main_df["Date"].copy()
            # ujednolic Date po obu stronach do jednego formatu str dd/mm/YYYY do merge
            main_df["_join_date"] = pd.to_datetime(main_df["Date"], dayfirst=True, errors="coerce").dt.strftime("%Y-%m-%d")
            merge_df["_join_date"] = pd.to_datetime(merge_df["Date"], dayfirst=True, errors="coerce").dt.strftime("%Y-%m-%d")
            merge_df = merge_df.drop(columns=["Date"])
            # usun stare kolumny Siamese jesli juz sa
            for c in siam_cols:
                if c in main_df.columns:
                    main_df = main_df.drop(columns=[c])
            merged = main_df.merge(merge_df, on=["_join_date", "HomeTeam", "AwayTeam"], how="left")
            merged = merged.drop(columns=["_join_date"])
            merged["Date"] = orig_date.values
            merged.to_csv(source_csv, index=False, encoding="utf-8-sig")
            print(f"\n  AUTO-MERGE: dopisano {len(siam_cols)} kolumn Siamese do {source_csv}")
            # usun plik posredni bars6
            if os.path.exists(output_file):
                os.remove(output_file)
                print(f"  Usunieto plik posredni: {output_file}")
        except Exception as e:
            print(f"\n  AUTO-MERGE BLAD: {e}")

if __name__ == "__main__":
    # Możesz podać plik jako argument: python bars5.py merged_data.csv
    files_to_process = []
    if len(sys.argv) > 1:
        files_to_process = [sys.argv[1]]
    else:
        if os.path.exists('merged_data.csv'):
            files_to_process = ['merged_data.csv']
        else:
            search_pattern = "*_Teams_Full_Calendar.csv"
            files_to_process = glob.glob(search_pattern)
    if not files_to_process:
        print("  Nie znaleziono plików do przetworzenia. Podaj ścieżkę jako argument lub utwórz 'merged_data.csv'.")
    else:
        print(f"  Znaleziono {len(files_to_process)} plik(ów) do przetworzenia.")
        for file in files_to_process:
            process_file(file)
        print(f"\n  Wszystkie pliki przetworzone!")
