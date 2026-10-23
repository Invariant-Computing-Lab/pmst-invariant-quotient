import os, json, math, itertools, hashlib, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy.linalg import block_diag
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

SEED = 20260812
rng = np.random.default_rng(SEED)
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'04_Data'
FIG = ROOT/'05_Figures'
TABLE = ROOT/'06_Tables'
REPORT = ROOT/'07_Reporting'
CODE = ROOT/'03_Reproducibility'
for p in [ROOT, DATA, FIG, TABLE, REPORT, CODE]: p.mkdir(parents=True, exist_ok=True)

# ---------- utilities ----------
def bootstrap_ci(x, stat=np.median, B=1000, alpha=0.05, seed=SEED):
    x=np.asarray(x)
    rr=np.random.default_rng(seed)
    vals=np.empty(B)
    n=len(x)
    for b in range(B): vals[b]=stat(x[rr.integers(0,n,n)])
    return float(np.quantile(vals,alpha/2)), float(np.quantile(vals,1-alpha/2))

def bootstrap_diff_accuracy(y, pa, pb, B=1000, seed=SEED+1):
    rr=np.random.default_rng(seed); y=np.asarray(y); pa=np.asarray(pa); pb=np.asarray(pb); n=len(y)
    vals=np.empty(B)
    for b in range(B):
        idx=rr.integers(0,n,n)
        vals[b]=np.mean(pb[idx]==y[idx])-np.mean(pa[idx]==y[idx])
    return float(np.quantile(vals,.025)), float(np.quantile(vals,.975))

def permutation_accuracy_pvalue(y, pred, n_perm=1000, seed=SEED+2):
    rr=np.random.default_rng(seed); y=np.asarray(y); pred=np.asarray(pred)
    obs=float(np.mean(pred==y)); ge=0
    for _ in range(n_perm):
        yp=rr.permutation(y)
        ge += np.mean(pred==yp) >= obs-1e-15
    return obs, (ge+1)/(n_perm+1)

def cliff_delta(x,y):
    x=np.asarray(x); y=np.asarray(y)
    # efficient via ranks/searchsorted
    ys=np.sort(y)
    gt=np.sum(np.searchsorted(ys,x,side='left'))
    lt=np.sum(len(y)-np.searchsorted(ys,x,side='right'))
    return float((gt-lt)/(len(x)*len(y)))

def rand_herm(n, rr=rng):
    A=rr.normal(size=(n,n))+1j*rr.normal(size=(n,n))
    H=(A+A.conj().T)/2
    return H/(norm(H,'fro')+1e-15)

def rand_unitary(n, rr=rng):
    Z=rr.normal(size=(n,n))+1j*rr.normal(size=(n,n))
    Q,R=np.linalg.qr(Z)
    d=np.diag(R); ph=np.where(np.abs(d)>0,d/np.abs(d),1)
    return Q@np.diag(np.conj(ph))

def rel(a,b): return float(norm(a-b)/(norm(a)+1e-15))

# ============================================================
# H1: Gauge exactness, off-orbit sensitivity, numerical stability
# ============================================================
D=9; N=3
# deterministic trace-word dictionary, lengths 1-3 plus commutator characteristic data
words=[(i,) for i in range(D)]
words += [(i,j) for i in range(D) for j in range(i,D)]
rrw=np.random.default_rng(11)
words += [tuple(rrw.integers(0,D,size=3)) for _ in range(60)]

def gauge_features(Xs):
    f=[]
    for w in words:
        M=np.eye(N,dtype=complex)
        for i in w: M=M@Xs[i]
        t=np.trace(M); f.extend([t.real,t.imag])
    for i in range(D):
        for j in range(i+1,D):
            C=Xs[i]@Xs[j]-Xs[j]@Xs[i]
            k2=np.trace(C@C); det=np.linalg.det(C)
            f.extend([k2.real,k2.imag,det.real,det.imag])
    return np.array(f,float)

rows=[]
for rep in range(250):
    X=[rand_herm(N) for _ in range(D)]
    U=rand_unitary(N)
    Xg=[U@x@U.conj().T for x in X]
    iv=rel(gauge_features(X),gauge_features(Xg))
    raw=math.sqrt(sum(norm(Xg[i]-X[i],'fro')**2 for i in range(D))/sum(norm(X[i],'fro')**2 for i in range(D)))
    rows.append((rep,iv,raw))
h1_exact=pd.DataFrame(rows,columns=['rep','invariant_rel_drift','raw_coordinate_rel_drift'])
h1_exact.to_csv(DATA/'H1_gauge_exact.csv',index=False)

# nuisance vs true perturbation dose response
rows=[]
eps_grid=[0,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,3e-2,1e-1,3e-1]
for eps in eps_grid:
    for rep in range(180):
        X=[rand_herm(N) for _ in range(D)]; U=rand_unitary(N); Xg=[U@x@U.conj().T for x in X]
        Xp=[x+eps*rand_herm(N) for x in Xg]
        rows.append((eps,rep,rel(gauge_features(X),gauge_features(Xp))))
h1_noise=pd.DataFrame(rows,columns=['epsilon','rep','invariant_rel_drift'])
h1_noise.to_csv(DATA/'H1_gauge_offorbit.csv',index=False)
h1_noise_summary=h1_noise.groupby('epsilon').invariant_rel_drift.agg(median='median',q025=lambda s:s.quantile(.025),q975=lambda s:s.quantile(.975)).reset_index()
h1_noise_summary.to_csv(TABLE/'H1_gauge_dose_response.csv',index=False)
mask=h1_noise_summary.epsilon>0
coef=np.polyfit(np.log10(h1_noise_summary.loc[mask,'epsilon']),np.log10(h1_noise_summary.loc[mask,'median']),1)

# GL condition-number stress: algebraic exactness vs floating point stability
rows=[]
for cond in [1,10,1e2,1e3,1e4,1e5,1e6]:
    for rep in range(100):
        X=[rand_herm(N) for _ in range(D)]
        U,V=rand_unitary(N),rand_unitary(N)
        sv=np.geomspace(1,cond,N)
        G=U@np.diag(sv)@V.conj().T; Gi=np.linalg.inv(G)
        Xg=[G@x@Gi for x in X]
        rows.append((cond,rep,rel(gauge_features(X),gauge_features(Xg))))
h1_gl=pd.DataFrame(rows,columns=['condition_number','rep','invariant_rel_drift'])
h1_gl.to_csv(DATA/'H1_GL_conditioning.csv',index=False)

# ============================================================
# H2: S3 composition invariants / completeness / noise
# ============================================================
charges=np.array([-1/3,0,1/3],float)
comps=[]
for nminus in range(4):
    for nzero in range(4-nminus):
        nplus=3-nminus-nzero
        vals=np.array([-1/3]*nminus+[0]*nzero+[1/3]*nplus,float)
        s1=vals.sum(); s2=vals[0]*vals[1]+vals[0]*vals[2]+vals[1]*vals[2]; s3=vals.prod()
        comps.append(dict(n_minus=nminus,n_zero=nzero,n_plus=nplus,s1=s1,s2=s2,s3=s3,vals=vals))
h2_classes=pd.DataFrame([{k:v for k,v in c.items() if k!='vals'} for c in comps])
h2_classes.to_csv(TABLE/'H2_composition_classes.csv',index=False)

# exact permutation test over every composition and all 6 permutations
rows=[]
for cid,c in enumerate(comps):
    vals=c['vals']; base=np.array([c['s1'],c['s2'],c['s3']])
    for pidx,p in enumerate(itertools.permutations(range(3))):
        v=vals[list(p)]; f=np.array([v.sum(),v[0]*v[1]+v[0]*v[2]+v[1]*v[2],v.prod()])
        rows.append((cid,pidx,norm(f-base)))
h2_perm=pd.DataFrame(rows,columns=['class_id','perm_id','feature_abs_drift'])
h2_perm.to_csv(DATA/'H2_permutation_exact.csv',index=False)

# noise robustness: nearest template in standardized elementary symmetric space vs charge-only vs sorted multiset
cent=h2_classes[['s1','s2','s3']].to_numpy(); scales=np.std(cent,axis=0); scales[scales==0]=1
sigmas=[0,.002,.005,.01,.02,.05,.075,.1,.15]
rows=[]
for sigma in sigmas:
    correct_full=correct_charge=correct_sorted=total=0
    for cid,c in enumerate(comps):
        for _ in range(1000):
            obs=c['vals']+rng.normal(0,sigma,3)
            f=np.array([obs.sum(),obs[0]*obs[1]+obs[0]*obs[2]+obs[1]*obs[2],obs.prod()])
            pred_full=int(np.argmin(np.sum(((cent-f)/scales)**2,axis=1)))
            # charge-only chooses nearest s1; ties broken first => intentionally demonstrates non-completeness
            pred_charge=int(np.argmin((h2_classes.s1.to_numpy()-f[0])**2))
            sorted_obs=np.sort(obs)
            pred_sorted=int(np.argmin([np.sum((sorted_obs-np.sort(cc['vals']))**2) for cc in comps]))
            correct_full += pred_full==cid; correct_charge += pred_charge==cid; correct_sorted += pred_sorted==cid; total+=1
    rows.append((sigma,correct_full/total,correct_charge/total,correct_sorted/total))
h2_noise=pd.DataFrame(rows,columns=['sigma','poly_invariant_accuracy','charge_only_accuracy','sorted_multiset_accuracy'])
h2_noise.to_csv(TABLE/'H2_noise_robustness.csv',index=False)

# ============================================================
# H3: full nuisance quotient, invariant modulus J, OOD, reconstruction
# ============================================================
def transform4(a,b,c,d,s,tau,lam):
    return (lam*a*s**2/tau, lam*b*tau*s**4, lam*c*tau*s**2, lam*d*tau*s**6)
def J4(a,b,c,d): return c*d/(b*b)
def canonicalize4(a,b,c,d):
    s=np.sqrt(c/b); tau=np.sqrt(a/c); lam=tau/(a*s*s)
    return np.array(transform4(a,b,c,d,s,tau,lam))

rows=[]
for rep in range(4000):
    a,b,c,d=10**rng.uniform(-4,4,4); s,tau,lam=10**rng.uniform(-4,4,3)
    vals=transform4(a,b,c,d,s,tau,lam)
    j0,j1=J4(a,b,c,d),J4(*vals)
    can=canonicalize4(a,b,c,d)
    rows.append((rep,j0,j1,abs(j1-j0)/(abs(j0)+1e-15),norm(np.log(vals)-np.log([a,b,c,d])),norm(can[:3]-1),abs(can[3]-j0)/(abs(j0)+1e-15)))
h3_exact=pd.DataFrame(rows,columns=['rep','J_before','J_after','J_rel_drift','raw_log_coeff_drift','gauge_fix_error','reconstruction_rel_error'])
h3_exact.to_csv(DATA/'H3_full_quotient_exact.csv',index=False)

# OOD classification under severe nuisance shift
family_J=np.array([0.04,0.12,0.4,1.3,4.2,13.0])
def sample_family(J,n,logrange,noise):
    base=np.array([1.,1.,1.,J]); raw=[]; inv=[]
    for _ in range(n):
        s,tau,lam=10**rng.uniform(logrange[0],logrange[1],3)
        vals=np.array(transform4(*base,s,tau,lam))*np.exp(rng.normal(0,noise,4))
        raw.append(np.log(vals)); inv.append([np.log(J4(*vals))])
    return np.array(raw),np.array(inv)
Xtr_raw=[]; Xtr_inv=[]; ytr=[]; Xte_raw=[]; Xte_inv=[]; yte=[]
for k,J in enumerate(family_J):
    xr,xi=sample_family(J,500,(-.35,.35),.006); Xtr_raw.append(xr); Xtr_inv.append(xi); ytr += [k]*len(xr)
    xr,xi=sample_family(J,500,(-3,3),.006); Xte_raw.append(xr); Xte_inv.append(xi); yte += [k]*len(xr)
Xtr_raw=np.vstack(Xtr_raw); Xtr_inv=np.vstack(Xtr_inv); ytr=np.array(ytr)
Xte_raw=np.vstack(Xte_raw); Xte_inv=np.vstack(Xte_inv); yte=np.array(yte)
clf_raw=LogisticRegression(max_iter=5000).fit(Xtr_raw,ytr); clf_inv=LogisticRegression(max_iter=5000).fit(Xtr_inv,ytr)
pr=clf_raw.predict(Xte_raw); pi=clf_inv.predict(Xte_inv)
acc_raw=accuracy_score(yte,pr); acc_inv=accuracy_score(yte,pi); ci_diff=bootstrap_diff_accuracy(yte,pr,pi)
obs_acc,p_perm=permutation_accuracy_pvalue(yte,pi,1000)
h3_ood=pd.DataFrame([['raw_coefficients',acc_raw],['quotient_modulus_J',acc_inv],['J_ablated_degree_weight_N_only',1/len(family_J)]],columns=['representation','OOD_accuracy'])
h3_ood.to_csv(TABLE/'H3_OOD_ablation.csv',index=False)

# adversarial nuisance grid at one canonical law: J should remain unchanged despite extreme transformations
rows=[]
base=np.array([1.7,.8,2.2,3.1]); jbase=J4(*base)
for ls in [-6,-4,-2,0,2,4,6]:
  for lt in [-6,-4,-2,0,2,4,6]:
    for ll in [-6,-4,-2,0,2,4,6]:
        vals=transform4(*base,10**ls,10**lt,10**ll)
        rows.append((ls,lt,ll,J4(*vals),abs(J4(*vals)-jbase)/(abs(jbase)+1e-15)))
h3_adv=pd.DataFrame(rows,columns=['log10_s','log10_tau','log10_lambda','J','J_rel_drift'])
h3_adv.to_csv(DATA/'H3_adversarial_nuisance_grid.csv',index=False)

# ============================================================
# H4: spectral identity under preserving vs breaking perturbations
# ============================================================
qsec=np.array([-1.,0.,-1/3,2/3]); k=3; ns=len(qsec); dim=k*ns
Q=np.diag(np.repeat(qsec,k)); slices=[slice(i*k,(i+1)*k) for i in range(ns)]

def H0_make():
    bases=[np.array([.5,2.0,4.5]),np.array([.75,2.3,4.9]),np.array([1.1,2.7,5.3]),np.array([1.4,3.1,5.8])]
    return block_diag(*[np.diag(b.astype(complex))+.05*rand_herm(k) for b in bases])
def V_pres(): return block_diag(*[rand_herm(k) for _ in range(ns)])/math.sqrt(ns)
def V_break():
    A=rng.normal(size=(dim,dim))+1j*rng.normal(size=(dim,dim)); V=(A+A.conj().T)/2
    for sl in slices: V[sl,sl]=0
    return V/(norm(V,'fro')+1e-15)
def spectral_metrics(H,H0):
    ev,vec=np.linalg.eigh(H); ev0=np.linalg.eigvalsh(H0)
    ed=norm(ev-ev0)/(norm(ev0)+1e-15)
    cm=norm(H@Q-Q@H,'fro')/(norm(H,'fro')*norm(Q,'fro')+1e-15)
    imp=[]
    for j in range(dim):
        psi=vec[:,j]; p=np.array([np.sum(np.abs(psi[sl])**2) for sl in slices]); imp.append(max(0.,1-np.sum(p*p)))
    return ed,cm,float(np.mean(imp))
rows=[]
for eps in [0,1e-4,1e-3,1e-2,.03,.1,.2,.5]:
    for rep in range(250):
        H0=H0_make(); scale=norm(H0,'fro')
        for typ,gen in [('preserving',V_pres),('breaking',V_break)]:
            H=H0+eps*scale*gen(); ed,cm,imp=spectral_metrics(H,H0)
            rows.append((eps,rep,typ,ed,cm,imp))
h4=pd.DataFrame(rows,columns=['epsilon','rep','perturbation','energy_drift','commutator_norm','sector_impurity'])
h4.to_csv(DATA/'H4_spectral_identity.csv',index=False)
h4_sum=h4.groupby(['epsilon','perturbation']).agg(energy_median=('energy_drift','median'),commutator_median=('commutator_norm','median'),impurity_median=('sector_impurity','median'),impurity_q025=('sector_impurity',lambda s:s.quantile(.025)),impurity_q975=('sector_impurity',lambda s:s.quantile(.975))).reset_index()
h4_sum.to_csv(TABLE/'H4_spectral_summary.csv',index=False)
# confirmatory eps=.1 separation
h4e=h4[h4.epsilon==.1]; yp=(h4e.perturbation=='breaking').astype(int).to_numpy(); score=h4e.sector_impurity.to_numpy(); auc=roc_auc_score(yp,score)
pres=h4e[h4e.perturbation=='preserving'].sector_impurity.to_numpy(); brk=h4e[h4e.perturbation=='breaking'].sector_impurity.to_numpy()
u,p_mwu=mannwhitneyu(brk,pres,alternative='greater'); delta=cliff_delta(brk,pres)

# spectral discriminant critical-locus robustness
mgrid=np.linspace(-2,2,401); rows=[]
for sigma in [0,1e-5,1e-4,1e-3,1e-2,.05,.1]:
    est=[]
    for rep in range(250):
        ds=[]
        for m in mgrid:
            ev=np.array([-1.,m,1.])+rng.normal(0,sigma,3)
            disc=np.prod([(ev[i]-ev[j])**2 for i in range(3) for j in range(i+1,3)])
            ds.append(disc)
        ds=np.array(ds); left=np.where(mgrid<0)[0]; right=np.where(mgrid>0)[0]
        ml=mgrid[left[np.argmin(ds[left])]]; mr=mgrid[right[np.argmin(ds[right])]]
        est.append((ml,mr))
    est=np.array(est)
    rows.append((sigma,np.median(np.abs(est[:,0]+1)),np.quantile(np.abs(est[:,0]+1),.95),np.median(np.abs(est[:,1]-1)),np.quantile(np.abs(est[:,1]-1),.95)))
h4_disc=pd.DataFrame(rows,columns=['noise_sigma','left_median_abs_error','left_q95_abs_error','right_median_abs_error','right_q95_abs_error'])
h4_disc.to_csv(TABLE/'H4_discriminant_localization.csv',index=False)

# ============================================================
# H5: literal Psi geometry falsification
# ============================================================
from scipy.spatial import ConvexHull

def next_poly(v):
    m,n=v.shape
    pts=np.vstack([np.hstack([v,np.full((m,1),t)]) for t in (-1.,0.,1.)])
    dim=n+1
    if dim==1:
        hv=np.array([[pts[:,0].min()],[pts[:,0].max()]])
        return pts,hv,None
    hull=ConvexHull(pts)
    hv=np.unique(np.round(pts[hull.vertices],12),axis=0)
    return pts,hv,hull
v=np.zeros((1,0)); rows=[]; central_rows=[]
for nd in range(1,7):
    old=v.copy(); pts,v,h=next_poly(old)
    vol=2. if nd==1 else h.volume
    rows.append((nd,len(pts),len(v),2**nd,vol,2**nd))
    if nd>=2:
        cent=np.hstack([old,np.zeros((len(old),1))])
        for idx,p in enumerate(cent):
            z=h.equations[:,:-1]@p+h.equations[:,-1]
            central_rows.append((nd,idx,bool(np.any(np.abs(z)<1e-9)),bool(np.all(z<-1e-9))))
h5=pd.DataFrame(rows,columns=['dimension','aggregated_points','extreme_vertices','hypercube_expected_vertices','volume','expected_hypercube_vertices'])
h5.to_csv(TABLE/'H5_literal_Psi_geometry.csv',index=False)
h5c=pd.DataFrame(central_rows,columns=['dimension','central_vertex_id','is_boundary','is_strict_interior'])
h5c.to_csv(DATA/'H5_central_copy_status.csv',index=False)

# ============================================================
# Summary + confirmatory gates
# ============================================================
ci_gauge=bootstrap_ci(h1_exact.invariant_rel_drift.to_numpy())
ci_raw=bootstrap_ci(h1_exact.raw_coordinate_rel_drift.to_numpy())
ci_J=bootstrap_ci(h3_exact.J_rel_drift.to_numpy())
ci_rec=bootstrap_ci(h3_exact.reconstruction_rel_error.to_numpy())

h1_pass = h1_exact.invariant_rel_drift.median() <= 1e-12 and h1_exact.raw_coordinate_rel_drift.median() > .5
h2_pass = h2_perm.feature_abs_drift.max() <= 1e-15 and h2_classes[['s1','s2','s3']].drop_duplicates().shape[0]==10 and h2_classes.s1.nunique()<10
h3_pass = h3_exact.J_rel_drift.median() <= 1e-12 and acc_inv>=.95 and (acc_inv-acc_raw)>=.30 and h3_exact.reconstruction_rel_error.median()<=1e-12
h4_pass = h4e[h4e.perturbation=='preserving'].sector_impurity.median() <= 1e-10 and h4e[h4e.perturbation=='preserving'].energy_drift.median()>1e-3 and h4e[h4e.perturbation=='breaking'].sector_impurity.median()>.1 and auc>.95
h5_pass = (h5.extreme_vertices.to_numpy()==h5.hypercube_expected_vertices.to_numpy()).all() and not h5c.is_strict_interior.any()

summary={
 'seed':SEED,
 'confirmatory_gates':{
  'H1_gauge_exactness':bool(h1_pass),'H2_composition_completeness':bool(h2_pass),'H3_quotient_reconstructibility':bool(h3_pass),'H4_spectral_identity_separation':bool(h4_pass),'H5_literal_Psi_falsification':bool(h5_pass)},
 'H1':{
   'median_invariant_drift':float(h1_exact.invariant_rel_drift.median()),'bootstrap95_median_ci':ci_gauge,
   'median_raw_coordinate_drift':float(h1_exact.raw_coordinate_rel_drift.median()),'raw_bootstrap95_median_ci':ci_raw,
   'offorbit_loglog_slope':float(coef[0]),
   'GL_cond1e6_median_drift':float(h1_gl[h1_gl.condition_number==1e6].invariant_rel_drift.median())},
 'H2':{
   'max_exact_permutation_drift':float(h2_perm.feature_abs_drift.max()),
   'unique_total_charge_classes':int(h2_classes.s1.nunique()),'unique_full_invariant_classes':int(h2_classes[['s1','s2','s3']].drop_duplicates().shape[0]),
   'accuracy_sigma_0_05':h2_noise[np.isclose(h2_noise.sigma,.05)].iloc[0].to_dict()},
 'H3':{
   'median_J_drift':float(h3_exact.J_rel_drift.median()),'bootstrap95_median_J_ci':ci_J,
   'median_reconstruction_error':float(h3_exact.reconstruction_rel_error.median()),'bootstrap95_reconstruction_ci':ci_rec,
   'raw_OOD_accuracy':float(acc_raw),'invariant_OOD_accuracy':float(acc_inv),'accuracy_difference_bootstrap95_ci':ci_diff,
   'invariant_permutation_test_p':float(p_perm),'adversarial_grid_max_J_drift':float(h3_adv.J_rel_drift.max())},
 'H4':{
   'eps_0_1_preserving_energy_median':float(h4e[h4e.perturbation=='preserving'].energy_drift.median()),
   'eps_0_1_preserving_impurity_median':float(np.median(pres)),'eps_0_1_breaking_impurity_median':float(np.median(brk)),
   'sector_impurity_AUC':float(auc),'MannWhitney_p':float(p_mwu),'Cliffs_delta':float(delta)},
 'H5':{
   'vertex_sequence':h5.extreme_vertices.astype(int).tolist(),'all_match_2powN':bool((h5.extreme_vertices==h5.hypercube_expected_vertices).all()),
   'strict_interior_central_vertices':int(h5c.is_strict_interior.sum())}
}
with open(REPORT/'FINAL_RESULTS.json','w') as f: json.dump(summary,f,indent=2)

# compact confirmatory table
confirm=pd.DataFrame([
 ['H1','Gauge trace/commutator invariants remain exact under U(3), while raw coordinates change','PASS' if h1_pass else 'FAIL'],
 ['H2','Elementary symmetric composition invariants are permutation-exact and complete on the 10-class alphabet','PASS' if h2_pass else 'FAIL'],
 ['H3','Full nuisance quotient modulus J is exact, OOD-discriminative, and reconstructs a canonical law representative','PASS' if h3_pass else 'FAIL'],
 ['H4','Charge-sector identity survives preserving perturbations but fails under sector-breaking perturbations','PASS' if h4_pass else 'FAIL'],
 ['H5','Literal Appendix-B Psi operator produces hypercubes, not the claimed ternary/octagonal family','PASS' if h5_pass else 'FAIL']
],columns=['Hypothesis','Pre-specified criterion summary','Outcome'])
confirm.to_csv(TABLE/'Confirmatory_Hypotheses.csv',index=False)

# ---------- figures ----------
plt.figure(figsize=(6.5,4.5))
x=h1_noise_summary.epsilon.to_numpy(); m=x>0
plt.loglog(x[m],h1_noise_summary.loc[m,'median'],marker='o')
plt.fill_between(x[m],h1_noise_summary.loc[m,'q025'],h1_noise_summary.loc[m,'q975'],alpha=.2)
plt.xlabel('Off-orbit perturbation epsilon'); plt.ylabel('Relative invariant drift'); plt.title('Gauge-invariant fingerprint: exact on orbit, responsive off orbit')
plt.tight_layout(); plt.savefig(FIG/'Figure_1_Gauge_Dose_Response.png',dpi=300); plt.close()

plt.figure(figsize=(6.5,4.5))
plt.plot(h2_noise.sigma,h2_noise.poly_invariant_accuracy,marker='o',label='Polynomial invariant')
plt.plot(h2_noise.sigma,h2_noise.sorted_multiset_accuracy,marker='o',label='Sorted multiset baseline')
plt.plot(h2_noise.sigma,h2_noise.charge_only_accuracy,marker='o',label='Charge-only ablation')
plt.xlabel('Constituent noise sigma (e units)'); plt.ylabel('10-class identity accuracy'); plt.ylim(0,1.03); plt.legend(); plt.title('Permutation-invariant composition identity under noise')
plt.tight_layout(); plt.savefig(FIG/'Figure_2_Composition_Robustness.png',dpi=300); plt.close()

plt.figure(figsize=(6.5,4.5))
plt.bar(h3_ood.representation,h3_ood.OOD_accuracy)
plt.ylabel('OOD law-orbit identity accuracy'); plt.ylim(0,1.03); plt.xticks(rotation=18,ha='right'); plt.title('Quotient modulus outperforms raw coefficients under nuisance shift')
plt.tight_layout(); plt.savefig(FIG/'Figure_3_Quotient_OOD.png',dpi=300); plt.close()

plt.figure(figsize=(6.5,4.5))
for typ,g in h4_sum.groupby('perturbation'):
    plt.plot(g.epsilon,g.impurity_median,marker='o',label=typ)
plt.xscale('symlog',linthresh=1e-4); plt.xlabel('Relative Hamiltonian perturbation epsilon'); plt.ylabel('Median charge-sector impurity'); plt.legend(); plt.title('Identity-preserving versus identity-breaking spectral perturbations')
plt.tight_layout(); plt.savefig(FIG/'Figure_4_Spectral_Identity.png',dpi=300); plt.close()

plt.figure(figsize=(6.5,4.5))
plt.plot(h5.dimension,h5.extreme_vertices,marker='o',label='Literal Psi')
plt.plot(h5.dimension,h5.hypercube_expected_vertices,marker='x',label='2^n')
plt.xlabel('Dimension n'); plt.ylabel('Extreme vertex count'); plt.legend(); plt.title('Literal Appendix-B operator follows the hypercube vertex law')
plt.tight_layout(); plt.savefig(FIG/'Figure_5_Psi_Falsification.png',dpi=300); plt.close()

# Method metadata
meta={
 'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,
 'seed':SEED,'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
 'n_h1_exact':len(h1_exact),'n_h1_offorbit':len(h1_noise),'n_h3_exact':len(h3_exact),'n_h4':len(h4)
}
with open(REPORT/'ENVIRONMENT.json','w') as f: json.dump(meta,f,indent=2)
print(json.dumps(summary,indent=2))
