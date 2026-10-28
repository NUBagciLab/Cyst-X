"""Reproduce Zhixiang's four supplementary tables from saved predictions."""
from pathlib import Path
import argparse, hashlib, json, platform
import numpy as np
import pandas as pd
import cohort_io as cmp
from render_tables import render_tables

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--classification-root',type=Path,required=True)
parser.add_argument('--labels',type=Path,required=True)
parser.add_argument('--readers',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,required=True)
parser.add_argument('--bootstrap',type=int,default=1000)
parser.add_argument('--seed',type=int,default=20261024)
args=parser.parse_args()
if args.bootstrap < 1: parser.error('--bootstrap must be positive')
C=args.classification_root.resolve(); O=args.output_dir.resolve()
if O.exists() and any(O.iterdir()): parser.error('Output directory must be empty; use a new directory')
O.mkdir(parents=True,exist_ok=True)
# Hash only consumed inputs. Keep filenames and hashes, not patient-level values.
consumed={}
_original_read_excel=pd.read_excel
_original_read_csv=pd.read_csv
def track(path):
 p=Path(path).resolve()
 name=str(p.relative_to(C)) if p.is_relative_to(C) else p.name
 consumed[name]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read_excel(path,*a,**kw):
 track(path);return _original_read_excel(path,*a,**kw)
def read_csv(path,*a,**kw):
 track(path);return _original_read_csv(path,*a,**kw)
pd.read_excel=read_excel;pd.read_csv=read_csv
alias,groups=cmp.cohorts(args.labels,args.readers)
for task,n in {'2-class':629,'low_vs_high':474,'histology':357}.items():
 if len(groups[task])!=n: raise ValueError(f'Unexpected {task} cohort size: {len(groups[task])}, expected {n}')
 if (groups[task].Ground_truth_Risk_Assessment==2).sum()!=142: raise ValueError('Unexpected high-risk count')
B=args.bootstrap
cis=[];pairs=[];aucs=[]
def roc_auc_score(y,p):
 y=np.asarray(y);p=np.asarray(p);n1=int((y==1).sum());n0=int((y==0).sum())
 if not n0 or not n1: raise ValueError('AUC requires both classes')
 return (pd.Series(p).rank(method='average').to_numpy()[y==1].sum()-n1*(n1+1)/2)/(n1*n0)

for task,g in groups.items():
 y=(g.Ground_truth_Risk_Assessment//2).to_numpy(); rp=np.array([g[f'reader{i}'].to_numpy()//2 for i in [1,2,3]]).T
 for name,inp,method,model in [x for x in cmp.METHODS if x[0]=="DenseNet-121" and x[1]=="T2W Only"]:
  for scope in ['internal']:
   p=C/scope/task/method/'saved'/model
   a=cmp.predictions(p/'result_calibrated.xlsx',alias,g);b=cmp.predictions(p/'result.xlsx',alias,g)
   assert np.array_equal(a.Fold,b.Fold) and np.allclose(a.Probability,b.Probability,atol=1e-6,rtol=0)
   if name=='DenseNet-121' and inp=='T2W Only':
    pred=a.Prediction.to_numpy(); methods=np.column_stack([pred,rp]); names=['T2W DenseNet','Reader 1','Reader 2','Reader 3','Reader mean']
    for positive,metric in [(True,'sens'),(False,'spec')]:
     rng=np.random.default_rng(args.seed+int(positive))
     correct=methods[y==int(positive)] if positive else 1-methods[y==0]
     ix=rng.integers(0,len(correct),(B,len(correct))); boots=correct[ix].mean(axis=1);boots=np.column_stack([boots,boots[:,1:4].mean(axis=1)])
     means=np.r_[correct.mean(axis=0),correct[:,1:4].mean()]
     for k,n in enumerate(names):
      lo,hi=np.quantile(boots[:,k],[.025,.975]);cis.append(dict(task=task,scope=scope,method=n,metric=metric,estimate=means[k]*100,lower=lo*100,upper=hi*100,n=len(correct)))
     for k in range(1,5):
      diff=boots[:,0]-boots[:,k];lo,hi=np.quantile(diff,[.025,.975])
      pairs.append(dict(task=task,scope=scope,comparator=names[k],metric=metric,difference_pp=(means[0]-means[k])*100,lower=lo*100,upper=hi*100))


def auc_ci(y,prob,seed):
 y=np.asarray(y,dtype=int);prob=np.asarray(prob);_,inv=np.unique(prob,return_inverse=True);k=inv.max()+1
 pos=np.bincount(inv[y==1],minlength=k);neg=np.bincount(inv[y==0],minlength=k);n1=pos.sum();n0=neg.sum()
 if not n1 or not n0:return (float('nan'),)*3
 rng=np.random.default_rng(seed);pc=rng.multinomial(n1,pos/n1,size=B);nc=rng.multinomial(n0,neg/n0,size=B)
 boot=(pc*(np.cumsum(nc,axis=1)-.5*nc)).sum(axis=1)/(n1*n0)*100
 return roc_auc_score(y,prob)*100,*np.quantile(boot,[.025,.975])

fr=[]
for scope in ['internal','external']:
 for task in ['2-class','low_vs_high','histology']:
  for mod in ['t1','t2']:
   pp=C/scope/task/'centralized/saved/densenet121'/mod/'result_calibrated.xlsx'
   d=cmp.predictions(pp,alias,groups['histology']);auc,lo,hi=auc_ci(d.Label,d.Probability,args.seed)
   fr.append(dict(scope=scope,training_task=task,modality=mod,n=len(d),auc=auc,lower=lo,upper=hi))


for task in ['2-class']:
 configs=[('centralized','densenet121')]+([('fedavg','densenet121'),('fedprox','densenet121_0.1'),('fedprox','densenet121_0.3')] if task=='2-class' else [])
 for method,model in configs:
  for mod in ['t1','t2']:
   frames={}
   for scope in ['internal','external']:
    p=C/scope/task/method/'saved'/model/mod/'result.xlsx'
    if not p.exists(): raise FileNotFoundError(p)
    d=pd.read_excel(p);d['id']=d.ID.map(cmp.case_id);d['center']=d.id.map(lambda x: next((i for i,keys in enumerate([['nyu'],['CAD','MCF'],['northwestern','NU'],['AHN','ahn'],['mca'],['IU'],['EMC']]) if any(k in x for k in keys)),None));frames[scope]=d
   if len(frames)!=2: raise ValueError('Both evaluation scopes are required')
   for d in frames.values():
    if d.id.duplicated().any() or d.center.isna().any(): raise ValueError('Duplicate patient IDs or unknown centers')
   if set(frames['internal'].id)!=set(frames['external'].id): raise ValueError('Patient sets differ')
   left=frames['internal'].set_index('id').Label.sort_index()
   right=frames['external'].set_index('id').Label.sort_index()
   if not left.equals(right): raise ValueError('Labels differ between evaluation scopes')
   for scope,d in frames.items():
    if len(d)!={'t1':652,'t2':655}[mod]: raise ValueError('Unexpected matched cohort size')
    if d.center.nunique()!=7 or any(x.Label.nunique()!=2 for _,x in d.groupby('center')): raise ValueError('Expected seven centers with both classes')
    foldcol='Fold' if 'Fold' in d else 'Held-out Center'
    groupauc=lambda col:[roc_auc_score(x.Label,x.Probability)*100 for _,x in d.groupby(col) if x.Label.nunique()==2]
    row=dict(task=task,method=method,model=model,modality=mod,scope=scope,n=len(d),pooled_auc=roc_auc_score(d.Label,d.Probability)*100,fold_mean_auc=np.mean(groupauc(foldcol)),center_mean_auc=np.mean(groupauc('center')),center_count=len(groupauc('center')))
    aucs.append(row)

frozen=[]
for scope in ['internal','external']:
 for task in ['2-class','low_vs_high','histology']:
  for mod in ['t1','t2']:
   d=cmp.predictions(C/scope/task/'centralized/saved/densenet121'/mod/'result_calibrated.xlsx',alias,groups['histology'])
   frozen.append(dict(scope=scope,training_task=task,modality=mod,**cmp.metrics(d.Label,d.Prediction),auc=roc_auc_score(d.Label,d.Probability)*100))
for name,rows in [('principal_ci',cis),('paired_comparison',pairs),('frozen_pathology',frozen),('frozen_pathology_auc_ci',fr),('matched_auc',aucs)]:
 pd.DataFrame(rows).to_csv(O/(name+'.csv'),index=False)
render_tables(O,B)
(O/'run_manifest.json').write_text(json.dumps(dict(bootstrap=B,seed=args.seed,python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,cohorts={k:len(v) for k,v in groups.items()},inputs=consumed),indent=2)+'\n')
print(f'Wrote four tables and five result CSVs to {O}')
