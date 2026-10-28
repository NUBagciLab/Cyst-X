"""Generate pink LaTeX tables; labels, rather than mutable table numbers, identify them."""
from pathlib import Path
import pandas as pd


def render_tables(out, bootstrap):
    out=Path(out)
    ci=pd.read_csv(out/'principal_ci.csv')
    paired=pd.read_csv(out/'paired_comparison.csv')
    auc=pd.read_csv(out/'frozen_pathology_auc_ci.csv')
    frozen=pd.read_csv(out/'frozen_pathology.csv')
    matched=pd.read_csv(out/'matched_auc.csv')
    tasks={'2-class':629,'low_vs_high':474,'histology':357}
    names={'2-class':'Broad','low_vs_high':'IPMN-only','histology':'Histopathology-only'}
    def interval(r, field='estimate'):
        return f'{r[field]:.2f} [{r.lower:.2f}, {r.upper:.2f}]'
    def one(frame, **filters):
        for col,val in filters.items(): frame=frame[frame[col]==val]
        if len(frame)!=1: raise ValueError(f'Expected one row: {filters}')
        return frame.iloc[0]
    tables=[]
    def table(filename,label,caption,header,rows):
        text='\\begin{table}[ht]\n\\centering\n\\caption{\\zwrev{'+caption+'}}\n\\label{'+label+'}\n{\\color{ZhixiangPink}\\small\n\\begin{tabular}{'+'l'*len(header)+'}\n\\toprule\n'
        text+=' & '.join(header)+r'\\'+'\n\\midrule\n'
        text+=''.join(' & '.join(map(str,row))+r'\\'+'\n' for row in rows)
        text+='\\bottomrule\n\\end{tabular}}\n\\end{table}\n'
        (out/filename).write_text(text);tables.append(text)
    rows=[]
    for task,n in tasks.items():
        for name in ['T2W DenseNet','Reader mean']:
            rows.append([n,name,*[interval(one(ci,task=task,method=name,metric=m,scope='internal')) for m in ['sens','spec']]])
    table('reader_ci.tex','tab:zw-reader-ci','Principal internal reader comparisons. Values are percentages with conditional 95\\% bootstrap confidence intervals.', ['Cohort','Method','Sensitivity [95\\% CI]','Specificity [95\\% CI]'],rows)
    rows=[]
    for task,n in tasks.items():
        for metric in ['sens','spec']:
            r=one(paired,task=task,comparator='Reader mean',metric=metric,scope='internal')
            rows.append([n,{'sens':'Sensitivity','spec':'Specificity'}[metric],f'{r.difference_pp:.2f}',f'[{r.lower:.2f}, {r.upper:.2f}]'])
    table('paired_differences.tex','tab:zw-paired-differences',f'Paired differences between internal T2W DenseNet-121 and the arithmetic reader mean, in percentage points. Confidence intervals use the same {bootstrap:,} paired patient bootstrap replicates as Table~\\ref{{tab:zw-reader-ci}}. Intervals crossing zero do not establish equivalence.', ['Cohort','Metric','AI minus reader mean','95\\% CI'],rows)
    rows=[]
    for scope in ['internal','external']:
        for task in tasks:
            for mod in ['t1','t2']:
                f=one(frozen,scope=scope,training_task=task,modality=mod)
                a=one(auc,scope=scope,training_task=task,modality=mod)
                rows.append([scope.title(),names[task],mod.upper()+'W',f'{f.sens:.2f}',f'{f.spec:.2f}',interval(a,'auc')])
    table('frozen_histopathology.tex','tab:zw-frozen-pathology','Evaluation on the same 357 histopathology-confirmed patients (142 high-risk, 215 low-risk). Broad and IPMN-only predictions and thresholds are frozen from the original tasks; histopathology-only rows use separately retrained models. Values are percentages. AUCs use pooled held-out probabilities; their intervals are conditional 95\\% bootstrap confidence intervals.', ['Evaluation','Training task','MRI','Sens','Spec','AUC [95\\% CI]'],rows)
    rows=[]
    for method,model,label in [('centralized','densenet121','Centralized'),('fedavg','densenet121','FedAvg'),('fedprox','densenet121_0.1','FedProx (0.1)'),('fedprox','densenet121_0.3','FedProx (0.3)')]:
        for mod in ['t1','t2']:
            a=one(matched,task='2-class',method=method,model=model,modality=mod,scope='internal')
            b=one(matched,task='2-class',method=method,model=model,modality=mod,scope='external')
            rows.append([label,mod.upper()+'W',*[f'{v:.2f}' for v in [a.pooled_auc,b.pooled_auc,a.center_mean_auc,b.center_mean_auc]]])
    table('matched_auc.tex','tab:zw-matched-auc','Matched AUC summaries for the broad task. Internal and external estimates use the same patients within each modality (652 T1W; 655 T2W). Center means are unweighted averages over the same seven centers. Values are percentages; these are descriptive point estimates.', ['Method','MRI','Pooled int.','Pooled ext.','Center int.','Center ext.'],rows)
    (out/'all_tables.tex').write_text('\n'.join(tables))
    (out/'preview.tex').write_text(r'''\documentclass{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs,xcolor}
\definecolor{ZhixiangPink}{HTML}{FF1493}
\newcommand{\zwrev}[1]{{\color{ZhixiangPink}#1}}
\begin{document}
'''+ '\n\\clearpage\n'.join(tables)+'\n\\end{document}\n')
