"""Compare saved predictions and readers on identical paired-patient cohorts."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

HERE = Path(__file__).resolve().parent
TASKS = ['2-class', 'low_vs_high', 'histology']
METHODS = [
    ('3D Radiomics', 'T1W Only', 'radiomics', 'random_forest/t1'),
    ('3D Radiomics', 'T2W Only', 'radiomics', 'random_forest/t2'),
    ('DenseNet-121', 'T1W Only', 'centralized', 'densenet121/t1'),
    ('DenseNet-121', 'T2W Only', 'centralized', 'densenet121/t2'),
    ('Early Fusion', 'T1W+T2W', 'early_fusion', 'densenet121'),
    ('Late Fusion', 'T1W+T2W', 'late_fusion', 'densenet121'),
    ('Siamese Fusion', 'T1W+T2W', 'siamese_fusion', 'densenet121'),
    ('Logit Fusion', 'T1W+T2W', 'logit_fusion', 'densenet121'),
]


def case_id(value):
    return str(value).strip().removesuffix('.gz').removesuffix('.nii')


def cohorts():
    reference = pd.read_excel(HERE / 'IPMN_labels_total.xlsx')
    aliases, metadata = {}, {}
    for _, row in reference.iterrows():
        if row[['Patient ID T1', 'Patient ID T2', 'risk assessment']].isna().any():
            continue
        canonical = case_id(row['Patient ID T1'])
        if canonical in metadata:
            raise ValueError(f'Duplicate reference patient: {canonical}')
        risk = int(row['risk assessment'])
        if risk not in (0, 1, 2):
            raise ValueError(f'Invalid reference risk: {canonical}')
        metadata[canonical] = (risk, 'no change' in str(row['follow-up information']).lower())
        for column in ['Patient ID T1', 'Patient ID T2']:
            alias = case_id(row[column])
            if alias in aliases and aliases[alias] != canonical:
                raise ValueError(f'Ambiguous patient alias: {alias}')
            aliases[alias] = canonical
    readers = pd.read_csv(HERE / 'Cyst-X_bigdata_risk_assessment.csv')
    readers['canonical'] = readers['Patient ID'].map(case_id).map(aliases)
    if readers['canonical'].isna().any() or readers['canonical'].duplicated().any():
        raise ValueError('Unmapped or duplicate reader cases')
    readers = readers.set_index('canonical').sort_index()
    for patient, row in readers.iterrows():
        if metadata[patient][0] != row['Ground_truth_Risk_Assessment']:
            raise ValueError(f'Reader/reference ground truth mismatch: {patient}')
        if not row[['Ground_truth_Risk_Assessment', 'reader1', 'reader2', 'reader3']].isin([0, 1, 2]).all():
            raise ValueError(f'Invalid reader assessment: {patient}')
    groups = {}
    for task in TASKS:
        keep = [patient for patient in readers.index if task == '2-class' or
                metadata[patient][0] > 0 and (task != 'histology' or not metadata[patient][1])]
        groups[task] = readers.loc[keep]
    return aliases, groups


def metrics(labels, predictions):
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    if tp + fn == 0 or tn + fp == 0:
        raise ValueError('Both risk classes are required')
    return dict(n=int(tn + fp + fn + tp), positive_n=int(tp + fn),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                sens=float(tp / (tp + fn) * 100), spec=float(tn / (tn + fp) * 100))


def predictions(path, aliases, cohort):
    frame = pd.read_excel(path)
    if 'Held-out Center' in frame and 'Fold' not in frame:
        frame = frame.rename(columns={'Held-out Center': 'Fold'})
    required = ['ID', 'Label', 'Prediction', 'Probability', 'Fold']
    if not set(required).issubset(frame):
        raise ValueError(f'Missing prediction columns: {path}')
    frame['canonical'] = frame['ID'].map(case_id).map(aliases)
    frame = frame[frame['canonical'].isin(cohort.index)].copy()
    if frame['canonical'].duplicated().any():
        raise ValueError(f'Duplicate cohort predictions: {path}')
    missing = set(cohort.index) - set(frame['canonical'])
    if missing:
        raise ValueError(f'{len(missing)} cohort cases missing in {path}: {sorted(missing)[:5]}')
    frame = frame.set_index('canonical').loc[cohort.index]
    if frame[required].isna().any().any():
        raise ValueError(f'Missing prediction values: {path}')
    if not frame['Label'].isin([0, 1]).all() or not frame['Prediction'].isin([0, 1]).all():
        raise ValueError(f'Invalid binary values: {path}')
    truth = (cohort['Ground_truth_Risk_Assessment'] // 2).to_numpy()
    if not np.array_equal(frame['Label'].to_numpy(), truth):
        raise ValueError(f'Model/reader labels disagree: {path}')
    if not np.isfinite(frame['Probability']).all() or not frame['Probability'].between(0, 1).all():
        raise ValueError(f'Invalid probabilities: {path}')
    if path.name == 'result.xlsx' and not np.array_equal(
            frame['Prediction'].to_numpy(), (frame['Probability'] >= 0.5).astype(int).to_numpy()):
        raise ValueError(f'Uncalibrated predictions do not use threshold 0.50: {path}')
    return frame


def render(template, results):
    task_index, updated = -1, 0
    output = []
    names = ['Radiologist 1', 'Radiologist 2', 'Radiologist 3', 'Average']
    names += list(dict.fromkeys(method[0] for method in METHODS))
    for line in template.splitlines():
        if re.search(r'\\textbf\{(?:629 broad|474 IPMN-only|357 pathology-confirmed)', line):
            task_index += 1
        name = next((name for name in names if name in line), None)
        if line.startswith('\\rowcolor') and name and '&' in line:
            cells = line.split('&')
            if len(cells) != 6 or task_index not in range(3):
                raise ValueError(f'Unexpected table row: {line}')
            task = TASKS[task_index]
            input_name = 'T1W Only' if 'T1W Only' in cells[1] else 'T2W Only' if 'T2W Only' in cells[1] else 'T1W+T2W'
            for index, value in enumerate(results[(task, name, input_name)], 2):
                if name == 'Average' and index < 4:
                    value = '\\bf{' + value + '}'
                if '\\color{blue}' in cells[index] and not value.startswith('\\textcolor'):
                    value = '\\color{blue}' + value
                cells[index] = value
            line = '&'.join(cells) + '\\\\'
            updated += 1
        output.append(line)
    if updated != 36 or task_index != 2:
        raise ValueError(f'Expected 36 table rows and three cohorts, got {updated}')
    return '\n'.join(output) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=HERE.parents[1])
    parser.add_argument('--output-dir', type=Path, default=HERE.parents[1] / 'results/tables')
    parser.add_argument('--template', type=Path, default=HERE / 'table_template.tex')
    args = parser.parse_args()
    aliases, groups = cohorts()
    expected = {'2-class': 629, 'low_vs_high': 474, 'histology': 357}
    records, table = [], {}
    for task, cohort in groups.items():
        if len(cohort) != expected[task]:
            raise ValueError(f'Unexpected {task} cohort size: {len(cohort)}')
        truth = (cohort['Ground_truth_Risk_Assessment'] // 2).to_numpy()
        reader_metrics = []
        for reader in range(1, 4):
            name = f'Radiologist {reader}'
            value = metrics(truth, (cohort[f'reader{reader}'] // 2).to_numpy())
            reader_metrics.append(value)
            records.append(dict(task=task, method=name, input='T1W+T2W', evaluation='reader', **value))
            table[(task, name, 'T1W+T2W')] = [f'{value["sens"]:.2f}', f'{value["spec"]:.2f}', '\\textcolor{blue}{-}', '\\textcolor{blue}{-}']
        average = {key: float(np.mean([row[key] for row in reader_metrics])) for key in ['sens', 'spec']}
        records.append(dict(task=task, method='Average', input='T1W+T2W', evaluation='reader',
                            n=len(cohort), positive_n=int(truth.sum()), **average))
        table[(task, 'Average', 'T1W+T2W')] = [f'{average["sens"]:.2f}', f'{average["spec"]:.2f}', '\\textcolor{blue}{-}', '\\textcolor{blue}{-}']
        print(f'{task}: {len(cohort)} paired reader cases, {int(truth.sum())} high risk')
        for name, input_name, method, model_path in METHODS:
            cells = []
            for scope in ['internal', 'external']:
                folder = args.project_root / 'Classification' / scope / task / method / 'saved' / model_path
                cal = predictions(folder / 'result_calibrated.xlsx', aliases, cohort)
                uncal = predictions(folder / 'result.xlsx', aliases, cohort)
                if not np.array_equal(cal['Fold'], uncal['Fold']) or not np.allclose(
                        cal['Probability'], uncal['Probability'], atol=1e-6, rtol=0):
                    raise ValueError(f'Calibrated/uncalibrated probabilities or folds differ: {folder}')
                calibrated = metrics(truth, cal['Prediction'])
                uncalibrated = metrics(truth, uncal['Prediction'])
                record = dict(task=task, method=name, input=input_name, evaluation=scope,
                              **calibrated, uncal_sens=uncalibrated['sens'], uncal_spec=uncalibrated['spec'],
                              uncal_tn=uncalibrated['tn'], uncal_fp=uncalibrated['fp'],
                              uncal_fn=uncalibrated['fn'], uncal_tp=uncalibrated['tp'],
                              calibrated_source=str(folder / 'result_calibrated.xlsx'),
                              uncalibrated_source=str(folder / 'result.xlsx'))
                records.append(record)
                cells.extend(f'{calibrated[key]:.2f}({uncalibrated[key]:.2f})' for key in ['sens', 'spec'])
            table[(task, name, input_name)] = cells
            print(f'  {name} {input_name}: ' + ' & '.join(cells))
    completed = render(args.template.read_text(), table)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / 'radiologist_comparison_updated.tex'
    output.write_text(completed)
    pd.DataFrame(records).to_csv(args.output_dir / 'radiologist_comparison_metrics.csv', index=False)
    (args.output_dir / 'radiologist_comparison_cohorts.json').write_text(json.dumps(
        {task: list(group.index) for task, group in groups.items()}, indent=2) + '\n')
    print(f'Updated 36 table rows from 96 saved prediction files: {output}')


if __name__ == '__main__':
    main()
