"""Write the manuscript summary tables (experiments/summary/) from the verified metrics.py tables."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.study import OUTPUT

TABLES, SUMMARY = OUTPUT / 'tables', ROOT / 'experiments/summary'
PLAN = json.loads((ROOT / 'config/study.json').read_text())
NAMES = PLAN['model_names']
DATASETS = {'calisol': 'CALiSol', 'chew_density': 'CheW density', 'chew_enthalpy': 'CheW enthalpy', 'des': 'DES',
            'ilthermo': 'ILThermo', 'des_common': 'DES-common', 'ilthermo_common': 'ILThermo-common'}
ROLES = {'calisol': 'benchmark; ablations and analyses', 'des_common': 'common subset of DES',
         'ilthermo_common': 'common subset of ILThermo'}
DIGITS = {'chew_density': 4}
VARIANTS = {
    'perform': ('Reference', 'Full PerForm'),
    'no_relu': ('Component', 'No ReLU in the MOTIF global (distance) attention branch'),
    'no_attention': ('Component', 'Formulation attention between molecules removed'),
    'no_motif': ('Component', 'MOTIF molecular encoder replaced by a token embedding'),
    'no_molar_ratio': ('Component', 'Molar-ratio encoder removed'),
    'no_temperature': ('Component', 'Temperature encoder removed'),
    'minimal': ('Component', 'MOTIF encoder, pooling and head only (no molar ratio, temperature or formulation attention)'),
    'additive_distance': ('Mechanism', 'Additive distance bias in the MOTIF global branch instead of multiplicative weighting'),
    'role_assigned': ('Mechanism', 'Learned solvent/salt role embedding added before formulation attention'),
    'vanilla': ('Mechanism', 'Plain scaled dot-product MOTIF attention (no adjacency or distance)'),
    'hadamard': ('Mechanism', 'Element-wise adjacency masking in the MOTIF local branch'),
    'deep_sets': ('Mechanism', 'Deep Sets aggregator instead of formulation attention'),
    'temp_concat': ('Mechanism', '[T, 1/T] concatenated instead of the temperature encoder'),
    'mr_concat': ('Mechanism', 'Raw molar ratio concatenated instead of the molar-ratio encoder'),
}
STUDIES = {'benchmark': 'Main benchmark (CALiSol, ILThermo, DES, CheW density, CheW enthalpy)',
           'common_subset': 'Common subsets (ILThermo-common, DES-common)', 'component': 'CALiSol component ablations',
           'mechanism': 'CALiSol mechanism ablations', 'ic_lic': 'CALiSol PerForm, IC target',
           'strict_lmo': 'CALiSol PerForm, strict composition hold-out (LIC)'}
REASONS = {'single_salt_restriction': 'more than one salt', 'requires_salt_branch': 'no salt',
           'requires_neutral_branch': 'no neutral component', 'unsupported_element_features': 'element outside GeoMix set'}


def cell(mean, sd, digits=3):
    return f'{mean:.{digits}f} ± {sd:.{digits}f}'


def span(values):
    return f'{min(values)}–{max(values)}' if min(values) != max(values) else str(min(values))


def dataset_table(runs, compatibility):
    rows = []
    for name, label in DATASETS.items():
        spec = PLAN['data']['datasets'][name]
        frame = pd.read_csv(ROOT / 'data' / spec.get('benchmark_path', spec['path']), low_memory=False)
        sizes = [[len(pd.read_csv(ROOT / PLAN['protocols'][f'cv_{name}_fold{k}']['split_dir'] / f'{name}_{part}.csv', usecols=['Row_ID']))
                  for k in range(1, 6)] for part in ('train', 'validation', 'test')]
        parts = frame.filter(regex=r'^(Solvent|Salt)_\d$').stack().astype(str)
        trained = compatibility.loc[(compatibility.dataset == name) & (compatibility.status != 'blocked'), 'model']
        rows.append({'dataset': label, 'role': ROLES.get(name, 'benchmark'), 'target': spec['target']['unit'],
                     'physical_unit': spec['target']['physical_unit'], 'rows': len(frame),
                     'compositions': frame.Composition_ID.nunique(), 'systems': frame.System_ID.nunique(),
                     'molecules': parts.str.rsplit(':', n=1).str[0].nunique(), 'salts_per_row': span(frame.Salt_count),
                     'components_per_row': span(frame.Component_count),
                     'temperature_c': f'{frame.Temperature.min():.1f} to {frame.Temperature.max():.1f}',
                     'train_rows': span(sizes[0]), 'validation_rows': span(sizes[1]), 'test_rows': span(sizes[2]),
                     'models_trained': len(trained), 'runs': int((runs.dataset == name).sum())})
    return pd.DataFrame(rows)


def matrix(groups, metric):
    frame = groups.loc[groups.regime.isin(['benchmark', 'common_subset'])]
    table = {}
    for name, label in DATASETS.items():
        part = frame.loc[frame.dataset == name].set_index('model')
        table[label] = {NAMES[m]: cell(r[metric], r[f'{metric}_fold_std'], DIGITS.get(name, 3)) if r.status == 'completed' else 'blocked'
                        for m, r in part.iterrows()}
    return pd.DataFrame(table).loc[[NAMES[m] for m in PLAN['models']]].rename_axis('model')


def variant_table(groups):
    part = groups.loc[(groups.dataset == 'calisol') & (groups.target == 'LIC')
                      & ((groups.regime.isin(['component', 'mechanism'])) | ((groups.regime == 'benchmark') & (groups.model == 'perform')))]
    rows = [{'study': VARIANTS[r.model][0], 'variant': r.model, 'change': VARIANTS[r.model][1],
             'test_r2': cell(r.r2, r.r2_fold_std), 'test_rmse_log10_mS_cm': cell(r.rmse, r.rmse_fold_std), 'order': -r.r2}
            for r in part.itertuples()]
    order = {'Reference': 0, 'Component': 1, 'Mechanism': 2}
    return pd.DataFrame(rows).sort_values(['study', 'order'], key=lambda s: s.map(order) if s.name == 'study' else s).drop(columns='order')


def target_table(groups):
    get = lambda regime, target: groups.loc[(groups.dataset == 'calisol') & (groups.model == 'perform')
                                            & (groups.regime == regime) & (groups.target == target)].iloc[0]
    rows = []
    for label, regime, target, physical in (('LIC target, CV', 'benchmark', 'LIC', False), ('LIC target, CV', 'benchmark', 'LIC', True),
                                            ('IC target, CV', 'ic_lic', 'IC', True),
                                            ('LIC target, strict LMO (test compositions held out)', 'strict_lmo', 'LIC', False),
                                            ('LIC target, strict LMO (test compositions held out)', 'strict_lmo', 'LIC', True)):
        r, p = get(regime, target), 'physical_' if physical and target == 'LIC' else ''
        rows.append({'run': label, 'scored_in': 'mS/cm' if physical else 'log10(mS/cm)', 'test_rows': int(r.observations),
                     'test_r2': cell(r[p + 'r2'], r[p + 'r2_fold_std']), 'test_rmse': cell(r[p + 'rmse'], r[p + 'rmse_fold_std'])})
    return pd.DataFrame(rows)


def fold_table(folds, regime, target):
    part = folds.loc[(folds.model == 'perform') & (folds.regime == regime) & (folds.target == target)].sort_values('fold')
    if list(part.fold) != [1, 2, 3, 4, 5]:
        raise SystemExit(f'Expected five CALiSol PerForm {target} folds')
    rows = []
    for split, prefix in (('Training', 'train_'), ('Validation', 'validation_'), ('Test', '')):
        row = {'split': split, 'rows': span(part[prefix + 'observations'])}
        for metric in ('rmse', 'r2'):
            values = part[prefix + metric]
            row.update({f'fold_{f}_{metric}': f'{v:.3f}' for f, v in zip(part.fold, values)})
            row[f'{metric}_mean_sd'] = cell(values.mean(), values.std(ddof=1))
        rows.append(row)
    order = ['split', 'rows'] + [f'fold_{f}_{m}' for f in range(1, 6) for m in ('rmse', 'r2')] + ['rmse_mean_sd', 'r2_mean_sd']
    return pd.DataFrame(rows)[order]


def main():
    groups = pd.read_csv(TABLES / 'repeat_metrics.csv')
    runs = pd.read_csv(TABLES / 'all_experiments.csv')
    if not runs.status.isin(['completed', 'blocked']).all():
        raise SystemExit('Run experiments/metrics.py on a complete study first')
    compatibility = pd.read_csv(OUTPUT / 'compatibility/compatibility_matrix.csv')
    SUMMARY.mkdir(exist_ok=True)
    done = runs.loc[runs.status == 'completed']
    tables = {}
    tables['datasets'] = dataset_table(done, compatibility)
    counts = runs.groupby('regime').agg(datasets=('dataset', 'nunique'), models=('model', 'nunique'),
                                        planned=('id', 'size'), trained=('status', lambda s: (s == 'completed').sum()))
    counts = counts.assign(blocked=counts.planned - counts.trained).loc[list(STUDIES)].reset_index()
    counts.insert(0, 'study', counts.regime.map(STUDIES))
    total = {'study': 'Total', 'regime': '', 'datasets': runs.dataset.nunique(), 'models': runs.model.nunique(),
             **counts[['planned', 'trained', 'blocked']].sum()}
    tables['run_counts'] = pd.concat([counts, pd.DataFrame([total])], ignore_index=True)
    blocked = compatibility.loc[compatibility.status == 'blocked']
    tables['blocked_pairs'] = pd.DataFrame({
        'dataset': blocked.dataset.map(DATASETS), 'model': blocked.model.map(NAMES), 'rows': blocked.observations,
        'rows_not_representable': blocked.affected_observations,
        'rows_by_reason': blocked.reasons.map(lambda s: '; '.join(f'{REASONS[k]}: {v}' for k, v in json.loads(s).items()))})
    tables['benchmark_test_r2'], tables['benchmark_test_rmse'] = matrix(groups, 'r2'), matrix(groups, 'rmse')
    tables['calisol_perform_variants'] = variant_table(groups)
    tables['calisol_ic_lic_strict_lmo'] = target_table(groups)
    folds = pd.read_csv(TABLES / 'calisol_folds.csv')
    tables['calisol_perform_folds_LIC'] = fold_table(folds, 'benchmark', 'LIC')
    tables['calisol_perform_folds_IC'] = fold_table(folds, 'ic_lic', 'IC')
    keep = ['dataset', 'regime', 'model', 'target', 'target_unit', 'physical_unit', 'observations']
    metrics = [f'{p}{m}' for p in ('', 'physical_') for m in ('r2', 'rmse', 'mae')]
    tables['group_metrics'] = groups.loc[groups.status == 'completed', keep + [c for m in metrics for c in (m, m + '_fold_std')]]
    tables['fold_metrics'] = done[['id', 'dataset', 'regime', 'model', 'target', 'fold', 'observations', 'target_unit', 'physical_unit', *metrics]]
    for name, frame in tables.items():
        frame.to_csv(SUMMARY / f'{name}.csv', index=name.startswith('benchmark'))
    print(f'Wrote {len(tables)} tables to {SUMMARY.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
