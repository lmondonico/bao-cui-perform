"""Resume CALiSol analyses and update study outputs."""

import fcntl
import csv
import io
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from core.study import (
    ROOT, OUTPUT, PLAN, atomic_json, load_data, load_protocols, local_path, read_run, require_training_enabled, run_dir, verify_completion,
)
from data.conversion_scripts.prepare_splits import split_filename


def job_metadata(job, protocols, sources):
    protocol = protocols[job['protocol']]
    physical_unit = sources['datasets'][job['dataset']]['target']['physical_unit']
    test = protocol['files'][split_filename(job['dataset'], 'test')]
    return {'target': job['target'], 'target_unit': physical_unit if job['target'] == 'IC' else protocol['target_unit'],
            'physical_unit': physical_unit, 'transformation': 'identity' if job['target'] == 'IC' else protocol['transformation'],
            'metric_scale': 'native', 'protocol': job['protocol'], 'training_regime': job['regime'],
            'test_observations': str(test['rows']),
            'test_unique_compositions': str(test['compositions']), 'test_chemical_systems': str(test['chemical_systems'])}


def annotate_table_metadata(plan):
    protocols = load_protocols()
    sources = load_data()
    metadata = {job['id']: job_metadata(job, protocols, sources) for job in plan}
    for filename in ('main_benchmark.csv', 'common_subset.csv', 'test_strata.csv', 'all_experiments.csv'):
        path = OUTPUT / 'tables' / filename
        if not path.exists():
            continue
        original = path.read_bytes()
        reader = csv.DictReader(io.StringIO(original.decode()))
        columns, rows = list(reader.fieldnames or []), list(reader)
        for row in rows:
            if filename in ('test_strata.csv', 'all_experiments.csv'):
                values = metadata[row['job_id' if filename == 'test_strata.csv' else 'id']]
            else:
                matches = [job for job in plan if job['dataset'] == row['dataset'] and job['model'] == row['model']
                           and job['regime'] == row['training_regime']]
                if not matches:
                    raise ValueError(f'No planned comparison matches {filename}: {row}')
                values = dict(metadata[matches[0]['id']])
                changing = {'protocol', 'test_observations', 'test_unique_compositions', 'test_chemical_systems'}
                for job in matches[1:]:
                    if any(metadata[job['id']][key] != value for key, value in values.items() if key not in changing):
                        raise ValueError(f'Aggregate mixes target or dataset definitions: {filename}')
                values['protocol'] = ';'.join(sorted({job['protocol'] for job in matches}))
                for key in ('test_observations', 'test_unique_compositions', 'test_chemical_systems'):
                    counts = [int(metadata[job['id']][key]) for job in matches]
                    values[key] = str(min(counts)) if min(counts) == max(counts) else f'{min(counts)}..{max(counts)} across repeats'
            row.update(values)
            columns.extend(key for key in values if key not in columns)
        if not rows:
            continue
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
        replacement = stream.getvalue().encode()
        if replacement != original:
            temporary = path.with_suffix('.csv.tmp')
            temporary.write_bytes(replacement)
            os.replace(temporary, path)
    return 'completed'


def ilthermo_multisalt(plan):
    from experiments.metrics import comparison_scores, scores
    from tasks.training.train_holdout import expected_identity, load_protocol

    tables = OUTPUT / 'tables'
    tables.mkdir(parents=True, exist_ok=True)
    breadth, inputs = [], {}
    planned = [job for job in plan if job['dataset'] == 'ilthermo' and job['regime'] == 'benchmark']
    for job in planned:
        if job['protocol'] in inputs:
            continue
        protocol, split_dir = load_protocol(job['protocol'])
        partitions = {name: pd.read_csv(split_dir / split_filename(job['dataset'], name)) for name in ('train', 'validation', 'test')}
        inputs[job['protocol']] = protocol, partitions
        for name, frame in partitions.items():
            masks = {'full': np.ones(len(frame), dtype=bool), 'single_salt': frame.Salt_count.eq(1),
                     'multi_salt': frame.Salt_count.ge(2), 'two_salts': frame.Salt_count.eq(2), 'three_salts': frame.Salt_count.eq(3)}
            for label, mask in masks.items():
                subset = frame.loc[mask]
                breadth.append({'protocol': job['protocol'], 'seed': job['seed'], 'fold': job['fold'],
                                'partition': name, 'salt_stratum': label, 'observations': len(subset),
                                'unique_compositions': subset.Composition_ID.nunique(), 'chemical_systems': subset.System_ID.nunique()})
    pd.DataFrame(breadth).to_csv(tables / 'ilthermo_split_salt_breadth.csv', index=False)
    state_path = OUTPUT / 'job_ledger.json'
    states = json.loads(state_path.read_text()).get('jobs', {}) if state_path.exists() else {}
    rows = []
    for job in planned:
        protocol, partitions = inputs[job['protocol']]
        state = states.get(job['id'], {})
        row = {key: job.get(key) for key in ('id', 'model', 'seed', 'run', 'fold', 'target', 'protocol', 'regime')}
        row.update(status=state.get('status', job.get('status', 'queued')), reason=state.get('reason', job.get('reason', '')),
                   dataset='ilthermo', training_dataset='ilthermo_full', evaluation_subset='test_Salt_count>=2')
        directory = run_dir(job)
        if row['status'] == 'completed' and 'completion' not in read_run(directory):
            row.update(status='failed', reason='Missing verified completion record')
        if 'completion' in read_run(directory):
            try:
                verify_completion(directory, expected_identity(job, protocol))
                predictions = pd.read_csv(directory / 'predictions_with_provenance.csv')
                test = predictions.loc[predictions.Task.eq('Test')].sort_values('Row_ID')
                expected = partitions['test'].sort_values('Row_ID')
                if test.Row_ID.duplicated().any() or not np.array_equal(test.Row_ID, expected.Row_ID):
                    raise ValueError('Full-trained predictions do not match the complete parent test partition')
                for column in ('Salt_count', 'Composition_ID', 'System_ID'):
                    if not np.array_equal(test[column], expected[column]):
                        raise ValueError(f'Prediction {column} differs from frozen membership')
                subset = test.loc[test.Salt_count.ge(2)]
                if not len(subset):
                    raise ValueError('No retained multi-salt test observations')
                row.update(scores(subset), **{'physical_' + key: value for key, value in scores(subset, physical=True).items() if key in ('rmse', 'mae', 'r2')})
                row.update(status='completed', reason='', chemical_systems=subset.System_ID.nunique(),
                           target_unit=subset.Target_unit.iloc[0], physical_unit=subset.Physical_unit.iloc[0], transformation=subset.Target_transform.iloc[0])
            except (OSError, ValueError, RuntimeError, KeyError) as error:
                row.update(status='failed', reason=f'{type(error).__name__}: {error}')
        rows.append(row)
    if len(rows) != 8 * 5:
        raise ValueError('Expected eight ILThermo full models with five folds each')
    frame = pd.DataFrame(rows)
    frame.to_csv(tables / 'ilthermo_multisalt.csv', index=False)
    comparison_scores(frame).assign(evaluation_subset='test_Salt_count>=2').to_csv(
        tables / 'ilthermo_multisalt_summary.csv', index=False)
    return 'completed' if all(row['status'] == 'completed' for row in rows) else 'blocked' if any(row['status'] in ('failed', 'blocked') for row in rows) else 'queued'


def write_study_status(plan, ledger):
    source_path = PLAN
    source = json.loads(source_path.read_text())
    job_path = OUTPUT / 'job_ledger.json'
    jobs = json.loads(job_path.read_text()).get('jobs', {}) if job_path.exists() else {}
    planned = {job['id']: job for job in plan}
    analyses = ledger.get('analyses', {})
    table_names = {'five_task_benchmark': ['main_benchmark.csv', 'main_benchmark_wide.csv'],
                   'common_subset': ['common_subset.csv'],
                   'cv_ic': ['calisol_folds.csv', 'repeat_metrics.csv'],
                   'cv_lic': ['calisol_folds.csv', 'repeat_metrics.csv'],
                   'ic_vs_lic': ['ic_lic_native_and_physical.csv'], 'composition_leakage': ['test_strata.csv'],
                   'strict_lmo': ['all_experiments.csv', 'strict_lmo_summary.csv', 'test_strata.csv'], 'temperature_bins': ['test_strata.csv'],
                   'runtime': ['configuration_and_runtime.csv'],
                   'salt_breakdown': ['test_strata.csv', 'stratum_summary.csv', 'ilthermo_multisalt.csv', 'ilthermo_multisalt_summary.csv', 'ilthermo_split_salt_breadth.csv'],
                   'component_breakdown': ['test_strata.csv', 'stratum_summary.csv'], 'coverage': ['coverage_and_capability.csv']}
    analysis_names = {'temperature_controls': ['temperature_controls'], 'attention_maps': ['attention'],
                      'attention_fixed_prior': ['attention'], 'parameter_counts': ['computational_runtime'],
                      'runtime': ['computational_runtime']}
    figure_names = {'five_task_benchmark': ['parity_plot_perform_mosaic', 'parity_plot_perform_variants'],
                    'common_subset': ['parity_plot_perform_variants'],
                    'cv_ic': ['cv_folds_IC'], 'cv_lic': ['cv_folds_LIC'], 'ic_vs_lic': ['parity_comparison_LIC_IC'],
                    'temperature_parity': ['temperature_trend/temperature_parity_comparison'],
                    'attention_maps': ['attention-study/moiety_attention']}
    rows = []
    for study in source['studies']:
        name = study['id']
        paths = [OUTPUT / 'tables' / filename for filename in table_names.get(name, [])]
        dependency_states = {job: jobs.get(job, {'status': planned[job].get('status', 'queued'),
                                               'reason': planned[job].get('reason', '')})
                             for job in study['required_training_jobs']}
        if name.startswith('component_') and name != 'component_breakdown':
            paths.append(OUTPUT / 'tables/component_ablations.csv')
            if name != 'component_no_relu':
                figure_names[name] = ['ablation_parity_mosaic', 'ablation_loss_mosaic']
        if name.startswith('mechanism_'):
            paths.append(OUTPUT / 'tables/mechanism_ablations.csv')
        paths += [run_dir(planned[job]) for job in dependency_states]
        analysis_states = {key: analyses.get(key, {'status': 'queued', 'reason': 'Analysis not yet executed'})
                           for key in analysis_names.get(name, [])}
        paths += [ROOT / state['output'] for state in analysis_states.values() if state.get('output')]
        figure_paths = [ROOT / 'figures' / f'{figure}.png' for figure in figure_names.get(name, [])]
        figure_states = {str(path): {'status': 'completed' if path.exists() else 'queued'} for path in figure_paths}
        paths += figure_paths
        statuses = [state['status'] for state in dependency_states.values()]
        if any(status in ('failed', 'blocked') for status in statuses):
            status = 'blocked'
        elif 'running' in statuses:
            status = 'running'
        elif any(status != 'completed' for status in statuses):
            status = 'queued'
        else:
            downstream = [state['status'] for state in (*analysis_states.values(), *figure_states.values())]
            status = 'blocked' if any(value in ('failed', 'blocked') for value in downstream) else 'queued' if any(value != 'completed' for value in downstream) else 'completed'
        reason = ''
        if name == 'coverage':
            paths += [ROOT / 'config/study.json', OUTPUT / 'compatibility/compatibility_matrix.csv']
        elif name == 'runtime':
            path = OUTPUT / 'analyses/computational_runtime/runtime_stats.json'
            if path.exists():
                stats = json.loads(path.read_text())
                if stats['training']['peak_memory_gb'] is None:
                    status, reason = 'blocked', stats['training']['peak_memory_limitation']
        elif name == 'parameter_counts':
            paths.append(OUTPUT / 'analyses/computational_runtime/parameter_breakdown.csv')
        if not paths:
            raise ValueError(f'No concrete output mapping for study {name}')
        artifacts = {str(path.relative_to(ROOT)): {'exists': path.exists(), 'kind': 'directory' if path.is_dir() else 'file'} for path in paths}
        missing = [path for path, value in artifacts.items() if not value['exists']]
        if status == 'completed' and missing:
            status, reason = 'blocked', 'Expected artifacts have not been produced'
        rows.append({'id': name, 'study': study['study'], 'status': status, 'reason': reason,
                     'artifacts': artifacts, 'missing_artifacts': missing,
                     'training_dependencies': {job: {key: state[key] for key in ('status', 'reason') if key in state} for job, state in dependency_states.items()},
                     'analysis_dependencies': analysis_states, 'figure_dependencies': figure_states})
    atomic_json(OUTPUT / 'study_status.json', {'updated_utc': datetime.now(timezone.utc).isoformat(),
                'study_count': len(rows), 'complete': all(row['status'] == 'completed' for row in rows),
                'studies': rows})


def postprocess(plan, ledger, output, plan_only=False):
    for name, function in [('ilthermo_multisalt', lambda: ilthermo_multisalt(plan)),
                           ('table_metadata', lambda: annotate_table_metadata(plan)),
                           ('figures', lambda: subprocess_logged([sys.executable, '-s', '-B', str(ROOT / 'figures/plot_figures.py')], output / 'figures.log'))]:
        if plan_only:
            ledger['analyses'].setdefault(name, {'status': 'queued', 'reason': 'Awaiting postprocessing'})
            continue
        record = {'status': 'running'}
        ledger['analyses'][name] = record
        atomic_json(OUTPUT / 'analysis_ledger.json', ledger)
        try:
            status = function()
            record['status'] = status or 'completed'
            if name == 'figures':
                record['output'] = 'figures'
            elif name == 'table_metadata':
                record['output'] = str((OUTPUT / 'tables').relative_to(ROOT))
            else:
                record['output'] = str((OUTPUT / 'tables/ilthermo_multisalt.csv').relative_to(ROOT))
        except Exception as error:
            record.update(status='failed', reason=f'{type(error).__name__}: {error}')
        atomic_json(OUTPUT / 'analysis_ledger.json', ledger)
    write_study_status(plan, ledger)


def ready_run(job, protocols):
    directory = run_dir(job)
    identity = read_run(directory)['identity']
    if identity.get('smoke') or identity['dataset'] != 'calisol':
        raise ValueError('Only new full CALiSol experiments may enter these analyses')
    for key in ('model', 'target', 'seed', 'run', 'fold'):
        if identity[key] != job[key]:
            raise ValueError(f'Checkpoint {key} differs from planned experiment')
    protocol = protocols[job['protocol']]
    if identity['split'] != job['protocol']:
        raise ValueError('Checkpoint protocol differs from planned experiment')
    from tasks.training.train_holdout import expected_identity
    verify_completion(directory, expected_identity(job, protocol))
    return directory


def representative_job(jobs):
    if len(jobs) != 5 or {(job['seed'], job['fold']) for job in jobs} != {(42, fold) for fold in range(1, 6)}:
        raise ValueError('Representative analysis requires five seed-42 folds')
    return next(job for job in jobs if job['seed'] == 42 and job['fold'] == 1)


def summarize_fits(frame, directory, metrics, groups=()):
    frame.to_csv(directory / 'fold_metrics.csv', index=False)
    keys = [*groups, 'seed']
    grouped = frame.groupby(keys, dropna=False)
    if not grouped.fold.apply(lambda values: set(values) == set(range(1, 6)) and len(values) == 5).all():
        raise ValueError('Every analysis repeat must contain exactly five fold metrics')
    repeats = grouped[list(metrics)].agg(lambda values: values.mean(skipna=False)).reset_index()
    repeats.to_csv(directory / 'repeat_metrics.csv', index=False)
    rows = []
    subsets = frame.groupby(list(groups), dropna=False) if groups else [((), frame)]
    for key, subset in subsets:
        if set(subset.seed) != {42}:
            raise ValueError('Analysis summaries require the seed-42 folds')
        labels = dict(zip(groups, key if isinstance(key, tuple) else (key,)))
        for metric in metrics:
            rows.append({**labels, 'metric': metric, 'mean': subset[metric].mean(skipna=False),
                         'std': subset[metric].std(ddof=1, skipna=False), 'std_kind': 'fold', 'folds': 5})
    pd.DataFrame(rows).to_csv(directory / 'summary.csv', index=False)


def temperature_comparison(frame, jobs, directory):
    full = {job['fold']: job for job in jobs if job['model'] == 'perform'}
    no_temperature = {job['fold']: job for job in jobs if job['model'] == 'no_temperature'}
    if len(jobs) != 10 or set(full) != set(range(1, 6)) or set(no_temperature) != set(full):
        raise ValueError('Temperature comparison requires five full and five no-temperature fits')
    if any(job['seed'] != 42 or job['dataset'] != 'calisol' or job['target'] != 'LIC' for job in jobs):
        raise ValueError('Temperature comparison requires CALiSol LIC seed 42')
    if any(full[fold]['protocol'] != no_temperature[fold]['protocol'] for fold in full):
        raise ValueError('Temperature comparison splits differ')
    matched = frame.loc[(frame.seed == 42) & (frame.split == 'test')].copy()
    if set(matched.model) != {'arrhenius', 'xgboost_temperature', 'temperature_mlp'}:
        raise ValueError('Temperature comparison requires all three existing controls')
    if any(row.job != full[row.fold]['id'] for row in matched.itertuples()):
        raise ValueError('Temperature controls do not use the full-model comparison folds')
    rows = matched.to_dict('records')
    for job in jobs:
        saved = read_run(run_dir(job))['metrics']
        score = saved['splits']['Test']['native']
        rows.append({'model': job['model'], 'fold': job['fold'], 'n': score['observations'],
                     'unique_compositions': score['unique_compositions'],
                     'RMSE': score['rmse'], 'MAE': score['mae'], 'R2': score['r2']})
    comparisons = []
    combined = pd.DataFrame(rows)
    if combined.n.nunique() != 1 or combined.unique_compositions.nunique() != 1:
        raise ValueError('Temperature comparison test counts differ between models')
    for model, group in combined.groupby('model'):
        if len(group) != 5 or set(group.fold) != set(full) or group.n.nunique() != 1 or group.unique_compositions.nunique() != 1:
            raise ValueError('Temperature comparison has incomplete or mismatched fold metrics')
        for metric in ('RMSE', 'MAE', 'R2'):
            comparisons.append({'model': model, 'split': 'test', 'metric': metric,
                                'mean': group[metric].mean(skipna=False), 'std': group[metric].std(ddof=1, skipna=False),
                                'std_kind': 'fold', 'seed': 42, 'folds': 5,
                                'observations': group.n.iloc[0], 'unique_compositions': group.unique_compositions.iloc[0],
                                'comparison': 'matched_seed42'})
    summary = pd.read_csv(directory / 'summary.csv').assign(comparison='controls')
    pd.concat([summary, pd.DataFrame(comparisons)], ignore_index=True).to_csv(directory / 'summary.csv', index=False)


def subprocess_logged(command, log):
    environment = dict(os.environ, PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    with log.open('a') as handle:
        handle.write(' '.join(map(str, command)) + '\n')
        handle.flush()
        subprocess.run(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, check=True)


def run_all(args):
    if not args.plan_only:
        require_training_enabled()
    output = local_path(args.output, OUTPUT)
    output.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT / 'analysis.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    data_root = local_path(args.data_root, ROOT / 'data')
    from data.conversion_scripts.prepare_splits import verify
    verify(data_root)
    protocols = load_protocols()
    plan = json.loads((ROOT / 'config/study.json').read_text())['jobs']
    calisol = [job for job in plan if job['dataset'] == 'calisol']
    available, directories = {}, {}
    for job in calisol:
        try:
            directories[job['id']] = ready_run(job, protocols)
            available[job['id']] = None
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            available[job['id']] = f"{job['id']}: {job.get('reason') or str(error)}"
    tasks = [dict(id='temperature_controls', method='controls', jobs=[])]
    folds = [job for job in calisol if job['model'] == 'perform' and job['target'] == 'LIC'
             and job['regime'] in ('benchmark', 'ic_lic')]
    representative_job(folds)
    tasks.append(dict(id='attention', method='attention', jobs=folds, representative=True))
    tasks.append(dict(id='computational_runtime', method='runtime', jobs=folds, representative=True))
    main = [job for job in calisol if job['model'] == 'perform' and job['regime'] == 'benchmark']
    representative_job(main)
    tasks[0]['fits'] = main
    tasks[0]['jobs'] = [job for job in main if job['seed'] == 42] + [job for job in calisol
        if job['model'] == 'no_temperature' and job['regime'] == 'component' and job['seed'] == 42]
    ledger_path = OUTPUT / 'analysis_ledger.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {'analyses': {}}
    for task in tasks:
        name = task['id']
        directory = output / name
        dependencies = {job['id']: available[job['id']] for job in task['jobs']}
        record = {'status': 'queued', 'output': str(directory.relative_to(ROOT)), 'dependencies': dependencies}
        ledger['analyses'][name] = record
        failures = [reason for reason in dependencies.values() if reason]
        if failures:
            record.update(status='blocked', reason='; '.join(failures))
            atomic_json(ledger_path, ledger)
            continue
        if task['method'] == 'runtime':
            training = json.loads((OUTPUT / 'job_ledger.json').read_text())
            if any(job.get('status') == 'running' for job in training['jobs'].values()):
                record['reason'] = 'Computational profiling waits until all training workers are idle'
                atomic_json(ledger_path, ledger)
                continue
        identity = {'analysis': name, 'dependencies': {job['id']: read_run(directories[job['id']])['completion']['elapsed_seconds']
                                                       for job in task['jobs']}}
        marker = directory / 'analysis_completed.json'
        if marker.exists():
            complete = json.loads(marker.read_text())
            if complete['identity'] == identity and all((directory / path).is_file() for path in complete['files']):
                record.update(status='completed', elapsed_seconds=complete['elapsed_seconds'])
                atomic_json(ledger_path, ledger)
                continue
        if args.plan_only:
            atomic_json(ledger_path, ledger)
            continue
        if directory.exists():
            previous = directory.with_name(directory.name + '_incomplete_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
            directory.rename(previous)
            record['preserved_previous_attempt'] = str(previous.relative_to(ROOT))
        log = output / f'{name}.log'
        record.update(status='running', started_utc=datetime.now(timezone.utc).isoformat(), log=str(log.relative_to(ROOT)))
        atomic_json(ledger_path, ledger)
        started = time.monotonic()
        try:
            command = [sys.executable, '-s', '-B']
            if task['method'] == 'controls':
                directory.mkdir()
                rows = []
                fits = task.get('fits', task['jobs'])
                for job in sorted(fits, key=lambda value: (value['seed'], value['fold'])):
                    fit_dir = directory / f"seed{job['seed']}_fold{job['fold']}"
                    run_command = command + [str(ROOT / 'experiments/analyses.py'), task['method'], '--data-root', str(data_root),
                        '--protocol', job['protocol'], '--seed', str(job['seed']), '--output', str(fit_dir), '--device', args.device]
                    subprocess_logged(run_command, log)
                    frame = pd.read_csv(fit_dir / 'metrics.csv')
                    rows.append(frame.assign(seed=job['seed'], run=job['run'], fold=job['fold'], job=job['id']))
                frame = pd.concat(rows, ignore_index=True)
                summarize_fits(frame, directory, ('RMSE', 'MAE', 'R2'), ('model', 'split'))
                temperature_comparison(frame, task['jobs'], directory)
            else:
                run_command = command + [str(ROOT / 'experiments/analyses.py'), task['method'], '--data-root', str(data_root), '--output', str(directory), '--device', args.device]
                if task.get('representative'):
                    selected = representative_job(task['jobs'])
                    record['selection'] = {'rule': 'Prespecified seed 42, fold 1; all fifteen fits must be complete. No score-based selection.', 'job': selected['id']}
                    run_command += ['--protocol', selected['protocol'], '--checkpoint', str(directories[selected['id']] / 'best_model.pth')]
                subprocess_logged(run_command, log)
            complete = {'identity': identity, 'elapsed_seconds': time.monotonic() - started,
                        'selection': record.get('selection'), 'files': sorted(str(path.relative_to(directory)) for path in directory.rglob('*') if path.is_file())}
            atomic_json(marker, complete)
            record.update(status='completed', elapsed_seconds=complete['elapsed_seconds'])
        except Exception as error:
            record.update(status='failed', elapsed_seconds=time.monotonic() - started, reason=f'{type(error).__name__}: {error}')
        atomic_json(ledger_path, ledger)
    postprocess(plan, ledger, output, plan_only=args.plan_only)
    ledger['updated_utc'] = datetime.now(timezone.utc).isoformat()
    atomic_json(ledger_path, ledger)
    counts = Counter(value['status'] for value in ledger['analyses'].values())
    print('Analyses: ' + ', '.join(f'{count} {status}' for status, count in sorted(counts.items())))
