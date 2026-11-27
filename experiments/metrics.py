"""Regenerate public-study metrics and plots from identified predictions."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from core.study import OUTPUT, atomic_json, read_run, run_dir, update_run, verify_completion
from data.conversion_scripts.prepare_splits import split_filename

METRICS = ('rmse', 'mae', 'r2', 'physical_rmse', 'physical_mae', 'physical_r2')
TRAINING_METRICS = tuple(f'{part}_{metric}' for part in ('train', 'validation') for metric in METRICS)


def repeat_scores(frame):
    rows = []
    keys = ['dataset', 'model', 'target', 'regime', 'run', 'seed']
    metrics = METRICS + tuple(name for name in TRAINING_METRICS if name in frame)
    for values, group in frame.groupby(keys, dropna=False):
        done = group.loc[group.status == 'completed']
        if len(group) != 5 or set(group.fold) != {1, 2, 3, 4, 5}:
            raise ValueError('Expected exactly five distinct planned folds per repeat')
        errors = [name for name in metrics if name.endswith(('rmse', 'mae'))]
        if len(done) and not np.isfinite(done[errors].to_numpy()).all():
            raise ValueError('Completed fold has missing or nonfinite error metrics')
        row = dict(zip(keys, values))
        row.update(planned_folds=len(group), completed_folds=len(done),
                   status='completed' if len(done) == len(group) == 5 else 'incomplete')
        for metric in metrics:
            row[metric] = done[metric].mean(skipna=False) if row['status'] == 'completed' else None
            row[metric + '_fold_std'] = done[metric].std(ddof=1, skipna=False) if row['status'] == 'completed' else None
        for field in ('observations', 'unique_compositions', 'target_unit', 'physical_unit', 'transformation'):
            if len(done) and field in done:
                if done[field].nunique() != 1:
                    raise ValueError(f'Fold test metadata differs within repeat: {field}')
                row[field] = done[field].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def comparison_scores(frame):
    repeats = repeat_scores(frame)
    metrics = METRICS + tuple(name for name in TRAINING_METRICS if name in frame)
    rows = []
    keys = ['dataset', 'model', 'target', 'regime']
    for values, group in repeats.groupby(keys):
        done = group.loc[group.status == 'completed']
        row = dict(zip(keys, values))
        source = frame
        for key, value in row.items():
            source = source.loc[source[key] == value]
        row.update(training_regime=row['regime'], planned_repeats=len(group), completed_repeats=len(done),
                   planned_fits=int(group.planned_folds.sum()), completed_fits=int(group.completed_folds.sum()),
                   status='complete' if len(done) == len(group) else ';'.join(sorted(set(source.status))),
                   reason=';'.join(sorted({str(v) for v in source.reason if pd.notna(v) and str(v)})))
        for metric in metrics:
            row[metric] = done[metric].mean(skipna=False) if len(done) else None
            row[metric + '_std'] = done[metric + '_fold_std'].iloc[0] if len(done) == 1 else None
        for field in ('observations', 'unique_compositions'):
            if len(done):
                row[field + '_min'] = int(done[field].min())
                row[field + '_max'] = int(done[field].max())
        for field in ('target_unit', 'physical_unit', 'transformation'):
            if len(done) and field in done:
                if done[field].nunique() != 1:
                    raise ValueError(f'Target definition differs between repeats: {field}')
                row[field] = done[field].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def stratum_scores(frame, strata):
    summaries = []
    if not len(strata):
        return pd.DataFrame()
    fixed = strata.loc[strata.stratum.isin(['Salt_count', 'Component_count'])]
    metrics = [*METRICS, 'observations', 'unique_compositions']
    for (dataset, stratum, value), group in fixed.groupby(['dataset', 'stratum', 'value']):
        planned = frame.loc[(frame.dataset == dataset) & (frame.regime == 'benchmark')]
        if not len(planned):
            continue
        planned = planned.drop(columns=[*metrics, *TRAINING_METRICS], errors='ignore')
        measured = group.loc[group.regime == 'benchmark', ['job_id', *metrics]].rename(columns={'job_id': 'id'})
        joined = planned.merge(measured, on='id', how='left', validate='one_to_one')
        joined.loc[(joined.status == 'completed') & joined.observations.isna(), 'status'] = 'empty_stratum'
        summaries.append(comparison_scores(joined).assign(stratum=stratum, value=value))
    return pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()


def scores(frame, physical=False):
    truth, prediction = frame.Truth.to_numpy(), frame.Prediction.to_numpy()
    if physical:
        transform = frame.Target_transform.iloc[0]
        if transform == 'log10':
            truth, prediction = np.power(10.,truth), np.power(10.,prediction)
        elif transform in ('ln', 'log'):
            truth, prediction = np.exp(truth), np.exp(prediction)
        elif transform != 'identity':
            raise ValueError(f'Unknown inverse target transform: {transform}')
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError('Nonfinite truth/prediction on requested scale')
    return {'rmse':float(mean_squared_error(truth,prediction)**.5),
            'mae':float(mean_absolute_error(truth,prediction)),
            'r2':float(r2_score(truth,prediction)) if len(frame)>1 and np.var(truth)>0 else None,
            'observations':len(frame),'unique_compositions':frame.Composition_ID.nunique()}


def verify_run(out, split_dir, job, smoke=False, render=True):
    predictions = pd.read_csv(out/'regr_predictions.csv')
    if not {'Task','Row_ID','Truth','Prediction'} <= set(predictions):
        raise ValueError('Missing prediction identities')
    if not set(predictions.Task.str.lower()) <= {'training','validation','test'}:
        raise ValueError('Unexpected prediction partition')
    joined = []
    for task, part in [('Training','train'),('Validation','validation'),('Test','test')]:
        source = pd.read_csv(split_dir / split_filename(job['dataset'], part))
        if smoke:
            source=source.iloc[:128]
        pred=predictions.loc[predictions.Task.str.lower()==task.lower(), ['Task','Row_ID','Truth','Prediction']].copy()
        if pred.Row_ID.duplicated().any() or set(pred.Row_ID)!=set(source.Row_ID):
            raise ValueError(f'Prediction membership differs from frozen {task} records')
        frame=pred.merge(source,on='Row_ID',how='inner',validate='one_to_one')
        if not np.allclose(frame.Truth,frame[job['target']],rtol=1e-5,atol=1e-6):
            raise ValueError(f'Prediction labels differ from frozen {task} targets')
        frame['Truth']=frame[job['target']].to_numpy()
        if job['target']=='IC':
            frame['Target_transform']='identity'
            frame['Target_unit']=frame['Physical_unit']
        if not np.isfinite(frame[['Truth','Prediction']].to_numpy()).all():
            raise ValueError('Nonfinite predictions')
        frame['Task']=task
        joined.append(frame)
    frame=pd.concat(joined,ignore_index=True)
    by_split={task:{'native':scores(part),'physical':scores(part,True)}
              for task,part in frame.groupby('Task')}
    if not render:
        if read_run(out)['metrics']['splits'] != by_split:
            raise RuntimeError('Saved metrics differ from regenerated predictions')
        return by_split
    frame.to_csv(out/'predictions_with_provenance.csv',index=False)
    update_run(out,metrics={'target':job['target'],
               'target_unit':frame.Target_unit.iloc[0],'physical_unit':frame.Physical_unit.iloc[0],
               'transformation':frame.Target_transform.iloc[0],'splits':by_split})
    training_comps=set(frame.loc[frame.Task=='Training','Composition_ID'])
    test=frame.loc[frame.Task=='Test'].copy()
    test['composition_seen']=test.Composition_ID.isin(training_comps)
    test['temperature_bin']=pd.cut(test.Temperature,[-np.inf,-50,-10,10,25,30,50,80,200,np.inf],include_lowest=True).astype(str)
    rows=[]
    for column in ['Salt_count','Component_count','composition_seen','temperature_bin']:
        for value,part in test.groupby(column,dropna=False):
            rows.append({'stratum':column,'value':str(value),**scores(part),
                         **{'physical_'+key: value for key, value in scores(part, True).items()
                            if key in ('rmse', 'mae', 'r2')}})
            if column == 'temperature_bin':
                rows[-1].update(T_min=part.Temperature.min(), T_max=part.Temperature.max(), T_std=part.Temperature.std())
    pd.DataFrame(rows).to_csv(out/'test_strata.csv',index=False)
    from core.plotting import plot_results, PlotLossesLR
    results={task:{'truths':part.Truth.to_numpy(),'predictions':part.Prediction.to_numpy()}
             for task,part in frame.groupby('Task')}
    metrics={task:spec['native'] for task,spec in by_split.items()}
    unit=frame.Physical_unit.iloc[0] if job['target']=='IC' else frame.Target_unit.iloc[0]
    plot_results(results['Training'],results['Validation'],results['Test'],metrics,str(out),f"{job['target']} [{unit}]")
    history=pd.read_csv(out/'loss_history.csv') if (out/'loss_history.csv').exists() else pd.DataFrame()
    if 'train_loss' in history:
        fig,_=PlotLossesLR.plot_training_loss_and_lr(history.train_loss.tolist(),history.val_loss.tolist(),history.learning_rate.tolist())
        fig.savefig(out/'loss_history.png',dpi=180)
        import matplotlib.pyplot as plt
        plt.close(fig)
    elif job['model'] in ('molsets', 'xgboost'):
        import matplotlib.pyplot as plt
        history = pd.read_csv(out/'loss_history.csv')
        columns = ('Training_RMSE', 'Validation_RMSE') if job['model'] == 'xgboost' else ('Training', 'Validation')
        if history.empty or not np.isfinite(history[list(columns)].to_numpy()).all():
            raise ValueError('Missing or nonfinite training/validation history')
        fig, axis = plt.subplots(figsize=(10, 6))
        axis.plot(history[columns[0]], color='orange', lw=2, label='Training')
        axis.plot(history[columns[1]], color='blue', lw=2, label='Validation')
        axis.set(xlabel='Epoch', title='XGBoost' if job['model'] == 'xgboost' else 'MolSets-SAGE')
        axis.set_ylabel('RMSE' if job['model'] == 'xgboost' else 'MSE')
        if job['model'] == 'xgboost':
            axis.set_xlabel('Boosting round')
        axis.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out/'loss_history.png', dpi=180)
        plt.close(fig)
    return by_split


def aggregate():
    plan=json.loads((ROOT/'config/study.json').read_text())
    ledger_path=OUTPUT/'job_ledger.json'
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {'jobs':{}}
    records=[];strata=[];folds=[]
    for job in plan['jobs']:
        out=run_dir(job)
        row={k:job.get(k) for k in ['id','model','dataset','protocol','target','seed','run','fold','regime']}
        state=ledger['jobs'].get(job['id'],{})
        row['status']=state.get('status',job.get('status','queued'))
        row['reason']=state.get('reason',job.get('reason',''))
        if row['status'] == 'completed' and 'completion' not in read_run(out):
            row.update(status='failed', reason='Missing verified completion record')
        complete='completion' in read_run(out)
        if complete:
            from tasks.training.train_holdout import expected_identity, load_protocol
            proto,_=load_protocol(job['protocol'])
            try:
                verify_completion(out,expected_identity(job,proto))
            except (OSError,ValueError,RuntimeError,KeyError) as error:
                row.update(status='failed',reason=f'Stale or invalid run: {error}');complete=False
        if complete:
            result=verify_run(out,ROOT/proto['split_dir'],job,render=False)
            row.update(result['Test']['native']);row['status']='completed'
            row.update({'physical_'+k:v for k,v in result['Test']['physical'].items() if k in ('rmse','mae','r2')})
            for part, prefix in (('Training', 'train'), ('Validation', 'validation')):
                row.update({f'{prefix}_{key}': value for key, value in result[part]['native'].items()})
                row.update({f'{prefix}_physical_{key}': value for key, value in result[part]['physical'].items()
                            if key in ('rmse', 'mae', 'r2')})
            definitions=read_run(out)['metrics']
            row.update({k:definitions[k] for k in ('target_unit','physical_unit','transformation')})
            by_stratum=pd.read_csv(out/'test_strata.csv').assign(job_id=job['id'],model=job['model'],dataset=job['dataset'],regime=job['regime'],seed=job['seed'],run=job['run'],fold=job['fold'],target=job['target'],**{k:definitions[k] for k in ('target_unit','physical_unit','transformation')})
            strata.extend(by_stratum.to_dict('records'))
        records.append(row)
        if job.get('fold') and job['dataset'] == 'calisol':folds.append(row.copy())
    tables=OUTPUT/'tables';tables.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(tables/'all_experiments.csv',index=False)
    pd.DataFrame(folds).to_csv(tables/'calisol_folds.csv',index=False)
    pd.DataFrame(strata).to_csv(tables/'test_strata.csv',index=False)
    df=pd.DataFrame(records)
    stratum_scores(df, pd.DataFrame(strata)).to_csv(tables/'stratum_summary.csv', index=False)
    repeats = repeat_scores(df)
    repeats.to_csv(tables/'repeat_metrics.csv', index=False)
    calisol = repeats.loc[repeats.dataset == 'calisol']
    comparison_scores(df.loc[(df.dataset == 'calisol') & (df.model == 'perform')
                             & df.regime.isin(['benchmark', 'ic_lic'])]).to_csv(
        tables/'ic_lic_native_and_physical.csv', index=False)
    strict = df.loc[df.regime == 'strict_lmo']
    if len(strict):
        comparison_scores(strict).to_csv(tables/'strict_lmo_summary.csv', index=False)
    full = calisol.loc[(calisol.model == 'perform') & (calisol.target == 'LIC') & (calisol.regime == 'benchmark')]
    pd.concat([full, calisol.loc[calisol.regime == 'component']]).to_csv(tables/'component_ablations.csv', index=False)
    pd.concat([full, calisol.loc[calisol.regime == 'mechanism']]).to_csv(tables/'mechanism_ablations.csv', index=False)
    runtime_rows=[]
    for job in plan['jobs']:
        out=run_dir(job)
        run=read_run(out)
        if 'completion' not in run:continue
        checkpoint,measured=run['completion'],run.get('training',{})
        runtime_rows.append({'id':job['id'],'model':job['model'],'dataset':job['dataset'],
                             'elapsed_seconds_including_features_and_evaluation':checkpoint['elapsed_seconds'],
                             'training_hours':measured.get('train_hours'),'epochs_run':measured.get('epochs_run'),
                             'training_peak_gpu_allocated_bytes': measured.get('training_peak_gpu_allocated_bytes'),
                             'configuration':json.dumps(run['identity']['configuration'],sort_keys=True)})
    pd.DataFrame(runtime_rows).to_csv(tables/'configuration_and_runtime.csv',index=False)
    coverage=OUTPUT/'compatibility/compatibility_matrix.csv'
    if coverage.exists():
        pd.read_csv(coverage).to_csv(tables/'coverage_and_capability.csv',index=False)
    benchmark=df.loc[df.regime.isin(['holdout','main','full','common_subset','benchmark'])].copy()
    summary=comparison_scores(benchmark)
    if len(summary):
        subset=summary.dataset.str.endswith('_common')
        summary.loc[~subset].to_csv(tables/'main_benchmark.csv',index=False)
        summary.loc[summary.dataset.str.endswith('_common')].to_csv(tables/'common_subset.csv',index=False)
        full=summary.loc[~subset].copy()
        full['reported_rmse']=full.apply(lambda r: str(r.rmse) if r.status == 'complete' else 'unavailable' if r.status == 'blocked' else f'pending ({r.completed_repeats}/{r.planned_repeats} repeats)',axis=1)
        full.pivot(index='model',columns='dataset',values='reported_rmse').to_csv(tables/'main_benchmark_wide.csv')
    atomic_json(tables/'completion.json',{'planned':len(records),'statuses':df.status.value_counts().to_dict(),
                                        'complete':bool(len(df) and (df.status=='completed').all())})


def prepared_inputs_summary():
    from data.conversion_scripts.prepare_inputs import MODELS
    plan = json.loads((ROOT / 'config/study.json').read_text())
    ready = {}
    for job in plan['jobs']:
        if job['model'] in MODELS:
            protocol = plan['protocols'][job['protocol']]
            present = (ROOT / protocol['split_dir'] / 'formats' / job['model']).is_dir()
            ready.setdefault((protocol['dataset'], job['model']), set()).add((job['protocol'], present))
    path = OUTPUT / 'compatibility/compatibility_matrix.csv'
    matrix = pd.read_csv(path).to_dict('records')
    for entry in matrix:
        protocols = ready.get((entry['dataset'], entry['model']))
        if protocols:
            missing = sum(not present for _, present in protocols)
            entry.update(prepared_inputs_ready=missing == 0, prepared_inputs_protocols=len(protocols),
                         prepared_inputs_reasons=f'inputs missing for {missing} splits' if missing else '')
    pd.DataFrame(matrix).to_csv(path, index=False)
    print(f"Input compatibility: {len(ready)} dataset/model pairs checked.")


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepared-inputs', action='store_true')
    arguments=parser.parse_args()
    if arguments.prepared_inputs:
        prepared_inputs_summary()
    else:
        aggregate()
