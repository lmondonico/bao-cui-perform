"""Run the planned folds in resumable GPU queues."""
import argparse
import csv
import fcntl
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from core.study import OUTPUT, atomic_json, read_run, run_dir, verify_completion
from tasks.training.train_holdout import VARIANTS, expected_identity, load_protocol


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpus')
    parser.add_argument('--jobs-per-gpu',type=int,choices=(1,2),default=2)
    parser.add_argument('--cpu-cores',type=int,default=os.cpu_count())
    parser.add_argument('--jobs-file',type=Path)
    parser.add_argument('--retry-failed',action='store_true')
    parser.add_argument('--smoke-only',action='store_true')
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    from core.study import require_training_enabled
    require_training_enabled()
    args.gpus=args.gpus or ','.join(subprocess.check_output(['nvidia-smi','--query-gpu=index','--format=csv,noheader'],text=True).split())
    gpus=[int(value) for value in args.gpus.split(',')]
    if len(set(gpus))!=len(gpus) or args.cpu_cores < len(gpus)*args.jobs_per_gpu:
        raise ValueError('Expected distinct GPUs and at least one CPU core per job')
    slots=[(gpu,slot) for gpu in gpus for slot in range(args.jobs_per_gpu)]
    cpu_threads=args.cpu_cores//len(slots)
    workers=min(4,max(1,cpu_threads//3))
    tensor_threads=max(1,cpu_threads-workers)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    lock=(OUTPUT/'queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((ROOT/'config/study.json').read_text())
    with (OUTPUT/'compatibility/compatibility_matrix.csv').open() as stream:
        compatibility={(r['dataset'],r['model']):r for r in csv.DictReader(stream)}
    state_path=OUTPUT/'job_ledger.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {'created_utc':now(),'jobs':{}}
    for identifier,entry in state['jobs'].items():
        if entry.get('status')=='running' and entry.get('pid'):
            process=Path('/proc')/str(entry['pid'])
            if process.exists():
                command=(process/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
                if identifier in command:
                    raise RuntimeError(f'Existing job is still alive (PID {entry["pid"]}): {identifier}. No competing job started.')
    busy=subprocess.check_output(['nvidia-smi','-i',args.gpus,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if busy:
        raise RuntimeError(f'Compute processes are already using GPUs ({busy}); leave them undisturbed and resume later.')
    state['controller']={'pid':os.getpid(),'started_utc':now(),'gpus':args.gpus,
                         'jobs_per_gpu':args.jobs_per_gpu,'cpu_cores':args.cpu_cores,
                         'cpu_threads_per_job':cpu_threads,'perform_workers':workers}
    for job in plan['jobs']:
        model='perform' if job['model'] in VARIANTS or job['model']=='full' else job['model']
        check=compatibility.get((job['dataset'],model))
        entry=state['jobs'].setdefault(job['id'],{'status':'queued','output':str(run_dir(job))})
        if entry['status']=='running':
            entry.update(status='queued',reason='Previous controller interrupted; strict identity checkpoint resume required')
        if args.retry_failed and entry['status']=='failed':
            entry.update(status='queued',reason='Explicit retry of failed run')
        if entry['status']=='completed':
            try:
                proto,_=load_protocol(job['protocol'])
                verify_completion(run_dir(job),expected_identity(job,proto))
            except (OSError,ValueError,KeyError,RuntimeError) as exc:
                entry.update(status='blocked',reason='Completed artifact failed verification: '+str(exc))
            continue
        if not check or check['status']=='blocked':
            entry.update(status='blocked',reason=check['reasons'] if check else 'Missing compatibility audit')
        elif check.get('prepared_inputs_ready', '').lower() == 'false':
            entry.update(status='blocked',reason=check['prepared_inputs_reasons'])
        elif entry['status']=='blocked' and 'completion' not in read_run(run_dir(job)):
            entry.update(status='queued',reason='Compatibility checks now pass')
    mutex=Lock()
    smoke_locks={}
    diagnostics_path=OUTPUT/'compatibility/runtime_diagnostics.json'
    diagnostics=json.loads(diagnostics_path.read_text()) if diagnostics_path.exists() else {}
    order={k:i for i,k in enumerate(args.jobs_file.read_text().split())} if args.jobs_file else None
    pending=[j for j in plan['jobs'] if state['jobs'][j['id']]['status']=='queued' and (order is None or j['id'] in order)]
    if order:pending.sort(key=lambda j:order[j['id']])
    if args.limit:pending=pending[:args.limit]
    atomic_json(state_path,state)

    def save():
        state['updated_utc']=now();atomic_json(state_path,state)

    def execute(job,gpu,slot,smoke):
        folder=run_dir(job,smoke)
        folder.parent.mkdir(parents=True,exist_ok=True)
        logs=OUTPUT/'logs'/job['dataset'];logs.mkdir(parents=True,exist_ok=True)
        log=logs/(job['id']+('.smoke.log' if smoke else '.log'))
        env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
                 CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS=str(tensor_threads),MKL_NUM_THREADS=str(tensor_threads),
                 OPENBLAS_NUM_THREADS=str(tensor_threads),PERFORM_WORKERS=str(workers),
                 MPLBACKEND='Agg',MPLCONFIGDIR=str(OUTPUT/'cache/matplotlib'),
                 XDG_CACHE_HOME=str(OUTPUT/'cache'),HF_HOME=str(OUTPUT/'cache/huggingface'),
                 TORCH_HOME=str(OUTPUT/'cache/torch'),PERFORM_ROOT=str(ROOT),
                 UV_LINK_MODE='copy',TF_NUM_INTRAOP_THREADS=str(tensor_threads),TF_NUM_INTEROP_THREADS='2')
        env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
        cmd=[str(ROOT/'.venv-perform/bin/python'),'-s','-u',
             str(ROOT/'tasks/training/train_holdout.py'),'--job',job['id'],
             '--gpu-slot',str(slot),'--cpu-threads',str(cpu_threads)]
        if smoke:cmd.append('--smoke')
        with log.open('a') as stream:
            stream.write('\n'+now()+' '+repr(cmd)+'\n');stream.flush()
            process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            with mutex:
                state['jobs'][job['id']].update(pid=process.pid,log=str(log),gpu=gpu,slot=slot,phase='feasibility' if smoke else 'training')
                save()
            result=process.wait()
        error=''
        if result:
            error='\n'.join(log.read_text(errors='replace').splitlines()[-12:])[-3000:]
        return result,error

    def lane(gpu_slot):
        gpu,slot=gpu_slot
        while True:
            with mutex:
                if not pending:return
                job=pending.pop(0)
                entry=state['jobs'][job['id']]
                entry.update(status='running',started_utc=now(),reason='')
                save()
            started=time.time()
            try:
                proto,_=load_protocol(job['protocol'])
                if args.smoke_only:
                    diagnostic_identity=expected_identity(job,proto,smoke=True)
                    key=job['id']
                    with mutex:
                        smoke_lock=smoke_locks.setdefault(key,Lock())
                    with smoke_lock:
                        previous=diagnostics.get(key,{})
                        if previous.get('identity')!=diagnostic_identity or (args.retry_failed and previous.get('status')=='failed'):
                            previous={}
                        if previous.get('status')=='passed':
                            verify_completion(Path(previous['output']),diagnostic_identity)
                        if not previous:
                            code,error=execute(job,gpu,slot,True)
                            previous={'status':'passed' if code==0 else 'failed',
                                      'identity':diagnostic_identity,'job':job['id'],'error':error,'finished_utc':now(),
                                      'output':str(run_dir(job,True))}
                            with mutex:
                                diagnostics[key]=previous;atomic_json(diagnostics_path,diagnostics)
                        if previous['status']!='passed':
                            raise RuntimeError('Feasibility failed: '+previous['error'])
                    with mutex:
                        entry.update(status='queued',reason='Feasibility passed; full run not started by --smoke-only');save()
                    continue
                code,error=execute(job,gpu,slot,False)
                if code:raise RuntimeError(error)
                out=run_dir(job)
                if 'completion' not in read_run(out):raise RuntimeError('Process exited without verified completion marker')
                completed=verify_completion(out,expected_identity(job,proto))
                with mutex:
                    entry.update(status='completed',finished_utc=now(),elapsed_seconds=time.time()-started,
                                 checkpoints=[str(out/p) for p in completed['checkpoints']],reason='');save()
            except Exception as exc:
                with mutex:
                    entry.update(status='failed',finished_utc=now(),elapsed_seconds=time.time()-started,reason=str(exc));save()
    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        list(pool.map(lane,slots))
    state['controller']['finished_utc']=now();atomic_json(state_path,state)
    if not args.smoke_only and not args.jobs_file:
        subprocess.run([sys.executable,'-s',str(ROOT/'experiments/metrics.py')],cwd=ROOT,check=True)
        subprocess.run([sys.executable,'-s',str(ROOT/'experiments/analyses.py'),'all'],cwd=ROOT,check=True)


if __name__=='__main__':
    main()
