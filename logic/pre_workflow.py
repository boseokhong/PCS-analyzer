"""Detached simplified proton Gamma2 fitting; no UI or application state."""
from dataclasses import dataclass, asdict, replace
from hashlib import sha256
import csv
import json

import numpy as np
from scipy.optimize import brentq
from logic.advanced_physics import pre_gamma2, GAMMA_H

MODEL='pre_proton_gamma2_effective_tau_v1'


def default_options():
    return dict(B0=14.1,mu_eff=3.68,preset='Nd',use_errors=False,tau_min_ns=.001,tau_max_ns=1000.)


def normalize_options(value=None):
    options=default_options()
    options.update(value or {})
    for key in ('B0','mu_eff','tau_min_ns','tau_max_ns'):
        options[key]=float(options[key])
        if not np.isfinite(options[key]) or options[key]<=0:
            raise ValueError('Field, moment and tau bounds must be finite and positive')
    if options['tau_min_ns']>=options['tau_max_ns']:
        raise ValueError('Minimum tau must be below maximum tau')
    if type(options['use_errors']) is not bool:
        raise ValueError('Uncertainty weighting must be a boolean')
    options['preset']=str(options['preset'])
    return options


def normalize_rows(value):
    if isinstance(value,dict):
        value=[dict(ref_id=int(r),gamma2=float(rate),uncertainty=None,nucleus='1H',
                    interpretation='legacy_increment',source_row=i+1) for i,(r,rate) in enumerate(value.items())]
    rows=json.loads(json.dumps(value or [],allow_nan=False))
    if not isinstance(rows,list) or any(not isinstance(row,dict) for row in rows):
        raise ValueError('PRE observations must be a list of records')
    seen=set()
    for row in rows:
        if type(row.get('ref_id')) is not int or row['ref_id'] in seen:
            raise ValueError('PRE needs one unique integer Ref per observation')
        seen.add(row['ref_id'])
        row['gamma2']=float(row['gamma2'])
        if not np.isfinite(row['gamma2']):
            raise ValueError('PRE increments must be finite')
        uncertainty=row.get('uncertainty')
        if uncertainty is not None:
            uncertainty=float(uncertainty)
            if not np.isfinite(uncertainty) or uncertainty<=0:
                raise ValueError('PRE uncertainties must be positive and finite')
        row['uncertainty']=uncertainty
        row['nucleus']=str(row.get('nucleus','1H')).upper()
        if row['nucleus'] not in ('H','1H'):
            raise ValueError('This simplified PRE model supports proton observations only')
        row['interpretation']=str(row.get('interpretation','gamma2_increment'))
        row['source_row']=int(row.get('source_row',len(seen)))
    return rows


def parse_pre_csv(stream):
    reader=csv.DictReader(stream)
    reader.fieldnames=[c.strip().lower() for c in reader.fieldnames or []]
    columns=set(reader.fieldnames)
    ref=next((c for c in ('ref_id','ref','id') if c in columns),None)
    rate=next((c for c in ('gamma2','g2','r2_para','r2') if c in columns),None)
    paired={'r2_para','r2_dia'}<=columns
    if ref is None or (rate is None and not paired):
        raise ValueError('PRE CSV requires Ref and gamma2 increment, or paired r2_para/r2_dia')
    rows=[]
    for number,record in enumerate(reader,2):
        try:
            if paired:
                value=float(record['r2_para'])-float(record['r2_dia'])
                sigma=(float(np.hypot(float(record['r2_para_err']),float(record['r2_dia_err'])))
                       if record.get('r2_para_err') and record.get('r2_dia_err') else None)
                if sigma is not None and any(float(record[k])<=0 for k in ('r2_para_err','r2_dia_err')):
                    raise ValueError('Both paired-rate errors must be positive')
                interpretation='paired_independent_rates'
            else:
                value=float(record[rate])
                error=record.get('gamma2_err') or record.get('sigma')
                sigma=float(error) if error else None
                interpretation='legacy_increment_alias' if rate in ('r2','r2_para') else 'gamma2_increment'
            row=dict(ref_id=int(record[ref]),gamma2=value,uncertainty=sigma,
                     nucleus=record.get('nucleus') or '1H',interpretation=interpretation,source_row=number)
            rows.append(normalize_rows([row])[0])
        except (ValueError,TypeError,KeyError) as exc:
            raise ValueError(f'PRE CSV line {number}: {exc}') from exc
    if not rows:
        raise ValueError('PRE CSV contains no observations')
    return normalize_rows(rows)


@dataclass(frozen=True)
class PREObservation:
    ref_id:int
    gamma2:float
    uncertainty:float|None
    nucleus:str='1H'
    interpretation:str='gamma2_increment'
    source_row:int=1


@dataclass(frozen=True)
class PRESnapshot:
    ref_ids:tuple
    coordinates:tuple
    elements:tuple
    metal:tuple
    observations:tuple[PREObservation,...]
    B0:float=14.1
    mu_eff:float=3.68
    use_errors:bool=False
    tau_min_ns:float=.001
    tau_max_ns:float=1000.
    skipped:tuple=()


def snapshot_payload(snapshot):
    return json.loads(json.dumps(dict(model=MODEL,**asdict(snapshot)),allow_nan=False))


def snapshot_from_payload(value):
    if not isinstance(value,dict) or value.get('model')!=MODEL:
        raise ValueError('Unsupported PRE snapshot model')
    data=dict(value)
    data.pop('model')
    for key in ('ref_ids','elements','metal','skipped'):
        data[key]=tuple(data[key])
    data['coordinates']=tuple(tuple(p) for p in data['coordinates'])
    data['observations']=tuple(PREObservation(**row) for row in normalize_rows(data['observations']))
    return PRESnapshot(**data)


def fingerprint(snapshot):
    return sha256(json.dumps(snapshot_payload(snapshot),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def data_arrays(snapshot):
    normalize_options(dict(B0=snapshot.B0,mu_eff=snapshot.mu_eff,use_errors=snapshot.use_errors,
                           tau_min_ns=snapshot.tau_min_ns,tau_max_ns=snapshot.tau_max_ns))
    coords=np.asarray(snapshot.coordinates,dtype=float)
    metal=np.asarray(snapshot.metal,dtype=float)
    if coords.shape!=(len(snapshot.ref_ids),3) or len(snapshot.elements)!=len(snapshot.ref_ids) or metal.shape!=(3,):
        raise ValueError('PRE geometry requires explicit matching Ref/element/coordinate arrays')
    if len(set(snapshot.ref_ids))!=len(snapshot.ref_ids) or not np.all(np.isfinite(coords)) or not np.all(np.isfinite(metal)):
        raise ValueError('PRE geometry needs unique Refs and finite coordinates')
    if not snapshot.observations:
        raise ValueError('Load at least one matched proton PRE observation')
    normalize_rows([asdict(o) for o in snapshot.observations])
    indices={r:i for i,r in enumerate(snapshot.ref_ids)}
    distance=[]
    for o in snapshot.observations:
        if o.ref_id not in indices:
            raise ValueError('PRE observation has no raw structure Ref')
        index=indices[o.ref_id]
        if snapshot.elements[index].upper()!='H':
            raise ValueError('PRE observations must match raw hydrogen atoms')
        distance.append(np.linalg.norm(coords[index]-metal))
    distance=np.asarray(distance)
    if np.any(distance<=0):
        raise ValueError('A PRE proton cannot coincide with the metal centre')
    observed=np.array([o.gamma2 for o in snapshot.observations])
    if snapshot.use_errors:
        errors=np.array([o.uncertainty if o.uncertainty is not None else np.nan for o in snapshot.observations])
        if np.any(~np.isfinite(errors)) or np.any(errors<=0):
            raise ValueError('Every matched PRE observation needs a positive supplied uncertainty')
    else:
        errors=np.maximum(observed*.05,.1)  # retain the original assumed-error rule
    return distance,observed,errors


def response(snapshot,tau):
    return float(pre_gamma2(np.array([1.]),snapshot.mu_eff,tau,GAMMA_H*snapshot.B0)[0])


def execute_fit(snapshot):
    distance,observed,errors=data_arrays(snapshot)
    basis=distance**-6
    denominator=np.sum((basis/errors)**2)
    amplitude=float(np.sum(basis*observed/errors**2)/denominator)
    if not np.isfinite(denominator) or denominator<=0 or not np.isfinite(amplitude):
        raise ValueError('PRE distances, rates or uncertainties exceed the supported numerical range')
    lo,hi=snapshot.tau_min_ns*1e-9,snapshot.tau_max_ns*1e-9
    lower,upper=response(snapshot,lo),response(snapshot,hi)
    if not np.isfinite(lower) or not np.isfinite(upper) or not upper>lower:
        raise ValueError('PRE response is invalid for these moment/field/tau bounds')
    if amplitude<=lower:
        tau,boundary=lo,'lower'
    elif amplitude>=upper:
        tau,boundary=hi,'upper'
    else:
        tau=10**brentq(lambda log:response(snapshot,10**log)-amplitude,np.log10(lo),np.log10(hi),xtol=1e-12)
        boundary='interior'
    predicted=pre_gamma2(distance,snapshot.mu_eff,tau,GAMMA_H*snapshot.B0)
    residual=predicted-observed
    points=[dict(ref_id=o.ref_id,distance_angstrom=float(distance[i]),experimental=float(observed[i]),
                 predicted=float(predicted[i]),residual=float(residual[i]),effective_error=float(errors[i]),
                 uncertainty=o.uncertainty,interpretation=o.interpretation,source_row=o.source_row)
            for i,o in enumerate(snapshot.observations)]
    return dict(model=MODEL,schema_version=1,success=True,fingerprint=fingerprint(snapshot),
        input_snapshot=snapshot_payload(snapshot),tau_c=float(tau),tau_ns=float(tau*1e9),boundary=boundary,
        rmsd=float(np.sqrt(np.mean(residual**2))),weighted_sum_squares=float(np.sum((residual/errors)**2)),
        per_point=points,skipped=list(snapshot.skipped),nonpositive_increments=int(np.sum(observed<=0)))


def validate_result(value):
    if value is None:return None
    value=json.loads(json.dumps(value,allow_nan=False))
    if value.get('model')!=MODEL or value.get('schema_version')!=1 or value.get('success') is not True:
        raise ValueError('Unsupported PRE result')
    snapshot=snapshot_from_payload(value['input_snapshot'])
    if value['fingerprint']!=fingerprint(snapshot):raise ValueError('PRE input fingerprint mismatch')
    distance,observed,errors=data_arrays(snapshot)
    tau=float(value['tau_c'])
    lo,hi=snapshot.tau_min_ns*1e-9,snapshot.tau_max_ns*1e-9
    if not lo<=tau<=hi or not np.isclose(value['tau_ns'],tau*1e9,rtol=1e-10):raise ValueError('Invalid saved PRE tau')
    basis=distance**-6
    amplitude=np.sum(basis*observed/errors**2)/np.sum((basis/errors)**2)
    expected=np.clip(amplitude,response(snapshot,lo),response(snapshot,hi))
    if not np.isclose(response(snapshot,tau),expected,rtol=1e-9,atol=1e-10):raise ValueError('Saved PRE fit is inconsistent with its objective')
    boundary='lower' if amplitude<=response(snapshot,lo) else 'upper' if amplitude>=response(snapshot,hi) else 'interior'
    if value['boundary']!=boundary:raise ValueError('Saved PRE boundary status is inconsistent')
    predicted=pre_gamma2(distance,snapshot.mu_eff,tau,GAMMA_H*snapshot.B0)
    residual=predicted-observed
    if len(value['per_point'])!=len(observed):raise ValueError('Saved PRE point count mismatch')
    for point,o,d,y,p,e in zip(value['per_point'],snapshot.observations,distance,observed,predicted,errors):
        if (point['ref_id'],point['source_row'],point['interpretation'],point['uncertainty'])!=(o.ref_id,o.source_row,o.interpretation,o.uncertainty):
            raise ValueError('Saved PRE observation identity mismatch')
        for key,expected_value in (('distance_angstrom',d),('experimental',y),('predicted',p),('residual',p-y),('effective_error',e)):
            if not np.isclose(point[key],expected_value,rtol=1e-9,atol=1e-10):raise ValueError('Saved PRE prediction mismatch')
    for key,metric in (('rmsd',np.sqrt(np.mean(residual**2))),('weighted_sum_squares',np.sum((residual/errors)**2))):
        if not np.isclose(value[key],metric,rtol=1e-8,atol=1e-10):raise ValueError('Saved PRE metric mismatch')
    if value['skipped']!=list(snapshot.skipped) or value['nonpositive_increments']!=int(np.sum(observed<=0)):
        raise ValueError('Saved PRE matching diagnostics mismatch')
    return value


def project_payload(rows,options,result):
    return dict(format='pre_project',schema_version=1,rows=normalize_rows(rows),
                options=normalize_options(options),last_result=validate_result(result))


def parse_project_payload(value):
    if value is None:return [],default_options(),None
    if not isinstance(value,dict) or value.get('format')!='pre_project' or value.get('schema_version')!=1:
        raise ValueError('Unsupported PRE project schema')
    return normalize_rows(value['rows']),normalize_options(value['options']),validate_result(value['last_result'])


def result_status(result,current):
    if not result or result.get('model')!=MODEL:return 'Unknown'
    if current is None:return 'Stale'
    saved=snapshot_from_payload(result['input_snapshot'])
    changes={}
    for key in ('coordinates','metal'):
        a,b=np.asarray(getattr(current,key)),np.asarray(getattr(saved,key))
        if a.shape==b.shape and np.allclose(a,b,rtol=0,atol=5.1e-9):changes[key]=getattr(saved,key)
    return 'Current' if fingerprint(replace(current,**changes))==result['fingerprint'] else 'Stale'
