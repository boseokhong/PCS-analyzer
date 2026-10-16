"""Multi-Ln analysis over detached data, with explicit shift-type boundaries."""
from hashlib import sha256
import csv
import json
import math
import numpy as np

MODEL = 'multi_ln_analysis_v1'
COEFFICIENT_VERSION = 'user_legacy_reilley_rhf_v1.3'
CD = dict(Ce=-6.3, Pr=-11., Nd=-4.2, Sm=-.7, Eu=4., Tb=-86., Dy=-100., Ho=-39., Er=33., Tm=53., Yb=22.)
RHF = dict(Ce=(-1.0714,3.2588),Pr=(-3.2,8.1348),Nd=(-4.9091,15.2287),
           Tb=(31.5,11.7506),Dy=(28.3333,7.7286),Ho=(22.5,4.7452),
           Er=(15.3,2.7447),Tm=(8.1667,1.4403),Yb=(2.5714,.57744))


def finite(value):
    value=float(value)
    if not math.isfinite(value):raise ValueError('Values must be finite')
    return value


def coefficients(temperature):
    temperature=finite(temperature)
    if temperature<=0:raise ValueError('Temperature must be positive (K)')
    sz={ln:a+b*1e-4*temperature for ln,(a,b) in RHF.items()}
    sz.update(Sm=.224,Eu=7.569)
    return dict(version=COEFFICIENT_VERSION,temperature=temperature,cd=CD.copy(),sz=sz,
                notes=['Legacy user-supplied relative coefficient convention; numerical table not independently certified.',
                       'Sm/Eu use fixed 300 K RHF values at every selected temperature.',
                       'C_D is a relative temperature-independent coefficient in this legacy convention.'])


def default_options():
    return dict(mode='tensor',temperature=300.,method='direct',use_errors=False,
                reference_ln='Nd',nucleus_i='',nucleus_j='',nucleus_c='')


def normalize_options(value=None):
    result=default_options();result.update(value or {})
    if result['mode'] not in ('tensor','reilley','scaling','nuclei','geometry','parameters'):raise ValueError('Unknown analysis')
    if result['method'] not in ('direct','sz','cd'):raise ValueError('Unknown normalization')
    result['temperature']=finite(result['temperature'])
    coefficients(result['temperature'])
    if type(result['use_errors']) is not bool:raise ValueError('Weighting must be a boolean')
    if result['reference_ln'] not in CD:raise ValueError('Unknown reference Ln')
    return result


def normalize_datasets(value=None):
    values=json.loads(json.dumps(value or [],allow_nan=False));result=[];seen=set()
    if not isinstance(values,list):raise ValueError('Datasets must be records')
    for index,d in enumerate(values):
        if not isinstance(d,dict):raise ValueError('Dataset must be a record')
        identifier=str(d.get('id',f'legacy:{index}'))
        if identifier in seen:raise ValueError('Duplicate dataset identity')
        seen.add(identifier)
        kind=d.get('kind','pcs');ln=str(d.get('ln','Nd'))
        if type(d.get('included',True)) is not bool:raise ValueError('Dataset inclusion must be a boolean')
        if kind not in ('pcs','paramagnetic'):raise ValueError('Explicit PCS or paramagnetic shift type required')
        if ln not in CD:raise ValueError(f'Unsupported Ln: {ln}')
        records=d.get('rows')
        if records is None:records=[dict(site=f'Ref {r}',ref_id=int(r),value=y) for r,y in (d.get('obs') or {}).items()]
        rows=[];sites=set()
        for row in records:
            site=str(row.get('site','')).strip()
            if not site or site in sites:raise ValueError('Each dataset needs unique nonempty nucleus/site labels')
            sites.add(site);ref=row.get('ref_id');sigma=row.get('error');g=row.get('geometry')
            if ref is not None and type(ref) is not int:raise ValueError('Ref must be an explicit integer or absent')
            if sigma is not None:
                sigma=finite(sigma)
                if sigma<=0:raise ValueError('Shift uncertainty must be positive')
            rows.append(dict(site=site,ref_id=ref,value=finite(row['value']),error=sigma,
                             geometry=finite(g) if g is not None else None))
        result.append(dict(id=identifier,name=str(d.get('name',ln)),ln=ln,kind=kind,
                           included=bool(d.get('included',True)),rows=rows,provenance=d.get('provenance')))
    return result


def parse_matrix_csv(stream):
    """Historical wide Nucleus,Ln,G_common,Ln_G_i data means total para shifts."""
    reader=csv.DictReader(stream);headers=reader.fieldnames or []
    site_key=next((h for h in ('Nucleus','nucleus','site') if h in headers),None)
    if not site_key:raise ValueError('Matrix CSV requires a Nucleus column')
    lns=[ln for ln in CD if ln in headers]
    if not lns:raise ValueError('No supported Ln columns')
    datasets=[dict(id=f'matrix:{ln}',name=f'{ln} paramagnetic',ln=ln,kind='paramagnetic',rows=[]) for ln in lns]
    seen=set()
    for line,row in enumerate(reader,2):
        site=(row.get(site_key) or '').strip()
        if not site or site in seen:raise ValueError(f'Duplicate/missing nucleus at CSV line {line}')
        seen.add(site)
        for ds in datasets:
            ln=ds['ln'];text=(row.get(ln) or '').strip()
            if not text:continue
            g=next((row[k] for k in (f'{ln}_G_i','G_common','G_i') if (row.get(k) or '').strip()),None)
            ref=(row.get('ref_id') or '').strip();err=(row.get(f'{ln}_err') or row.get('error') or '').strip()
            ds['rows'].append(dict(site=site,ref_id=int(ref) if ref else None,value=finite(text),
                                   error=finite(err) if err else None,geometry=finite(g) if g is not None else None))
    return normalize_datasets(datasets)


def parse_dataset_csv(stream,ln,kind):
    reader=csv.DictReader(stream);records=[]
    for line,row in enumerate(reader,2):
        ref=row.get('ref_id',row.get('ref',row.get('id','')))
        key='pcs_exp' if kind=='pcs' else 'delta_para'
        value=next((row[k] for k in (key,'delta_exp','d_exp','ppm') if (row.get(k) or '').strip()),None)
        if value is None:raise ValueError(f'Missing {key} at line {line}')
        site=(row.get('Nucleus') or row.get('site') or (f'Ref {int(ref)}' if ref else '')).strip()
        sigma=row.get('pcs_err',row.get('error',''));g=row.get('G_common',row.get('geometry',''))
        records.append(dict(site=site,ref_id=int(ref) if ref else None,value=finite(value),
                            error=finite(sigma) if sigma else None,geometry=finite(g) if g else None))
    return normalize_datasets([dict(ln=ln,kind=kind,rows=records)])[0]


def fingerprint(snapshot):
    return sha256(json.dumps(snapshot,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def solve(x,y,errors=None):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if x.ndim!=2 or y.shape!=(len(x),) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):raise ValueError('Invalid regression arrays')
    error=np.ones(len(y)) if errors is None else np.asarray(errors,float)
    if error.shape!=y.shape or np.any(~np.isfinite(error)) or np.any(error<=0):raise ValueError('Every fitted point needs a positive supplied error')
    a=x/error[:,None];b=y/error
    scales=np.linalg.norm(a,axis=0)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)) or not np.all(np.isfinite(scales)):raise ValueError('Regression exceeds supported numerical range')
    if np.any(scales==0):raise ValueError('Rank deficient design')
    scaled=a/scales
    beta,_,rank,singular=np.linalg.lstsq(scaled,b,rcond=None)
    if rank<x.shape[1]:raise ValueError('Rank deficient design: coefficients cannot be separated')
    beta/=scales;pred=x@beta;residual=pred-y;dof=len(y)-x.shape[1]
    rss=float(np.sum((residual/error)**2));cov=None
    if dof>0:
        _,s,v=np.linalg.svd(scaled,full_matrices=False)
        inv=(v.T/(s*s))@v
        cov=inv/scales[:,None]/scales[None,:]*(rss/dof if errors is None else 1.)
    total=float(np.sum((y-y.mean())**2));r2=1-float(np.sum(residual**2))/total if dof>0 and total>1e-20 else None
    if not np.all(np.isfinite(beta)) or not np.all(np.isfinite(pred)) or not math.isfinite(rss) or (cov is not None and not np.all(np.isfinite(cov))):raise ValueError('Regression result exceeds supported numerical range')
    return dict(coefficients=beta.tolist(),predicted=pred.tolist(),residual=residual.tolist(),
                covariance=cov.tolist() if cov is not None else None,rank=int(rank),dof=dof,
                condition=float(singular[0]/singular[-1]),rmsd=float(np.sqrt(np.mean(residual**2))),
                weighted_sum_squares=rss,r_squared=r2,
                note='Two-point estimate; no residual degrees of freedom' if dof==0 else 'Supplied errors treated as known; coefficient table treated as exact' if errors is not None else 'Unweighted; covariance uses estimated residual variance')


def matrix(snapshot):
    datasets=[d for d in normalize_datasets(snapshot['datasets']) if d['included'] and d['kind']=='paramagnetic']
    labels=[d['ln'] for d in datasets]
    if len(labels)!=len(set(labels)):raise ValueError('Select one paramagnetic dataset per Ln for series analysis')
    records={}
    for ds in datasets:
        for row in ds['rows']:records.setdefault(row['site'],[]).append(dict(row,ln=ds['ln'],dataset_id=ds['id']))
    return datasets,records


def reilley(snapshot):
    options=normalize_options(snapshot['options']);table=coefficients(options['temperature']);_,sites=matrix(snapshot)
    outputs=[]
    for site,rows in sites.items():
        try:
            if len(rows)<2:raise ValueError('At least two distinct Ln measurements required')
            cd=np.array([table['cd'][r['ln']] for r in rows]);sz=np.array([table['sz'][r['ln']] for r in rows]);y=np.array([r['value'] for r in rows]);x=np.column_stack([cd,sz])
            errors=np.array([r['error'] if r['error'] is not None else np.nan for r in rows]) if options['use_errors'] else None
            divisor=sz if options['method']=='sz' else cd if options['method']=='cd' else np.ones(len(rows))
            if np.any(np.abs(divisor)<1e-12):raise ValueError('Normalization denominator is zero')
            fit=solve(x/divisor[:,None],y/divisor,errors/np.abs(divisor) if errors is not None else None)
            g,f=fit['coefficients'];original_pred=x@np.array([g,f]);cov=fit['covariance']
            points=[]
            for i,r in enumerate(rows):
                variance=cov[0][0]*cd[i]**2 if cov is not None else None
                points.append(dict(r,pcs_model=float(g*cd[i]),fcs_model=float(f*sz[i]),predicted=float(original_pred[i]),
                    residual=float(original_pred[i]-y[i]),pcs_measured_minus_contact=float(y[i]-f*sz[i]),
                    pcs_model_error=float(np.sqrt(max(0,variance))) if variance is not None else None))
            outputs.append(dict(site=site,status='Fitted',G_eff=g,F_eff=f,fit=fit,points=points,
                                rmsd_original=float(np.sqrt(np.mean((original_pred-y)**2)))))
        except ValueError as exc:outputs.append(dict(site=site,status='Excluded',reason=str(exc),points=rows))
    if not any(o['status']=='Fitted' for o in outputs):raise ValueError('No identifiable Reilley nucleus: review matching, errors and rank')
    return dict(table=table,sites=outputs,assumptions=['Isostructural series and common coefficient interpretation assumed; linearity is diagnostic, not proof.',
                      'Model PCS and measured-minus-contact PCS differ by the residual.', 'Separated values share coefficient uncertainty and are correlated.'])


def series_diagnostics(snapshot):
    options=normalize_options(snapshot['options']);table=coefficients(options['temperature']);datasets,sites=matrix(snapshot)
    ref=options['reference_ln'];mode=options['mode'];outputs=[]
    if mode=='scaling':
        for ds in datasets:
            points=[]
            for site,rows in sites.items():
                lookup={r['ln']:r for r in rows}
                if ref in lookup and ds['ln'] in lookup:points.append(dict(site=site,x=lookup[ref]['value'],y=lookup[ds['ln']]['value']))
            try:fit=solve([[p['x'],1.] for p in points],[p['y'] for p in points]);outputs.append(dict(ln=ds['ln'],status='Fitted',fit=fit,points=points))
            except ValueError as exc:outputs.append(dict(ln=ds['ln'],status='Excluded',reason=str(exc),points=points))
        return dict(reference_ln=ref,series=outputs,assumptions=['Total-shift slopes may include contact contributions; they are not certified Bleaney coefficients.'])
    i,j,c=(options[k] for k in ('nucleus_i','nucleus_j','nucleus_c'))
    if i not in sites or j not in sites or i==j:raise ValueError('Choose two distinct matching nuclei')
    a={r['ln']:r for r in sites[i]};b={r['ln']:r for r in sites[j]};anchor={r['ln']:r for r in sites.get(c,[])}
    points=[];excluded=[]
    for ln in [d['ln'] for d in datasets if d['ln'] in a and d['ln'] in b]:
        try:
            g1,g2=a[ln]['geometry'],b[ln]['geometry'];spin=table['sz'][ln]
            if g1 is None or g2 is None or abs(g2)<1e-12 or abs(spin)<1e-12:raise ValueError('Missing/zero structural G or spin denominator')
            ratio=g1/g2;y=a[ln]['value']/spin;x=b[ln]['value']/spin
            p=dict(ln=ln,x=x,y=y,geometry_ratio=ratio,Q=y-ratio*x)
            if mode=='geometry':
                if abs(b[ln]['value'])<1e-12:raise ValueError('Zero shift denominator')
                p.update(shift_ratio=a[ln]['value']/b[ln]['value'])
            if c:
                if c in (i,j) or ln not in anchor or anchor[ln]['geometry'] is None or abs(anchor[ln]['geometry'])<1e-12:raise ValueError('Third nucleus needs distinct matching data and nonzero G')
                p.update(Q_A=y-g1/anchor[ln]['geometry']*anchor[ln]['value']/spin,
                         Q_B=x-g2/anchor[ln]['geometry']*anchor[ln]['value']/spin)
            points.append(p)
        except ValueError as exc:excluded.append(dict(ln=ln,reason=str(exc)))
    if not points:raise ValueError('No diagnostic points; supply explicit structural geometry factors')
    fit=None
    try:fit=solve([[p['x'],1] for p in points],[p['y'] for p in points])
    except ValueError:pass  # descriptive points remain valid without identifiable regression
    cfp=None
    if mode=='geometry':
        separated=reilley(snapshot);cfp_points=[];cfp_excluded=[]
        for site in separated['sites']:
            if site['status']!='Fitted':cfp_excluded.append(dict(site=site['site'],reason=site['reason']));continue
            values=[r['geometry'] for r in site['points']]
            if any(g is None or abs(g)<1e-9 for g in values) or not np.allclose(values,values[0],rtol=1e-8,atol=1e-12):
                cfp_excluded.append(dict(site=site['site'],reason='CFP comparison requires a common nonzero structural G across Ln'));continue
            cfp_points.append(dict(site=site['site'],G_struct=values[0],G_eff=site['G_eff'],ratio=site['G_eff']/values[0]))
        if cfp_points:
            origin=solve([[p['G_struct']] for p in cfp_points],[p['G_eff'] for p in cfp_points])
            try:free=solve([[p['G_struct'],1.] for p in cfp_points],[p['G_eff'] for p in cfp_points])
            except ValueError:free=None
            cfp=dict(points=cfp_points,origin=origin,free=free,excluded=cfp_excluded,
                     units='Relative coefficient under legacy C_D convention; no absolute energy unit')
    return dict(nucleus_i=i,nucleus_j=j,nucleus_c=c,points=points,excluded=excluded,fit=fit,cfp=cfp,
                assumptions=['Structural ratios are explicit inputs; absolute PCS is not determined by this diagnostic.',
                             'Relative geometry checks do not prove isostructurality or provide absolute CFP energy units.'])


def tensor_fit(snapshot):
    options=normalize_options(snapshot['options']);factors=snapshot.get('factors',{});outputs=[]
    datasets=[d for d in normalize_datasets(snapshot['datasets']) if d['included'] and d['kind']=='pcs']
    if len(datasets)<2:raise ValueError('Tensor fitting requires at least two included PCS datasets')
    for ds in datasets:
        points=[];skipped=[]
        try:
            for row in ds['rows']:
                key=str(row['ref_id'])
                if key not in factors:skipped.append(dict(site=row['site'],reason='No explicit matching main Ref'));continue
                points.append(row)
            x=[factors[str(r['ref_id'])] for r in points]
            fit=solve(x,[r['value'] for r in points],[r['error'] for r in points] if options['use_errors'] else None)
            outputs.append(dict(id=ds['id'],name=ds['name'],ln=ds['ln'],status='Fitted',fit=fit,
                points=[dict(row,predicted=fit['predicted'][k],residual=fit['residual'][k]) for k,row in enumerate(points)],skipped=skipped))
        except (ValueError,TypeError) as exc:outputs.append(dict(id=ds['id'],name=ds['name'],ln=ds['ln'],status='Excluded',reason=str(exc),points=points,skipped=skipped))
    if not any(o['status']=='Fitted' for o in outputs):raise ValueError('No identifiable tensor dataset: review rank, matching and errors')
    return dict(datasets=outputs,frame=snapshot.get('frame'),assumptions=['Fixed supplied geometry and frame; independent two-scalar tensors.'])


def execute_analysis(snapshot):
    snapshot=json.loads(json.dumps(snapshot,allow_nan=False));options=normalize_options(snapshot['options'])
    if options['mode']=='tensor':output=tensor_fit(snapshot)
    elif options['mode']=='reilley':output=reilley(snapshot)
    elif options['mode']=='parameters':output=coefficients(options['temperature'])
    else:output=series_diagnostics(snapshot)
    return dict(model=MODEL,schema_version=1,mode=options['mode'],input_snapshot=snapshot,
                fingerprint=fingerprint(snapshot),output=output)


def validate_result(value):
    if value is None:return None
    if not isinstance(value,dict) or value.get('model')!=MODEL or value.get('schema_version')!=1:raise ValueError('Unknown Multi-Ln result schema')
    expected=execute_analysis(value['input_snapshot'])
    if value['fingerprint']!=expected['fingerprint'] or value['mode']!=expected['mode']:raise ValueError('Invalid Multi-Ln result identity')
    def equal(a,b):
        if isinstance(b,dict):return isinstance(a,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in b)
        if isinstance(b,list):return isinstance(a,list) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        if isinstance(b,(float,int)) and not isinstance(b,bool):return isinstance(a,(float,int)) and math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-10)
        return a==b
    if not equal(value['output'],expected['output']):raise ValueError('Recorded Multi-Ln predictions/metrics do not match their inputs')
    return value


def project_payload(datasets,options,results):
    if results is not None and not isinstance(results,dict):raise ValueError('Results must be keyed by analysis mode')
    return dict(schema_version=1,datasets=normalize_datasets(datasets),options=normalize_options(options),
                results={k:validate_result(v) for k,v in (results or {}).items() if v is not None})


def parse_project_payload(value):
    if value is None:return [],default_options(),{}
    if not isinstance(value,dict) or value.get('schema_version')!=1:raise ValueError('Unknown Multi-Ln project schema')
    payload=project_payload(value['datasets'],value['options'],value['results'])
    if any(k!=v['mode'] for k,v in payload['results'].items()):raise ValueError('Analysis result mode mismatch')
    return payload['datasets'],payload['options'],payload['results']
