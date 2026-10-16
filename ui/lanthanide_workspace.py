"""Document-owned Multi-Ln workspace. All Tk access is on the main thread."""
from pathlib import Path
import csv
import json
import queue
import threading
import uuid
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from logic import lanthanide_workflow as workflow
from logic.advanced_physics import geom_factors, pcs_from_G
from logic.pcs_tensor_state import canonical_tensor_from_legacy, ApplicationPCSTensor, PCS_TENSOR_STATE_KEY
from logic.pcs_document import (active_document,active_document_view,mark_document_mutated,
    capture_worker_identity,worker_completion_matches,apply_worker_document_updates)
from ui.multiconformer_workspace import current_geometry,collect_observations
from ui.scrollable_workspace import VerticalScrolledWorkspace
from ui.widget_style import ui_native_colors,ui_font_name
from ui.style import get_app_fonts

PREFIX='analysis.advanced.multilanthanide.'
MODES=dict(tensor='Tensor fitting',reilley='Reilley separation',scaling='Ln scaling',
           nuclei='Nuclei diagnostics',geometry='Geometry / CFP',parameters='Parameter table')
METHODS=dict(direct='No normalization',sz='Normalize by S(T)',cd='Normalize by C_D')
SOURCES=dict(pcs_model='Model PCS',pcs_measured_minus_contact='Measured − contact')


def compile_snapshot(state):
    options=workflow.normalize_options(state.get('advfit_ml_options'))
    snapshot=dict(options=options,datasets=workflow.normalize_datasets(state.get('advfit_ml_datasets')),
                  coefficient_version=workflow.COEFFICIENT_VERSION)
    if options['mode']=='tensor':
        if state.get('fit_override'):raise ValueError('Clear geometry overrides before fixed-frame tensor fitting')
        geometry,_=current_geometry(state)
        if len(set(geometry.raw_ref_ids))!=len(geometry.raw_ref_ids):raise ValueError('Raw structure Refs must be unique')
        if geometry.torsion_enabled:raise ValueError('Multi-Ln fixed-frame tensor fitting does not support torsional averaging')
        tensor=state.get(PCS_TENSOR_STATE_KEY)
        if not isinstance(tensor,ApplicationPCSTensor):tensor=canonical_tensor_from_legacy(dict(state))
        frame=tensor.to_forward_tensor().frame
        raw=(np.asarray(geometry.raw_coordinates)-geometry.metal_position)@np.asarray(frame.rotation_matrix).T
        if not np.all(np.isfinite(raw)):
            raise ValueError('Fixed-frame structure needs finite coordinates')
        _,_,_,gax,grh=geom_factors(raw,np.zeros(3))
        basis=np.column_stack([pcs_from_G(gax,grh,1,0),pcs_from_G(gax,grh,0,1)])
        snapshot.update(factors={str(r):list(map(float,v)) for r,v,p in zip(geometry.raw_ref_ids,basis,raw) if np.linalg.norm(p)>0},
                        frame=dict(metal=list(geometry.metal_position),rotation=list(map(list,frame.rotation_matrix))),
                        geometry=dict(refs=list(geometry.raw_ref_ids),coordinates=list(map(list,geometry.raw_coordinates))))
    return json.loads(json.dumps(snapshot,allow_nan=False))


class LanthanideWorkspace:
    def __init__(self,state,notebook):
        self.state,self.root,self.alive=state,state['root'],True
        self.poll_id=self.map_id=None;self.input_height=320;self.report_height=180;self.collapsed=False;self.last_plot=None
        self.tab=ttk.Frame(notebook);notebook.add(self.tab,text='Multi-Lanthanide')
        for key,default in (('advfit_ml_datasets',[]),('advfit_ml_options',{}),('advfit_ml_results',{})):state.setdefault(key,default)
        self.mode_var=tk.StringVar(master=self.root)
        self.scroll=VerticalScrolledWorkspace(self.tab,padding=0);self.scroll.pack(fill='both',expand=True)
        ui_native_colors(self.scroll.canvas,background='surface')
        content=self.scroll.content
        self.header=tk.Frame(content,borderwidth=0);ui_native_colors(self.header,background='selected');self.header.pack(fill='x')
        self.toggle=tk.Button(self.header,text='▾ Input data',command=self.collapse,relief='flat',borderwidth=0,
                              padx=10,pady=6,font=ui_font_name(content,'ui_bold'))
        ui_native_colors(self.toggle,background='selected',foreground='text',activebackground='pressed',activeforeground='text')
        self.toggle.pack(side='left')
        self.summary=tk.Label(self.header,anchor='e',font=ui_font_name(content,'ui'))
        ui_native_colors(self.summary,background='selected',foreground='text');self.summary.pack(side='right',fill='both',expand=True,padx=8)
        self.summary.bind('<Button-1>',lambda _e:self.collapse())
        self.split=ttk.Panedwindow(content,orient='vertical',style='PCS.TPanedwindow',height=800);self.split.pack(fill='both',expand=True)
        self.input_frame=ttk.Frame(self.split,height=320,width=1)
        self.report_frame=ttk.Frame(self.split,height=180,width=1)
        self.plot_frame=ttk.Frame(self.split,height=300,width=1)
        for pane,weight in ((self.input_frame,0),(self.report_frame,0),(self.plot_frame,1)):
            pane.pack_propagate(False);self.split.add(pane,weight=weight)
        self.conditions=ttk.LabelFrame(self.input_frame,text='Analysis conditions',padding=6);self.conditions.pack(fill='x',pady=(3,5))
        self.variables={k:tk.StringVar(master=self.root) for k in ('temperature','method','reference_ln','nucleus_i','nucleus_j','nucleus_c')}
        self.import_ln=tk.StringVar(master=self.root,value='Nd');self.use_errors=tk.BooleanVar(master=self.root)
        self.source=tk.StringVar(master=self.root,value='Model PCS')
        self.input_view=tk.StringVar(master=self.root,value='Datasets')
        self.cells=[]
        self.condition_controls=ttk.Frame(self.conditions)
        specs=[('temperature','Temperature (K)',None),('method','Normalization',tuple(METHODS.values())),
               ('reference_ln','Reference / import Ln',tuple(workflow.CD)),('weighting','Weighting',None),
               ('nucleus_i','Nucleus i',()),('nucleus_j','Nucleus j',()),('nucleus_c','Third nucleus (optional)',()),
               ('source','PCS transfer quantity',tuple(SOURCES.values())),
               ('input_view','Input view',('Datasets','Ref mapping'))]
        self.controls={}
        self.heading_commands={}
        for key,title,values in specs:
            cell=ttk.Frame(self.condition_controls);self.cells.append((key,cell));ttk.Label(cell,text=title).pack(anchor='w')
            if key=='weighting':control=ttk.Checkbutton(cell,text='Use supplied errors',variable=self.use_errors,command=self.save_conditions)
            elif values is None:control=ttk.Entry(cell,textvariable=self.variables[key],width=12);control.bind('<FocusOut>',self.save_conditions);control.bind('<Return>',self.save_conditions)
            else:
                var=self.source if key=='source' else self.input_view if key=='input_view' else self.variables[key]
                control=ttk.Combobox(cell,textvariable=var,values=values,state='readonly',
                    width=max(map(len,values)) if key in ('method','source') else 14)
                control.bind('<<ComboboxSelected>>',(lambda _e:self.refresh()) if key=='input_view' else self.save_conditions)
            control.pack(fill='x');self.controls[key]=control
        self.conditions.bind('<Configure>',self.condition_layout,add='+')
        self.coefficient_frame=ttk.Frame(self.conditions,padding=(4,0))
        coefficient_header=ttk.Frame(self.coefficient_frame);coefficient_header.pack(fill='x',pady=(3,0))
        ttk.Label(coefficient_header,text='Coefficients',font=ui_font_name(content,'ui_bold')).pack(side='left')
        self.coefficient_temperature=ttk.Label(coefficient_header);self.coefficient_temperature.pack(side='left',padx=10)
        self.coefficient_canvas=tk.Canvas(self.coefficient_frame,width=1,height=1,highlightthickness=0,borderwidth=0)
        ui_native_colors(self.coefficient_canvas,background='surface');self.coefficient_canvas.pack(fill='x')
        self.coefficient_grid=tk.Frame(self.coefficient_canvas,borderwidth=0,padx=1,pady=1)
        ui_native_colors(self.coefficient_grid,background='edge')
        self.coefficient_window=self.coefficient_canvas.create_window(0,0,window=self.coefficient_grid,anchor='nw')
        self.coefficient_scroll=ttk.Scrollbar(self.coefficient_frame,orient='horizontal',command=self.coefficient_canvas.xview)
        self.coefficient_canvas.configure(xscrollcommand=self.coefficient_scroll.set)
        self.coefficient_reference=ttk.Label(self.coefficient_frame,
            text='* Sm / Eu: fixed 300 K values    Ref: Bleaney (1972); Pinkerton et al. (1985)',wraplength=800)
        self.coefficient_reference.pack(fill='x',pady=(3,2))
        self.coefficient_labels={};self.coefficient_lns=None
        for widget in (self.coefficient_grid,self.coefficient_canvas):widget.bind('<Configure>',self.coefficient_layout,add='+')
        self.data_caption=ttk.Frame(self.input_frame);self.data_caption.pack(fill='x',pady=(4,3),padx=3)
        self.data_title=ttk.Label(self.data_caption,font=ui_font_name(content,'ui_bold'));self.data_title.pack(side='left')
        self.data_quantity=ttk.Label(self.data_caption);self.data_quantity.pack(side='right')
        self.matching=ttk.Label(self.input_frame,wraplength=800,padding=3);self.matching.pack(side='bottom',fill='x')
        table=ttk.Frame(self.input_frame);table.pack(fill='both',expand=True)
        self.tree=ttk.Treeview(table,columns=('dataset','ln','site','ref','value','error','geometry','kind'),show='headings',height=6,selectmode='extended')
        for key,title,width in (('dataset','Dataset',145),('ln','Ln',45),('site','Nucleus',100),('ref','Ref',55),('value','Shift (ppm)',100),('error','σ (ppm)',80),('geometry','G (Å⁻³)',100),('kind','Input type',105)):
            self.tree.heading(key,text=title);self.tree.column(key,width=width,minwidth=40)
        self.scrollbars(table,self.tree);self.tree.pack(fill='both',expand=True)
        self.tree.bind('<Double-1>',self.edit_ref)
        report_header=ttk.Frame(self.report_frame);report_header.pack(fill='x')
        ttk.Label(report_header,text='Result',font=ui_font_name(content,'ui_bold'),padding=4).pack(side='left')
        self.status=ttk.Label(report_header,padding=4);self.status.pack(side='right')
        self.report=tk.Text(self.report_frame,width=1,height=6,wrap='none',font=ui_font_name(content,'report'),state='disabled',padx=8,pady=4)
        self.scrollbars(self.report_frame,self.report);self.report.pack(fill='both',expand=True)
        bar=ttk.Frame(self.plot_frame);bar.pack(fill='x')
        self.diagnostic_title=ttk.Label(bar,text='Diagnostics',font=ui_font_name(content,'ui_bold'));self.diagnostic_title.pack(side='left')
        self.target=tk.StringVar(master=self.root);self.view=tk.StringVar(master=self.root,value='Correlation / residuals')
        self.target_combo=ttk.Combobox(bar,textvariable=self.target,state='readonly',width=18);self.target_combo.pack(side='left',padx=8)
        self.view_combo=ttk.Combobox(bar,textvariable=self.view,state='readonly',width=24);self.view_combo.pack(side='right')
        for combo in (self.target_combo,self.view_combo):combo.bind('<<ComboboxSelected>>',lambda _e:self.draw(force=True))
        self.figure=Figure(figsize=(8,3),dpi=100,layout='constrained');self.canvas=FigureCanvasTkAgg(self.figure,master=self.plot_frame)
        self.canvas.get_tk_widget().configure(width=1,height=1);self.canvas.get_tk_widget().pack(fill='both',expand=True)
        self.canvas.get_tk_widget().bind('<Map>',self.schedule_visible_plot,add='+')
        self.split.bind('<Configure>',self.restore_layout,add='+');self.split.bind('<ButtonRelease-1>',self.remember_layout,add='+')
        self.scroll.canvas.bind('<Configure>',self.fit_viewport,add='+');self.header.bind('<Configure>',self.fit_viewport,add='+')
        self.tab.bind('<Destroy>',self.destroy,add='+')
        self.tab.bind('<<ThemeChanged>>',lambda _e:self.root.after_idle(lambda:self.draw(force=True) if self.alive else None),add='+')
        state.update(ml_workspace=self,can_run_ml_analysis=self.can_run,advfit_ml_result_text=self.report,
                     advfit_ml_fig=self.figure,advfit_ml_canvas=self.canvas)
        state.setdefault('follow_active_document_refresh_hooks',[]).append(self.refresh)
        self.refresh()

    @staticmethod
    def scrollbars(parent,widget):
        y=ttk.Scrollbar(parent,command=widget.yview);y.pack(side='right',fill='y')
        x=ttk.Scrollbar(parent,orient='horizontal',command=widget.xview);x.pack(side='bottom',fill='x')
        widget.configure(yscrollcommand=y.set,xscrollcommand=x.set)

    def attach_selector(self,group):
        combo=ttk.Combobox(group.content,textvariable=self.mode_var,values=tuple(MODES.values()),state='readonly',width=19)
        combo.grid(row=0,column=0,padx=5,pady=4,sticky='nw');combo.bind('<<ComboboxSelected>>',self.change_mode)

    def change_mode(self,_event=None):
        options=workflow.normalize_options(self.state.get('advfit_ml_options'))
        options['mode']=next(k for k,v in MODES.items() if v==self.mode_var.get());self.mutate('advfit_ml_options',options)

    def condition_layout(self,event=None):
        mode=workflow.normalize_options(self.state.get('advfit_ml_options'))['mode']
        visible=[(k,c) for k,c in self.cells if k=='reference_ln' or (k=='weighting' and mode in ('tensor','reilley','geometry')) or
                 (k=='temperature' and mode!='tensor') or (k in ('method','source') and mode=='reilley') or
                 (k=='input_view' and mode=='tensor') or
                 (k in ('nucleus_i','nucleus_j','nucleus_c') and mode in ('nuclei','geometry'))]
        widths=[cell.winfo_reqwidth()+10 for _,cell in visible]
        four=max((sum(widths[i:i+4]) for i in range(0,len(widths),4)),default=0)+20
        two=max((sum(widths[i:i+2]) for i in range(0,len(widths),2)),default=0)+20
        width=self.conditions.winfo_width()
        has_coefficients=mode in ('reilley','nuclei','geometry')
        side_by_side=has_coefficients and width>two+self.coefficient_grid.winfo_reqwidth()+40
        control_width=width-self.coefficient_grid.winfo_reqwidth()-30 if side_by_side else width
        columns=2 if side_by_side else 4 if control_width>max(800,four) else 2 if control_width>max(430,two) else 1
        key=(mode,columns,side_by_side)
        if key!=getattr(self,'condition_key',None):
            self.condition_key=key
            self.condition_controls.grid_forget();self.coefficient_frame.grid_forget()
            self.conditions.columnconfigure(0,weight=0 if side_by_side else 1);self.conditions.columnconfigure(1,weight=1 if side_by_side else 0)
            self.condition_controls.grid(row=0,column=0,sticky='new')
            if has_coefficients:self.coefficient_frame.grid(row=0 if side_by_side else 1,column=1 if side_by_side else 0,sticky='new',padx=(8,0) if side_by_side else 0,pady=(0,4) if side_by_side else (5,2))
            for _,cell in self.cells:cell.grid_forget()
            for col in range(4):self.condition_controls.columnconfigure(col,weight=1 if col<columns and not side_by_side else 0)
            for i,(_,cell) in enumerate(visible):cell.grid(row=i//columns,column=i%columns,sticky='ew',padx=5,pady=3)
        self.fit_viewport()

    def fit_viewport(self,_event=None):
        if not self.alive:return
        self.matching.configure(wraplength=max(250,self.scroll.canvas.winfo_width()-15))
        minimum=self.report_height+300+(0 if self.collapsed else max(self.input_height,self.minimum_input_height()))
        height=max(minimum,self.scroll.canvas.winfo_height()-self.header.winfo_reqheight())
        if int(self.split.cget('height'))!=height:self.split.configure(height=height)

    def restore_layout(self,_event=None):
        if self.collapsed:self.split.sashpos(0,self.report_height)
        else:
            first=max(self.input_height,self.minimum_input_height())
            self.split.sashpos(0,first);self.split.sashpos(1,first+self.report_height)

    def remember_layout(self,_event=None):
        if self.collapsed:self.report_height=max(85,self.split.sashpos(0))
        else:
            self.input_height=max(self.minimum_input_height(),self.split.sashpos(0))
            self.report_height=max(85,self.split.sashpos(1)-self.input_height)
        self.fit_viewport()

    def minimum_input_height(self):
        return self.conditions.winfo_reqheight()+self.data_caption.winfo_reqheight()+130

    def coefficient_layout(self,_event=None):
        if not self.alive:return
        width=self.coefficient_grid.winfo_reqwidth();height=self.coefficient_grid.winfo_reqheight()
        if int(self.coefficient_canvas.cget('height'))!=height:self.coefficient_canvas.configure(height=height)
        table_width=max(width,self.coefficient_canvas.winfo_width())
        if int(float(self.coefficient_canvas.itemcget(self.coefficient_window,'width'))) != table_width:
            self.coefficient_canvas.itemconfigure(self.coefficient_window,width=table_width)
        self.coefficient_canvas.configure(scrollregion=(0,0,table_width,height))
        overflow=width>self.coefficient_canvas.winfo_width()+1
        if overflow and not self.coefficient_scroll.winfo_manager():
            self.coefficient_scroll.pack(fill='x',before=self.coefficient_reference)
        elif not overflow and self.coefficient_scroll.winfo_manager():
            self.coefficient_scroll.pack_forget();self.coefficient_canvas.xview_moveto(0)
        wrap=max(200,self.coefficient_canvas.winfo_width())
        if int(self.coefficient_reference.cget('wraplength'))!=wrap:self.coefficient_reference.configure(wraplength=wrap)
        self.condition_layout()

    def refresh_coefficients(self,opts,datasets):
        visible=opts['mode'] in ('reilley','nuclei','geometry')
        if not visible:
            self.coefficient_frame.grid_remove();return
        lns=tuple(dict.fromkeys(d['ln'] for d in datasets if d['included'] and d['kind']=='paramagnetic'))
        if lns!=self.coefficient_lns:
            for child in self.coefficient_grid.winfo_children():child.destroy()
            self.coefficient_labels={};self.coefficient_lns=lns
            for column in range(len(workflow.CD)+1):self.coefficient_grid.columnconfigure(column,weight=1 if 0<column<=max(1,len(lns)) else 0)
            for row,title in enumerate(('Ln','C_D','S(T)')):
                label=tk.Label(self.coefficient_grid,text=title,anchor='w',padx=6,pady=2,font=ui_font_name(self.tab,'ui'))
                ui_native_colors(label,background='selected',foreground='text')
                label.grid(row=row,column=0,sticky='nsew',padx=(0,1),pady=(0,1) if row<2 else 0)
                for column,ln in enumerate(lns,1):
                    label=tk.Label(self.coefficient_grid,anchor='center' if row==0 else 'e',padx=6,pady=2,font=ui_font_name(self.tab,'ui'))
                    ui_native_colors(label,background='selected' if row==0 else 'field',foreground='text')
                    label.grid(row=row,column=column,sticky='nsew',padx=(0,1) if column<len(lns) else 0,pady=(0,1) if row<2 else 0);self.coefficient_labels[row,ln]=label
            if not lns:
                label=tk.Label(self.coefficient_grid,text='No included Ln datasets',anchor='w',padx=8,pady=2,font=ui_font_name(self.tab,'ui'))
                ui_native_colors(label,background='field',foreground='text')
                label.grid(row=0,column=1,rowspan=3,sticky='nsew')
        table=workflow.coefficients(opts['temperature'])
        self.coefficient_temperature.configure(text=f"{opts['temperature']:g} K")
        for (row,ln),label in self.coefficient_labels.items():
            text=(ln+('*' if ln in ('Sm','Eu') else '')) if row==0 else f"{table['cd'][ln]:.6g}" if row==1 else f"{table['sz'][ln]:.5f}"
            if label.cget('text')!=text:label.configure(text=text)

    def collapse(self):
        if self.collapsed:self.split.insert(0,self.input_frame,weight=0)
        else:self.remember_layout();self.split.forget(self.input_frame)
        self.collapsed=not self.collapsed;self.toggle.configure(text=('▸' if self.collapsed else '▾')+' Input data')
        self.fit_viewport();self.restore_layout()

    def mutate(self,key,value):
        try:active_document_view(self.state)[key]=value
        except RuntimeError:self.state[key]=value
        mark_document_mutated(self.state,'multi_ln_inputs');self.refresh();self.refresh_commands()

    def refresh_commands(self):
        fn=self.state.get('refresh_fitting_commands')
        if callable(fn):fn()

    def save_conditions(self,_event=None):
        try:
            values={k:v.get() for k,v in self.variables.items()}
            values['method']=next((k for k,v in METHODS.items() if v==values['method']),values['method'])
            opts=workflow.normalize_options(dict(values,
                        mode=workflow.normalize_options(self.state.get('advfit_ml_options'))['mode'],use_errors=self.use_errors.get()))
            if opts!=workflow.normalize_options(self.state.get('advfit_ml_options')):self.mutate('advfit_ml_options',opts)
            return True
        except (ValueError,TypeError) as exc:self.error(exc);return False

    def current_result(self):
        mode=workflow.normalize_options(self.state.get('advfit_ml_options'))['mode']
        return (self.state.get('advfit_ml_results') or {}).get(mode)

    def current_status(self):
        result=self.current_result()
        if result is None:return 'Not analysed'
        try:
            current=compile_snapshot(self.state);saved=result['input_snapshot']
            if 'geometry' in current and 'geometry' in saved:
                a,b=current['geometry'],saved['geometry']
                if (a['refs']==b['refs'] and np.asarray(a['coordinates']).shape==np.asarray(b['coordinates']).shape
                    and np.allclose(a['coordinates'],b['coordinates'],rtol=0,atol=5.1e-9)
                    and np.allclose(current['frame']['metal'],saved['frame']['metal'],rtol=0,atol=5.1e-9)
                    and np.allclose(current['frame']['rotation'],saved['frame']['rotation'],rtol=0,atol=1e-12)):
                    for key in ('geometry','factors','frame'):current[key]=saved[key]
            return 'Current' if workflow.fingerprint(current)==result['fingerprint'] else 'Stale'
        except (ValueError,TypeError,RuntimeError):return 'Stale'

    def can_run(self):
        if self.state.get('advfit_ml_running'):return False
        opts=workflow.normalize_options(self.state.get('advfit_ml_options'))
        if opts['mode']=='parameters':return True
        kind='pcs' if opts['mode']=='tensor' else 'paramagnetic'
        datasets=[d for d in workflow.normalize_datasets(self.state.get('advfit_ml_datasets')) if d['included'] and d['kind']==kind]
        return len(datasets)>=2

    def refresh(self):
        if not self.alive:return
        opts=workflow.normalize_options(self.state.get('advfit_ml_options'))
        datasets=workflow.normalize_datasets(self.state.get('advfit_ml_datasets'))
        self.mode_var.set(MODES[opts['mode']]);self.use_errors.set(opts['use_errors'])
        sites=list(dict.fromkeys(row['site'] for ds in datasets for row in ds['rows']))
        for key,var in self.variables.items():var.set(METHODS[opts[key]] if key=='method' else str(opts[key]))
        for key in ('nucleus_i','nucleus_j','nucleus_c'):self.controls[key].configure(values=('',*sites))
        relevant=[d for d in datasets if d['kind']==('pcs' if opts['mode']=='tensor' else 'paramagnetic')]
        self.conditions.configure(text='Fit conditions' if opts['mode']=='reilley' else 'Analysis conditions')
        self.refresh_coefficients(opts,datasets)
        self.data_title.configure(text='Coefficients' if opts['mode']=='parameters' else 'PCS datasets' if opts['mode']=='tensor' else 'Paramagnetic shifts')
        self.data_quantity.configure(text='' if opts['mode']=='parameters' else 'PCS (ppm)' if opts['mode']=='tensor' else 'δ_para (ppm)')
        self.summary.configure(text=(f"{len(workflow.CD)} Ln · {opts['temperature']:g} K" if opts['mode']=='parameters' else
            f"{sum(d['included'] for d in relevant)} / {len(relevant)} datasets · {len(set(r['site'] for d in relevant for r in d['rows']))} nuclei · "+('main frame fixed' if opts['mode']=='tensor' else f"{opts['temperature']:g} K")))
        rows=[];columns=[];headings=[];widths=[];row_map={}
        if opts['mode']=='parameters':
            table=workflow.coefficients(opts['temperature'])
            columns=['ln','cd','sz','note'];headings=['Ln','C_D (relative)','S(T) (RHF)','Temperature treatment'];widths=[65,150,180,350]
            rows=[(ln,table['cd'][ln],f"{table['sz'][ln]:.6g}",'300 K fixed' if ln in ('Sm','Eu') else 'Calculated at selected T') for ln in workflow.CD]
        elif opts['mode']=='tensor' and self.input_view.get()=='Datasets':
            selected=[d for d in datasets if d['kind']=='pcs'];available=set(self.state.get('atom_ids_raw') or self.state.get('atom_ids') or [])
            columns=['dataset','ln','matched','total','error','status'];headings=['Use / Dataset','Ln','Matched','Total','Supplied σ','Input status'];widths=[230,55,95,85,130,190]
            for index,ds in enumerate(selected):
                matched=sum(r['ref_id'] in available for r in ds['rows']);errors=sum(r['error'] is not None for r in ds['rows'])
                rows.append((('✓ ' if ds['included'] else '— ')+ds['name'],ds['ln'],matched,len(ds['rows']),f"{errors} / {len(ds['rows'])}",'Needs mapping' if matched<len(ds['rows']) else 'Ready'))
                row_map[str(index)]=(ds['id'],None)
        else:
            kind='pcs' if opts['mode']=='tensor' else 'paramagnetic';selected=[d for d in datasets if d['kind']==kind]
            columns=['site','ref','geometry',*[d['id'] for d in selected]]
            headings=['Nucleus','Ref','G (Å⁻³)',*[('✓ ' if d['included'] else '— ')+d['ln']+' (ppm)' for d in selected]];widths=[125,65,125]+[105]*len(selected)
            chosen_sites=list(dict.fromkeys(r['site'] for ds in selected for r in ds['rows']))
            for index,site in enumerate(chosen_sites):
                cells=[next((r for r in ds['rows'] if r['site']==site),None) for ds in selected];present=[r for r in cells if r is not None]
                refs={r['ref_id'] for r in present};gs={r['geometry'] for r in present}
                rows.append((site,('' if present[0]['ref_id'] is None else present[0]['ref_id']) if len(refs)==1 else 'Varies',
                    ('' if present[0]['geometry'] is None else f"{present[0]['geometry']:.6g}") if len(gs)==1 else 'Ln-specific',
                    *[f"{r['value']:.6g}" if r is not None else '—' for r in cells]))
                row_map[str(index)]=(None,site)
        table_key=(columns,headings,rows)
        if table_key!=getattr(self,'last_rows',None):
            self.tree.delete(*self.tree.get_children());self.tree.configure(columns=columns)
            for col,title,width in zip(columns,headings,widths):
                command=''
                if col not in ('site','ref','geometry','dataset','ln','cd','sz','note','matched','total','error','status'):
                    if col not in self.heading_commands:self.heading_commands[col]=self.tree.register(lambda identifier=col:self.toggle_dataset(identifier))
                    command=self.heading_commands[col]
                self.tree.heading(col,text=title,command=command)
                self.tree.column(col,width=width,minwidth=55,anchor='w')
            for index,row in enumerate(rows):self.tree.insert('',tk.END,iid=str(index),values=row)
            self.row_map=row_map;self.last_rows=table_key
        self.matching.configure(text='Click an Ln heading / double-click Dataset to include or exclude. Ref mapping edits apply to the same site across Ln. PCS and total paramagnetic shifts stay separate; Sm/Eu use 300 K coefficients.')
        self.status.configure(text='Analysing…' if self.state.get('advfit_ml_running') else self.current_status())
        if opts['mode']=='parameters':
            self.target.set('');self.view.set('');self.target_combo.configure(values=(),state='disabled');self.view_combo.configure(values=(),state='disabled');self.diagnostic_title.configure(text='Parameter notes')
        else:
            self.target_combo.configure(state='readonly');self.view_combo.configure(state='readonly');self.diagnostic_title.configure(text='Diagnostics')
        self.condition_layout();self.render_report();self.draw()

    def render_report(self):
        result=self.current_result()
        if result is None:
            text='No result.\nChoose an analysis in the Ribbon. Prepare matching datasets, review conditions and select Run Analysis.'
        else:
            opts=result['input_snapshot']['options'];output=result['output']
            lines=[MODES[result['mode']].upper(),f'Result state: {self.current_status()}']
            if result['mode']!='tensor':lines += [f"Recorded temperature: {opts['temperature']:g} K / normalization: {METHODS[opts['method']]}",'Sm/Eu: fixed 300 K values','Ref: Bleaney (1972); Pinkerton et al. (1985)']
            lines += ['Weighting: '+('not applicable' if result['mode']=='parameters' else 'unweighted diagnostic correlation' if result['mode'] in ('scaling','nuclei') else 'supplied errors' if opts['use_errors'] else 'unweighted'),'']
            if result['mode']=='reilley':
                for site in output['sites']:
                    lines.append(f"{site['site']}: {site['status']}")
                    if site['status']!='Fitted':lines.append('  '+site['reason']);continue
                    lines += [f"  G_eff={site['G_eff']:+.6g} / F_eff={site['F_eff']:+.6g}",
                        f"  N={len(site['points'])} / rank={site['fit']['rank']} / condition={site['fit']['condition']:.4g} / dof={site['fit']['dof']}",
                        f"  Original-space RMSD={site['rmsd_original']:.6g} ppm / {site['fit']['note']}",
                        '  Ln       Exp       PCS model    FCS model    Pred−Exp (ppm)']
                    covariance=site['fit']['covariance']
                    lines.insert(len(lines)-1,('  SE(G_eff) / SE(F_eff): '+f"{np.sqrt(covariance[0][0]):.6g} / {np.sqrt(covariance[1][1]):.6g}" if covariance is not None else '  Coefficient errors: not estimated'))
                    lines += [f"  {p['ln']:3} {p['value']:+.6g}  {p['pcs_model']:+.6g}  {p['fcs_model']:+.6g}  {p['residual']:+.6g}" for p in site['points']]
            elif result['mode']=='tensor':
                lines.append('Tensor scalars: E-32 m³; main metal and frame fixed')
                for ds in output['datasets']:
                    lines.append(f"{ds['name']} / {ds['ln']}: {ds['status']}")
                    if ds['status']=='Fitted':
                        ax,rh=ds['fit']['coefficients'];lines.append(f"  Δχ_ax={ax:+.6g} / Δχ_rh={rh:+.6g} / RMSD={ds['fit']['rmsd']:.6g} ppm / rank={ds['fit']['rank']}")
                        lines += [f"  Ref {p['ref_id']}: Exp {p['value']:+.6g} / Pred {p['predicted']:+.6g} / Pred−Exp {p['residual']:+.6g} ppm" for p in ds['points']]
                    else:lines.append('  '+ds['reason'])
                    lines += ['  Skipped: '+json.dumps(ds['skipped'])] if ds['skipped'] else []
            elif result['mode']=='parameters':
                lines.append('Ln       C_D (relative)       S(T) (RHF)')
                lines += [f"{ln:3}      {output['cd'][ln]:+.6g}               {output['sz'][ln]:+.6g}" for ln in output['cd']]
            else:lines.append(json.dumps(output,indent=2,ensure_ascii=False,allow_nan=False))
            lines+=['',*output.get('assumptions',[])];text='\n'.join(lines)
        if text!=getattr(self,'last_report',None):
            self.report.configure(state='normal');self.report.delete('1.0','end');self.report.insert('1.0',text);self.report.configure(state='disabled');self.last_report=text

    def draw(self,force=False):
        if not self.alive or not self.canvas.get_tk_widget().winfo_viewable():return
        result=self.current_result();key=(id(result),self.target.get(),self.view.get(),str(get_app_fonts(self.state)))
        if not force and key==self.last_plot:return
        self.last_plot=key;self.figure.clear()
        if result is None:
            ax=self.figure.add_subplot(111);ax.set_axis_off();ax.text(.5,.5,'Prepare matching inputs and choose Run Analysis.\nReilley uses total paramagnetic shifts; Tensor fitting uses PCS.',ha='center',va='center',transform=ax.transAxes)
            self.canvas.draw_idle();return
        out=result['output'];mode=result['mode'];left,right=self.figure.subplots(1,2)
        if mode in ('reilley','tensor'):
            items=[i for i in out['sites' if mode=='reilley' else 'datasets'] if i['status']=='Fitted']
            labels=[i['site'] if mode=='reilley' else f"{k+1} · {i['name']}" for k,i in enumerate(items)];self.target_combo.configure(values=labels)
            selected=next((k for k,label in enumerate(labels) if label==self.target.get()),0);item=items[selected];self.target.set(labels[selected])
            points=item['points']
            if mode=='reilley':
                self.view_combo.configure(values=('δ / Sz','δ / C_D','PCS / FCS'))
                if self.view.get() not in ('δ / Sz','δ / C_D','PCS / FCS'):self.view.set('δ / Sz')
                table=out['table'];indices=np.arange(len(points));lns=[p['ln'] for p in points]
                if self.view.get()=='PCS / FCS':
                    left.plot(indices,[p['pcs_model'] for p in points],'o-',label='PCS model');left.plot(indices,[p['fcs_model'] for p in points],'s--',label='FCS model');left.set_xticks(indices,lns);left.set_xlabel('Ln');left.set_ylabel('Shift (ppm)');left.legend(loc='best')
                else:
                    cd=np.array([table['cd'][ln] for ln in lns]);sz=np.array([table['sz'][ln] for ln in lns]);div=cd if self.view.get()=='δ / C_D' else sz
                    valid=np.abs(div)>1e-12
                    x=(sz if self.view.get()=='δ / C_D' else cd)[valid]/div[valid]
                    observed=np.array([p['value'] for p in points])[valid]/div[valid];predicted=np.array([p['predicted'] for p in points])[valid]/div[valid]
                    left.scatter(x,observed);order=np.argsort(x);left.plot(x[order],predicted[order])
                    if not np.all(valid):left.text(.03,.95,'Zero normalization denominators omitted',transform=left.transAxes,va='top')
                    left.set_xlabel(r'$\langle S_z\rangle/C_D$' if self.view.get()=='δ / C_D' else r'$C_D/\langle S_z\rangle$')
                    left.set_ylabel(r'$\delta_{para}/C_D$ (ppm)' if self.view.get()=='δ / C_D' else r'$\delta_{para}/\langle S_z\rangle$ (ppm)')
                left.set_title(item['site']+' · Reilley')
                right.scatter(indices,[p['residual'] for p in points]);right.set_xticks(indices,lns);right.set_xlabel('Ln');right.set_ylabel('Pred − Exp (ppm)')
            else:
                self.view_combo.configure(values=('Correlation / residuals','Standardized residuals'))
                if self.view.get() not in ('Correlation / residuals','Standardized residuals'):self.view.set('Correlation / residuals')
                x=[p['value'] for p in points];y=[p['predicted'] for p in points]
                left.scatter(x,y);lo=min(x+y);hi=max(x+y);left.plot([lo,hi],[lo,hi],'--',color='.5');left.set_xlabel(r'$\delta_{Exp}$ (ppm)');left.set_ylabel(r'$\delta_{Pred}$ (ppm)');left.set_title(item['ln']+' · PCS correlation')
                standardized=self.view.get()=='Standardized residuals' and all(p['error'] for p in points)
                right.scatter([p['ref_id'] for p in points],[p['residual']/(p['error'] if standardized else 1.) for p in points]);right.set_xlabel('Ref');right.set_ylabel('(Pred − Exp) / σ' if standardized else 'Pred − Exp (ppm)')
            right.axhline(0,color='.5',ls='--');right.set_title('Original-space residuals')
        elif mode=='scaling':
            self.view_combo.configure(values=('Pair / all-Ln scaling',));self.view.set('Pair / all-Ln scaling')
            series=[s for s in out['series'] if s['status']=='Fitted'];names=[s['ln'] for s in series];self.target_combo.configure(values=names)
            if series:
                s=next((s for s in series if s['ln']==self.target.get()),series[0]);self.target.set(s['ln']);x=[p['x'] for p in s['points']];y=[p['y'] for p in s['points']];left.scatter(x,y);order=np.argsort(x);left.plot(np.array(x)[order],np.array(s['fit']['predicted'])[order]);left.set_xlabel(f"{out['reference_ln']} shift (ppm)");left.set_ylabel(s['ln']+' shift (ppm)')
                right.scatter(np.arange(len(series)),[s['fit']['coefficients'][0] for s in series]);right.set_xticks(np.arange(len(series)),names);right.set_xlabel('Ln');right.set_ylabel('Empirical slope (relative)')
        elif mode in ('nuclei','geometry'):
            points=out['points'];left.scatter([p['x'] for p in points],[p['y'] for p in points]);left.set_xlabel(r'$\delta_j/\langle S_z\rangle$ (ppm)');left.set_ylabel(r'$\delta_i/\langle S_z\rangle$ (ppm)')
            self.view_combo.configure(values=('Geometry ratio','Relative CFP') if mode=='geometry' else ('Nuclei relation',))
            if mode=='geometry' and self.view.get() not in ('Geometry ratio','Relative CFP'):self.view.set('Geometry ratio')
            if mode=='nuclei':self.view.set('Nuclei relation')
            if mode=='geometry':
                if self.view.get()=='Relative CFP' and out.get('cfp'):
                    cp=out['cfp'];x=[p['G_struct'] for p in cp['points']];y=[p['G_eff'] for p in cp['points']];order=np.argsort(x)
                    right.scatter(x,y);right.plot(np.array(x)[order],np.array(x)[order]*cp['origin']['coefficients'][0]);right.set_xlabel(r'$G_{struct}$ (Å⁻³)');right.set_ylabel(r'$G_{eff}$ (relative)')
                else:
                    right.plot(np.arange(len(points)),[p['shift_ratio'] for p in points],'o-',label='Shift ratio');right.plot(np.arange(len(points)),[p['geometry_ratio'] for p in points],'s--',label='Geometry ratio');right.set_ylabel('Ratio (relative)');right.legend(loc='best')
            elif out['nucleus_c']:
                right.scatter([p['Q_B'] for p in points],[p['Q_A'] for p in points]);right.set_xlabel(r'$Q_B$ (ppm)');right.set_ylabel(r'$Q_A$ (ppm)')
            else:
                right.scatter(np.arange(len(points)),[p['Q'] for p in points]);right.set_ylabel('Contact difference Q (ppm)')
            if (not out['nucleus_c'] or mode=='geometry') and self.view.get()!='Relative CFP':right.set_xticks(np.arange(len(points)),[p['ln'] for p in points]);right.set_xlabel('Ln')
        else:
            for ax in (left,right):ax.set_axis_off()
            left.text(.1,.8,'Coefficient values are shown in Input data.',transform=left.transAxes)
        fonts=get_app_fonts(self.state)
        for ax in (left,right):
            ax.grid(alpha=.25);ax.tick_params(labelsize=fonts['plot_tick']);ax.title.set_fontsize(fonts['plot_title']);ax.xaxis.label.set_fontsize(fonts['plot_label']);ax.yaxis.label.set_fontsize(fonts['plot_label'])
            for label in (ax.title,ax.xaxis.label,ax.yaxis.label,*ax.get_xticklabels(),*ax.get_yticklabels()):label.set_fontfamily(fonts['plot_family'])
            legend=ax.get_legend()
            if legend:
                for label in legend.get_texts():label.set_fontsize(fonts['plot_tick']);label.set_fontfamily(fonts['plot_family'])
        self.canvas.draw_idle()

    def schedule_visible_plot(self,_event=None):
        if self.map_id is not None:return
        def draw():
            self.map_id=None
            if self.alive:self.draw(force=True)
        self.map_id=self.root.after_idle(draw)

    def error(self,exc):messagebox.showerror('Multi-Lanthanide',str(exc),parent=self.root)

    def load_csv(self):
        path=filedialog.askopenfilename(parent=self.root,filetypes=[('CSV','*.csv')])
        if not path:return
        try:
            with open(path,encoding='utf-8-sig',newline='') as f:headers=next(csv.reader(f),[])
            with open(path,encoding='utf-8-sig',newline='') as f:
                if any(ln in headers for ln in workflow.CD):new=workflow.parse_matrix_csv(f)
                else:
                    opts=workflow.normalize_options(self.state.get('advfit_ml_options'));new=[workflow.parse_dataset_csv(f,opts['reference_ln'],'pcs' if opts['mode']=='tensor' else 'paramagnetic')]
            for d in new:d.update(id=str(uuid.uuid4()),name=Path(path).stem+' · '+d['ln'])
            self.mutate('advfit_ml_datasets',workflow.normalize_datasets(self.state.get('advfit_ml_datasets'))+new)
        except (ValueError,TypeError,OSError) as exc:self.error(exc)

    def use_current(self):
        rows=[]
        try:
            for o in collect_observations(self.state):
                members=o.target.site_member_ref_ids
                if len(members)!=1 or len(members[0])!=1:raise ValueError('Use Current currently requires single-atom PCS targets; grouped targets need explicit review')
                rows.append(dict(site=o.assignment or f'Ref {o.ref_id}',ref_id=int(members[0][0]),value=o.pcs,error=o.uncertainty_ppm,geometry=None))
        except (ValueError,TypeError,RuntimeError) as exc:self.error(exc);return
        if not rows:self.error('No main PCS observations');return
        opts=workflow.normalize_options(self.state.get('advfit_ml_options'))
        dataset=dict(id=str(uuid.uuid4()),name='Current PCS · '+opts['reference_ln'],ln=opts['reference_ln'],kind='pcs',rows=rows)
        try:self.mutate('advfit_ml_datasets',workflow.normalize_datasets(workflow.normalize_datasets(self.state.get('advfit_ml_datasets'))+[dataset]))
        except ValueError as exc:self.error(exc)

    def remove(self):
        ids={self.row_map[r][0] for r in self.tree.selection() if r in self.row_map}
        if ids=={None}:self.error('Choose the Tensor fitting Dataset view to remove a dataset');return
        self.mutate('advfit_ml_datasets',[d for d in workflow.normalize_datasets(self.state.get('advfit_ml_datasets')) if d['id'] not in ids])

    def clear(self):self.mutate('advfit_ml_datasets',[])

    def toggle_dataset(self,identifier):
        data=workflow.normalize_datasets(self.state.get('advfit_ml_datasets'))
        for d in data:
            if d['id']==identifier:d['included']=not d['included']
        self.mutate('advfit_ml_datasets',data)

    def edit_ref(self,event):
        row=self.tree.identify_row(event.y);col=self.tree.identify_column(event.x)
        if row not in self.row_map:return
        column=self.tree['columns'][int(col[1:])-1] if col else None
        if column=='dataset':
            identifier,_=self.row_map[row];data=workflow.normalize_datasets(self.state.get('advfit_ml_datasets'))
            for d in data:
                if d['id']==identifier:d['included']=not d['included']
            self.mutate('advfit_ml_datasets',data);return
        if column!='ref':return
        box=self.tree.bbox(row,col)
        if not box:return
        editor=ttk.Entry(self.tree);editor.place(x=box[0],y=box[1],width=box[2],height=box[3]);editor.insert(0,self.tree.set(row,'ref'));editor.focus_set()
        def commit(_event=None):
            try:
                ref=int(editor.get()) if editor.get().strip() else None;identifier,site=self.row_map[row]
                data=workflow.normalize_datasets(self.state.get('advfit_ml_datasets'))
                for d in data:
                    for r in d['rows']:
                        if r['site']==site:r['ref_id']=ref
                editor.destroy();self.mutate('advfit_ml_datasets',data)
            except ValueError as exc:self.error(exc)
        editor.bind('<Return>',commit);editor.bind('<Escape>',lambda _e:editor.destroy())

    def validate(self):
        if not self.save_conditions():return
        try:result=workflow.execute_analysis(compile_snapshot(self.state));messagebox.showinfo('Validation',f"{MODES[result['mode']]} input checks passed. No result installed.",parent=self.root)
        except (ValueError,TypeError,RuntimeError) as exc:self.error(exc)

    def run(self):
        if self.state.get('advfit_ml_running'):return
        if not self.save_conditions():return
        try:
            snapshot=compile_snapshot(self.state);identity=capture_worker_identity(self.state)
            records=dict(self.state.get('advfit_ml_results') or {})
        except (ValueError,TypeError,RuntimeError) as exc:self.error(exc);return
        messages=queue.Queue()
        def work():
            try:messages.put((workflow.execute_analysis(snapshot),None))
            except Exception as exc:messages.put((None,f'{type(exc).__name__}: {exc}'))
        self.state['advfit_ml_running']=True;self.refresh_commands()
        def poll():
            self.poll_id=None
            if not self.alive:return
            try:result,error=messages.get_nowait()
            except queue.Empty:self.poll_id=self.root.after(40,poll);return
            self.state['advfit_ml_running']=False
            if identity is not None and not worker_completion_matches(self.state,identity):self.refresh();self.refresh_commands();return
            if error:
                if identity is None or (active_document(self.state) and active_document(self.state).document_id==identity.document_id):self.error(error)
            else:
                records[result['mode']]=result;updates=dict(advfit_ml_results=records,advfit_ml_last_result=result)
                if identity is None:self.state.update(updates)
                else:
                    origin=apply_worker_document_updates(self.state,identity,updates,domain='multi_ln_result')
                    if origin is active_document(self.state):self.state.update(updates)
            self.refresh();self.refresh_commands()
        try:threading.Thread(target=work,daemon=True).start();self.poll_id=self.root.after(40,poll)
        except RuntimeError as exc:self.state['advfit_ml_running']=False;self.refresh_commands();self.error(exc)

    def transfer(self):
        result=(self.state.get('advfit_ml_results') or {}).get('reilley')
        if result is None:self.error('Run Reilley analysis first');return
        try:
            current=compile_snapshot(self.state)
            if current['options']['mode']!='reilley' or workflow.fingerprint(current)!=result['fingerprint']:raise ValueError('Only Current Reilley results can create a PCS dataset')
            grouped={}
            quantity=next(k for k,v in SOURCES.items() if v==self.source.get())
            for site in result['output']['sites']:
                if site['status']!='Fitted':continue
                for p in site['points']:grouped.setdefault(p['ln'],[]).append(dict(site=site['site'],ref_id=p['ref_id'],value=p[quantity],error=None,geometry=p['geometry']))
            data=workflow.normalize_datasets(self.state.get('advfit_ml_datasets'))
            for ln,rows in grouped.items():data.append(dict(id=str(uuid.uuid4()),name='Reilley PCS · '+ln,ln=ln,kind='pcs',rows=rows,
                provenance=dict(result_fingerprint=result['fingerprint'],quantity=quantity,correlated=True)))
            self.mutate('advfit_ml_datasets',data)
            self.input_view.set('Ref mapping')
            opts=workflow.normalize_options(self.state.get('advfit_ml_options'));opts.update(mode='tensor',use_errors=False);self.mutate('advfit_ml_options',opts)
            messagebox.showinfo('PCS transfer review','Local PCS datasets created. Review nucleus-to-Ref mapping before Run Fit. Values share fitted-coefficient uncertainty; errors are not treated as independent. Main PCS data was not changed.',parent=self.root)
        except (ValueError,TypeError) as exc:self.error(exc)

    def export(self):
        result=self.current_result()
        if result is None:return
        path=filedialog.asksaveasfilename(parent=self.root,defaultextension='.json',filetypes=[('JSON','*.json')])
        if path:
            try:Path(path).write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
            except OSError as exc:self.error(exc)

    def template(self):
        path=filedialog.asksaveasfilename(parent=self.root,defaultextension='.csv',filetypes=[('CSV','*.csv')])
        if path:Path(path).write_text('Nucleus,ref_id,G_common,Nd,Nd_err,Eu,Eu_err,Yb,Yb_err\nH1,1,0.002,-1.0,0.03,1.2,0.03,2.4,0.03\n',encoding='utf-8-sig')

    def callbacks(self):
        return {PREFIX+k:v for k,v in dict([('input.use_current',self.use_current),('input.add_csv',self.load_csv),('input.clear',self.clear),('input.remove',self.remove),('input.validate',self.validate),('run',self.run),('results.transfer',self.transfer),('output.export',self.export),('output.template',self.template)]).items()}

    def destroy(self,event):
        if event.widget is not self.tab:return
        self.alive=False
        if self.poll_id:self.root.after_cancel(self.poll_id)
        if self.map_id:self.root.after_cancel(self.map_id)
        hooks=self.state.get('follow_active_document_refresh_hooks',[])
        if self.refresh in hooks:hooks.remove(self.refresh)
