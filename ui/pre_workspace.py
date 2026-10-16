"""PRE workspace with inline conditions and detached main-thread run capture."""
from pathlib import Path
import json
import queue
import threading
import tkinter as tk
from tkinter import ttk,filedialog,messagebox

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from logic.pre_workflow import (PRESnapshot,PREObservation,default_options,normalize_options,
    normalize_rows,parse_pre_csv,execute_fit,result_status,snapshot_from_payload,data_arrays)
from logic.advanced_physics import pre_gamma2,GAMMA_H,LANTHANIDE_DB
from logic.pcs_document import (active_document,active_document_view,mark_document_mutated,
    capture_worker_identity,worker_completion_matches,apply_worker_document_updates)
from ui.multiconformer_workspace import current_geometry
from ui.scrollable_workspace import VerticalScrolledWorkspace
from ui.widget_style import ui_font_name,ui_native_colors
from ui.style import get_app_fonts

PREFIX='analysis.advanced.pre.'


def compile_snapshot(state):
    options=normalize_options(state.get('advfit_pre_options'))
    geometry,elements=current_geometry(state)
    if state.get('fit_override') or geometry.torsion_enabled:
        raise ValueError('PRE fixed-distance model does not support geometry overrides or torsional averaging')
    available=set(geometry.raw_ref_ids)
    observations,skipped=[],[]
    for row in normalize_rows(state.get('advfit_pre_obs')):
        if row['ref_id'] not in available:
            skipped.append(f"PRE row {row['source_row']}: raw Ref {row['ref_id']} not found")
        else:
            observations.append(PREObservation(**row))
    return PRESnapshot(geometry.raw_ref_ids,geometry.raw_coordinates,elements,geometry.metal_position,
        tuple(observations),options['B0'],options['mu_eff'],options['use_errors'],
        options['tau_min_ns'],options['tau_max_ns'],tuple(skipped))


class PREWorkspace:
    def __init__(self,state,notebook):
        self.state,self.root,self.alive=state,state['root'],True
        self.poll_id=self.font_id=None
        self.last_plot=None
        self.collapsed=False
        self.input_height,self.report_height=320,155
        self.input_error=None
        self.tab=ttk.Frame(notebook)
        notebook.add(self.tab,text='PRE')
        state.setdefault('advfit_pre_obs',[])
        state.setdefault('advfit_pre_options',default_options())
        self.scroll=VerticalScrolledWorkspace(self.tab,padding=0)
        self.scroll.pack(fill=tk.BOTH,expand=True)
        ui_native_colors(self.scroll.canvas,background='surface')
        content=self.scroll.content
        self.header=tk.Frame(content,borderwidth=0)
        ui_native_colors(self.header,background='selected')
        self.header.pack(fill=tk.X,pady=(0,3))
        self.toggle=tk.Button(self.header,text='▾ Input data',command=self.collapse,anchor='w',
            relief='flat',borderwidth=0,highlightthickness=1,padx=10,pady=6,font=ui_font_name(content,'ui_bold'))
        ui_native_colors(self.toggle,background='selected',foreground='text',activebackground='pressed',
            activeforeground='text',highlightbackground='selected',highlightcolor='accent')
        self.toggle.pack(side=tk.LEFT)
        self.summary=tk.Label(self.header,anchor='e',font=ui_font_name(content,'ui'),borderwidth=0)
        ui_native_colors(self.summary,background='selected',foreground='text')
        self.summary.pack(side=tk.RIGHT,fill=tk.BOTH,expand=True,padx=8)
        self.summary.bind('<Button-1>',lambda _e:self.collapse())
        self.toggle.bind('<Return>',lambda _e:self.collapse())
        self.split=ttk.Panedwindow(content,orient=tk.VERTICAL,style='PCS.TPanedwindow',height=750)
        self.split.pack(fill=tk.BOTH,expand=True)
        self.input_frame=ttk.Frame(self.split,height=320,width=1)
        self.report_frame=ttk.Frame(self.split,height=155,width=1)
        self.plot_frame=ttk.Frame(self.split,height=275,width=1)
        for pane,weight in ((self.input_frame,0),(self.report_frame,0),(self.plot_frame,1)):
            pane.pack_propagate(False)
            self.split.add(pane,weight=weight)
        self.conditions=ttk.LabelFrame(self.input_frame,text='Fit conditions',padding=8)
        self.conditions.pack(fill=tk.X,pady=(0,5))
        self.fields={k:tk.StringVar(master=self.root) for k in ('B0','preset','mu_eff','tau_min_ns','tau_max_ns')}
        self.use_errors=tk.BooleanVar(master=self.root)
        self.cells=[]
        for key,title in (('B0','B₀ (T)'),('preset','Ln preset'),('mu_eff','μ_eff (μB)'),('weighting','Weighting')):
            cell=ttk.Frame(self.conditions)
            ttk.Label(cell,text=title).pack(anchor='w',pady=(0,4))
            if key=='preset':
                control=ttk.Combobox(cell,textvariable=self.fields[key],values=tuple(LANTHANIDE_DB)+('Custom',),state='readonly',width=9)
                control.bind('<<ComboboxSelected>>',self.preset_changed)
            elif key=='weighting':
                control=ttk.Checkbutton(cell,text='Use supplied Γ₂ errors',variable=self.use_errors,command=self.save_conditions)
            else:
                control=ttk.Entry(cell,textvariable=self.fields[key],width=10)
                control.bind('<FocusOut>',self.save_conditions)
                control.bind('<Return>',self.save_conditions)
            control.pack(fill=tk.X)
            self.cells.append(cell)
        self.bottom=ttk.Frame(self.conditions)
        self.bottom.columnconfigure(1,weight=1)
        bounds=ttk.Frame(self.bottom)
        self.bounds=bounds
        bounds.grid(row=0,column=0,sticky='nw',padx=(0,20))
        ttk.Label(bounds,text='τ_c range (ns)').pack(anchor='w',pady=(0,4))
        pair=ttk.Frame(bounds)
        pair.pack(fill=tk.X)
        for i,key in enumerate(('tau_min_ns','tau_max_ns')):
            if i:ttk.Label(pair,text='—').pack(side=tk.LEFT,padx=6)
            entry=ttk.Entry(pair,textvariable=self.fields[key],width=9)
            entry.pack(side=tk.LEFT)
            entry.bind('<FocusOut>',self.save_conditions)
            entry.bind('<Return>',self.save_conditions)
        model=ttk.Frame(self.bottom)
        self.model=model
        model.grid(row=0,column=1,sticky='nw')
        ttk.Label(model,text='Model').pack(anchor='w',pady=(0,4))
        self.model_label=ttk.Label(model,text='Proton Γ₂ · effective τ_c\nFixed geometry and μ_eff')
        self.model_label.pack(anchor='w')
        self.conditions.bind('<Configure>',self.condition_layout,add='+')
        self.matching=ttk.Label(self.input_frame,text='Load proton PRE increments from the Ribbon',padding=(4,3))
        self.matching.pack(side=tk.BOTTOM,fill=tk.X)
        table=ttk.Frame(self.input_frame)
        table.pack(fill=tk.BOTH,expand=True)
        self.tree=ttk.Treeview(table,columns=('ref','r','rate','sigma','match'),show='headings',height=6)
        for key,title,width in (('ref','Ref / nucleus',105),('r','r (Å)',100),('rate','Γ₂,Exp (s⁻¹)',140),('sigma','σ (s⁻¹)',110),('match','Matching',180)):
            self.tree.heading(key,text=title)
            self.tree.column(key,width=width,minwidth=65,anchor='w' if key in ('ref','match') else 'e')
        self.scrollbars(table,self.tree)
        self.tree.pack(fill=tk.BOTH,expand=True)
        report_header=ttk.Frame(self.report_frame)
        report_header.pack(fill=tk.X)
        ttk.Label(report_header,text='Result',font=ui_font_name(content,'ui_bold'),padding=4).pack(side=tk.LEFT)
        self.status=ttk.Label(report_header,padding=4)
        self.status.pack(side=tk.RIGHT)
        self.report=tk.Text(self.report_frame,height=6,width=1,wrap='none',padx=8,pady=4,
                            font=ui_font_name(content,'report'),state='disabled')
        self.scrollbars(self.report_frame,self.report)
        self.report.pack(fill=tk.BOTH,expand=True)
        bar=ttk.Frame(self.plot_frame)
        bar.pack(fill=tk.X)
        ttk.Label(bar,text='Diagnostics',font=ui_font_name(content,'ui_bold'),padding=4).pack(side=tk.LEFT)
        self.view=tk.StringVar(master=self.root,value='Distance profile')
        views=ttk.Combobox(bar,textvariable=self.view,values=('Distance profile','Correlation','τ_c scan'),state='readonly',width=17)
        views.pack(side=tk.RIGHT,padx=4)
        views.bind('<<ComboboxSelected>>',lambda _e:self.draw_result(force=True))
        self.figure=Figure(figsize=(8,3),dpi=100,layout='constrained')
        self.canvas=FigureCanvasTkAgg(self.figure,master=self.plot_frame)
        self.canvas.get_tk_widget().configure(width=1,height=1)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH,expand=True)
        self.split.bind('<Configure>',self.restore_layout,add='+')
        self.split.bind('<ButtonRelease-1>',self.remember_layout,add='+')
        self.scroll.canvas.bind('<Configure>',self.fit_viewport,add='+')
        self.header.bind('<Configure>',self.fit_viewport,add='+')
        self.tab.bind('<Destroy>',self.destroy,add='+')
        self.tab.bind('<<ThemeChanged>>',self.theme_changed,add='+')
        state.update(pre_workspace=self,can_run_pre_fit=self.can_run,advfit_pre_result_text=self.report,
                     advfit_pre_fig=self.figure,advfit_pre_canvas=self.canvas)
        state.setdefault('follow_active_document_refresh_hooks',[]).append(self.refresh)
        self.refresh()

    @staticmethod
    def scrollbars(parent,widget):
        y=ttk.Scrollbar(parent,command=widget.yview)
        y.pack(side=tk.RIGHT,fill=tk.Y)
        x=ttk.Scrollbar(parent,orient=tk.HORIZONTAL,command=widget.xview)
        x.pack(side=tk.BOTTOM,fill=tk.X)
        widget.configure(yscrollcommand=y.set,xscrollcommand=x.set)

    def condition_layout(self,event):
        four_width=sum(cell.winfo_reqwidth()+10 for cell in self.cells)+20
        two_width=max(sum(cell.winfo_reqwidth()+10 for cell in self.cells[i:i+2]) for i in (0,2))+20
        columns=4 if event.width>=max(600,four_width) else 2 if event.width>=max(370,two_width) else 1
        if columns!=getattr(self,'condition_columns',None):
            self.condition_columns=columns
            for col in range(4):self.conditions.columnconfigure(col,weight=1 if col<columns else 0)
            for i,cell in enumerate(self.cells):cell.grid(row=i//columns,column=i%columns,sticky='ew',padx=(0,10),pady=(0,8))
            self.bottom.grid(row=(len(self.cells)+columns-1)//columns,column=0,columnspan=columns,sticky='ew')
            self.model.grid(row=1 if columns==1 else 0,column=0 if columns==1 else 1,
                            sticky='nw',pady=(8,0) if columns==1 else 0)
        self.fit_viewport()

    def fit_viewport(self,_event=None):
        if not self.alive:return
        minimum_input=max(self.input_height,self.conditions.winfo_reqheight()+110)
        minimum=self.report_height+275+(0 if self.collapsed else minimum_input)
        available=self.scroll.canvas.winfo_height()-self.header.winfo_reqheight()-3
        height=max(minimum,available)
        if int(self.split.cget('height'))!=height:self.split.configure(height=height)

    def restore_layout(self,_event=None):
        if self.collapsed:self.split.sashpos(0,self.report_height)
        else:
            height=max(self.input_height,self.conditions.winfo_reqheight()+110)
            self.split.sashpos(0,height)
            self.split.sashpos(1,height+self.report_height)

    def remember_layout(self,_event=None):
        if self.collapsed:self.report_height=max(85,self.split.sashpos(0))
        else:
            self.input_height=max(self.conditions.winfo_reqheight()+110,self.split.sashpos(0))
            self.report_height=max(85,self.split.sashpos(1)-self.input_height)
        self.fit_viewport()

    def collapse(self):
        if self.collapsed:self.split.insert(0,self.input_frame,weight=0)
        else:
            self.remember_layout()
            self.split.forget(self.input_frame)
        self.collapsed=not self.collapsed
        self.toggle.configure(text=('▸' if self.collapsed else '▾')+' Input data')
        self.fit_viewport()
        self.restore_layout()

    def refresh_commands(self):
        callback=self.state.get('refresh_fitting_commands')
        if callable(callback):callback()

    def mutate(self,key,value):
        try:active_document_view(self.state)[key]=value
        except RuntimeError:self.state[key]=value
        mark_document_mutated(self.state,'pre_inputs')
        self.refresh()
        self.refresh_commands()

    def preset_changed(self,_event=None):
        preset=self.fields['preset'].get()
        if preset in LANTHANIDE_DB:self.fields['mu_eff'].set(str(LANTHANIDE_DB[preset]['mu_eff']))
        self.save_conditions()

    def save_conditions(self,_event=None):
        try:
            options=normalize_options(dict({k:v.get() for k,v in self.fields.items()},use_errors=self.use_errors.get()))
            if options['preset'] in LANTHANIDE_DB and options['mu_eff']!=LANTHANIDE_DB[options['preset']]['mu_eff']:
                options['preset']='Custom'
            self.input_error=None
            if options!=normalize_options(self.state.get('advfit_pre_options')):self.mutate('advfit_pre_options',options)
        except ValueError as exc:
            self.input_error=str(exc)
            self.matching.configure(text=str(exc))
            self.refresh_commands()

    def current_status(self):
        result=self.state.get('advfit_pre_last_result')
        if not result:return 'Not fitted'
        try:return result_status(result,compile_snapshot(self.state))
        except (ValueError,TypeError,KeyError,RuntimeError):return 'Stale'

    def can_run(self):
        return bool(not self.input_error and not self.state.get('advfit_pre_running') and self.state.get('advfit_pre_obs')
                    and (self.state.get('atom_data_raw') or self.state.get('atom_data')))

    def refresh(self):
        if not self.alive:return
        options=normalize_options(self.state.get('advfit_pre_options'))
        for key,var in self.fields.items():var.set(str(options[key]))
        self.use_errors.set(options['use_errors'])
        self.input_error=None
        try:
            snapshot=compile_snapshot(self.state)
            raw_map=dict(zip(snapshot.ref_ids,snapshot.coordinates))
            rows=[(f'{o.ref_id} / ¹H',f'{np.linalg.norm(np.array(raw_map[o.ref_id])-snapshot.metal):.3f}',
                   f'{o.gamma2:.6g}','—' if o.uncertainty is None else f'{o.uncertainty:.4g}','Matched') for o in snapshot.observations]
            rows += [(reason,'—','—','—','Unmatched') for reason in snapshot.skipped]
            self.summary.configure(text=f"{len(snapshot.observations)} matched ¹H · {options['B0']:g} T · μ_eff {options['mu_eff']:g} μB · τ_c {options['tau_min_ns']:g}–{options['tau_max_ns']:g} ns")
            legacy=any(o.interpretation.startswith('legacy') for o in snapshot.observations)
            self.matching.configure(text=f"{len(snapshot.skipped)} unmatched · Γ₂ = R₂,para − R₂,dia · "+('Legacy R₂ alias: verify increment interpretation' if legacy else 'Supplied σ' if options['use_errors'] else 'Assumed errors: 5%, minimum 0.1 s⁻¹'))
        except (ValueError,TypeError,KeyError,RuntimeError) as exc:
            rows=[]
            self.summary.configure(text='Prepare structure and proton PRE increments')
            self.matching.configure(text=str(exc))
        if rows!=getattr(self,'last_rows',None):
            self.tree.delete(*self.tree.get_children())
            for row in rows:self.tree.insert('',tk.END,values=row)
            self.last_rows=rows
        self.status.configure(text='Fitting…' if self.state.get('advfit_pre_running') else self.current_status())
        self.render_report()
        self.draw_result()

    def render_report(self):
        result=self.state.get('advfit_pre_last_result')
        if not result or 'input_snapshot' not in result:
            text='No verified PRE result.\nPrepare structure and load proton Γ₂ increments.\nValidate inputs and select Run Fit.'
        else:
            data=result['input_snapshot']
            lines=['PRE — SIMPLIFIED PROTON Γ₂',f'Result state: {self.current_status()}',
                f"Recorded conditions: B₀ {data['B0']:g} T / μ_eff {data['mu_eff']:g} μB",
                'Input: PRE increment R₂,para − R₂,dia; fixed raw geometry and metal',
                'Weighting: '+('supplied Γ₂ uncertainties' if data['use_errors'] else 'assumed max(5% of observed, 0.1 s⁻¹)'),
                f"Data: {len(result['per_point'])} matched proton observations / {len(result['skipped'])} unmatched",'',
                f"Effective τ_c: {result['tau_ns']:.6g} ns",
                f"Bounds: {data['tau_min_ns']:g}–{data['tau_max_ns']:g} ns / {result['boundary']} solution",
                f"Γ₂ RMSD: {result['rmsd']:.6g} s⁻¹",f"Weighted residual sum of squares: {result['weighted_sum_squares']:.6g}",
                'Confidence intervals are not estimated.']
            if result['boundary']!='interior':lines += ['Boundary reached: the unconstrained best amplitude lies outside the selected tau range.']
            if result['nonpositive_increments']:lines += [f"{result['nonpositive_increments']} nonpositive measured increments; linear plot axes are used."]
            lines += ['', 'Ref\tr (Å)\tΓ₂,Exp\tΓ₂,Pred\tPred − Exp (s⁻¹)']
            lines += [f"{p['ref_id']}\t{p['distance_angstrom']:.3f}\t{p['experimental']:+.6g}\t{p['predicted']:+.6g}\t{p['residual']:+.6g}" for p in result['per_point']]
            lines += ['', 'Unmatched input: '+('; '.join(result['skipped']) or 'none')]
            text='\n'.join(lines)
        if text!=getattr(self,'last_report',None):
            self.report.configure(state='normal')
            self.report.delete('1.0',tk.END)
            self.report.insert('1.0',text)
            self.report.configure(state='disabled')
            self.last_report=text

    def draw_result(self,force=False):
        result=self.state.get('advfit_pre_last_result')
        fonts=get_app_fonts(self.state)
        key=(id(result),self.view.get(),tuple(fonts.get(k) for k in ('plot_family','plot_title','plot_label','plot_tick')))
        if not force and key==self.last_plot:return
        self.last_plot=key
        self.figure.clear()
        if not result or 'input_snapshot' not in result:
            ax=self.figure.add_subplot(111)
            ax.set_axis_off()
            ax.text(.5,.5,'Prepare the structure and load proton Γ₂ increments.\nValidate inputs and select Run Fit to evaluate the PRE model.',ha='center',va='center',
                    transform=ax.transAxes,fontsize=fonts.get('plot_label',10),fontfamily=fonts.get('plot_family'))
        else:
            snapshot=snapshot_from_payload(result['input_snapshot'])
            distance,observed,errors=data_arrays(snapshot)
            predicted=np.array([p['predicted'] for p in result['per_point']])
            positive=np.all(observed>0)
            left,right=self.figure.subplots(1,2)
            if self.view.get()=='Correlation':
                left.errorbar(observed,predicted,xerr=errors,fmt='o',ms=4,capsize=2)
                lo,hi=min(observed.min(),predicted.min()),max(observed.max(),predicted.max())
                left.plot([lo,hi],[lo,hi],ls='--',color='0.45',lw=1)
                if positive:left.set_xscale('log');left.set_yscale('log')
                left.set_xlabel(r'$\Gamma_{2,\mathrm{Exp}}$ (s⁻¹)')
                left.set_ylabel(r'$\Gamma_{2,\mathrm{Pred}}$ (s⁻¹)')
                left.set_title('Γ₂ correlation')
            else:
                curve=np.linspace(distance.min()*.94,distance.max()*1.06,160)
                left.plot(curve,pre_gamma2(curve,snapshot.mu_eff,result['tau_c'],GAMMA_H*snapshot.B0),label='Model')
                left.errorbar(distance,observed,yerr=errors,fmt='o',ms=4,capsize=2,label='Measured')
                if positive:left.set_yscale('log')
                left.set_xlabel('r (Å)');left.set_ylabel(r'$\Gamma_2$ (s⁻¹)');left.set_title('PRE distance profile')
                left.legend(loc='best',fontsize=fonts.get('plot_tick',9))
            if self.view.get()=='τ_c scan':
                times=np.logspace(np.log10(snapshot.tau_min_ns),np.log10(snapshot.tau_max_ns),220)*1e-9
                objective=[np.sum(((pre_gamma2(distance,snapshot.mu_eff,t,GAMMA_H*snapshot.B0)-observed)/errors)**2) for t in times]
                right.plot(times*1e9,objective)
                right.axvline(result['tau_ns'],ls='--',color='0.45',lw=1)
                right.set_xscale('log');right.set_yscale('symlog',linthresh=1)
                right.set_ylim(bottom=max(0,min(min(objective),result['weighted_sum_squares'])*.8))
                right.set_xlabel(r'$\tau_c$ (ns)');right.set_ylabel('Weighted sum of squares')
                right.set_title('Recorded-data τ_c scan')
            else:
                right.scatter([o.ref_id for o in snapshot.observations],(predicted-observed)/errors,s=24)
                right.axhline(0,ls='--',color='0.45',lw=1)
                right.set_xlabel('Ref');right.set_ylabel(r'$(\Gamma_{2,\mathrm{Pred}}-\Gamma_{2,\mathrm{Exp}})/\sigma$')
                right.set_title('Standardized residuals')
            for ax in (left,right):
                ax.grid(alpha=.25);ax.tick_params(labelsize=fonts.get('plot_tick',9))
                ax.title.set_fontsize(fonts.get('plot_title',11))
                for label in (ax.xaxis.label,ax.yaxis.label):label.set_fontsize(fonts.get('plot_label',10))
                for label in [ax.title,ax.xaxis.label,ax.yaxis.label,*ax.get_xticklabels(),*ax.get_yticklabels()]:
                    if fonts.get('plot_family'):label.set_fontfamily(fonts['plot_family'])
        self.canvas.draw_idle()

    def error(self,error):messagebox.showerror('PRE',str(error),parent=self.root)

    def load_csv(self):
        path=filedialog.askopenfilename(parent=self.root,title='Load proton PRE increments',filetypes=[('CSV','*.csv')])
        if not path:return
        try:
            with open(path,encoding='utf-8-sig',newline='') as stream:rows=parse_pre_csv(stream)
            self.mutate('advfit_pre_obs',rows)
            if any(r['interpretation']=='legacy_increment_alias' for r in rows):
                messagebox.showwarning('PRE input interpretation','Legacy r2/r2_para values are treated as Γ₂ increments. Verify they represent R₂,para − R₂,dia; raw R₂ is not a PRE increment.',parent=self.root)
        except (OSError,ValueError) as exc:self.error(exc)

    def validate(self):
        self.save_conditions()
        if self.input_error:self.error(self.input_error);return
        try:
            result=execute_fit(compile_snapshot(self.state))
            messagebox.showinfo('PRE validation',f"{len(result['per_point'])} matched protons / {len(result['skipped'])} unmatched\nFixed-distance model valid; {result['boundary']} solution. No result installed.",parent=self.root)
        except (ValueError,TypeError,KeyError,RuntimeError) as exc:self.error(exc)

    def run(self):
        if self.state.get('advfit_pre_running'):return
        self.save_conditions()
        if self.input_error:self.error(self.input_error);return
        try:snapshot,identity=compile_snapshot(self.state),capture_worker_identity(self.state)
        except (ValueError,TypeError,KeyError,RuntimeError) as exc:self.error(exc);return
        results=queue.Queue()
        def work():
            try:results.put((execute_fit(snapshot),None))
            except Exception as exc:results.put((None,f'{type(exc).__name__}: {exc}'))
        self.state['advfit_pre_running']=True
        self.refresh();self.refresh_commands()
        def poll():
            self.poll_id=None
            if not self.alive:return
            try:payload,error=results.get_nowait()
            except queue.Empty:self.poll_id=self.root.after(40,poll);return
            self.state['advfit_pre_running']=False
            if identity is not None and not worker_completion_matches(self.state,identity):self.refresh();self.refresh_commands();return
            if error:
                if identity is None or (active_document(self.state) and active_document(self.state).document_id==identity.document_id):self.error(error)
            elif identity is None:self.state['advfit_pre_last_result']=payload
            else:
                origin=apply_worker_document_updates(self.state,identity,{'advfit_pre_last_result':payload},domain='pre_result')
                if origin is active_document(self.state):self.state['advfit_pre_last_result']=payload
            self.refresh();self.refresh_commands()
        try:
            threading.Thread(target=work,daemon=True).start()
            self.poll_id=self.root.after(40,poll)
        except RuntimeError as exc:
            self.state['advfit_pre_running']=False
            self.refresh_commands();self.error(exc)

    def export(self):
        result=self.state.get('advfit_pre_last_result')
        if not result:return
        path=filedialog.asksaveasfilename(parent=self.root,title='Export recorded PRE result',defaultextension='.json',filetypes=[('JSON','*.json')])
        if path:
            try:Path(path).write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
            except (OSError,ValueError) as exc:self.error(exc)

    def callbacks(self):
        return {PREFIX+key:method for key,method in {'input.load_csv':self.load_csv,'input.validate':self.validate,'run':self.run,'output.export':self.export}.items()}

    def theme_changed(self,_event=None):
        if self.font_id is None and self.alive:
            def refresh():self.font_id=None;self.draw_result()
            self.font_id=self.root.after_idle(refresh)

    def destroy(self,event):
        if event.widget is not self.tab:return
        self.alive=False
        self.state['advfit_pre_running']=False
        for task in (self.poll_id,self.font_id):
            if task:self.root.after_cancel(task)
        hooks=self.state.get('follow_active_document_refresh_hooks',[])
        if self.refresh in hooks:hooks.remove(self.refresh)
