"""Explicit author source/metadata/private scalar QA. Default performs no IO."""
import argparse

def qa():
    import ast,base64,copy,hashlib,json,math,re,subprocess,sys
    from pathlib import Path
    import itsmixer18_protocol as p
    stage=Path(__file__).resolve().parent;work=stage.parents[1];checks=[]
    def check(value,label):
        assert value,label;checks.append(label)
    def reject(fn,label):
        try:fn()
        except (ValueError,TypeError,KeyError):checks.append(label);return
        raise AssertionError('unblocked weakening '+label)
    def sh(b):return hashlib.sha256(b).hexdigest()
    def read(n):return json.loads((stage/n).read_bytes())
    original=work/'main_experiments/MODELS2026_ITSMIXER_MINIMUM6_SCIENCE_SOURCE_PENDING_20261012'
    mb=(original/'ITSMIXER6_SOURCE_PACKAGE_MANIFEST_PENDING.json').read_bytes()
    check(sh(mb)==p.H96_SCIENCE_MANIFEST_SHA256,'unchanged exact e8fe source manifest')
    proof=read('AUTHOR_EXACT_VARIANT_AND_KERNEL_ADAPTATION_PROOF.json')
    for n,pin in proof['original_e8fe_h96_bytes_retained'].items():
        check((stage/n).read_bytes()==(original/n).read_bytes() and sh((stage/n).read_bytes())==pin,'original H96 exact bytes '+n)
    root=read(p.ROOT_REMAINING18_POLICY_NAME)
    check(sh((stage/p.ROOT_REMAINING18_POLICY_NAME).read_bytes())==p.ROOT_REMAINING18_POLICY_SHA256,'exact ROOT15/15/20 pre-score policy')
    check(root['maximum_epochs_by_horizon']=={'192':15,'336':15,'720':20},'ROOT per-horizon full budgets')
    check(root['actual_remaining18_allowed_before_minimum6_strict12_gate_pass'] is False and root['H96_checkpoint_retraining_allowed'] is False,'ROOT qualification and H96 no retrain')
    parent=(work/'main_experiments/MODELS2026_PILOT_20261011/source_qa_package/MINIMUM12_PLAN.json').read_bytes()
    check(sh(parent)==p.PARENT_PLAN_SHA256,'original immutable metadata plan pin')
    metadata=json.loads(parent)['data_metadata'];recipe=read('ITSMIXER_REMAINING18_CONFIG_RECIPE_PENDING.json')
    protocol=read('ITSMIXER_REMAINING18_SCIENCE_PROTOCOL_PENDING.json')
    check(protocol['native_data_metadata']==metadata,'exact unchanged six native metadata objects')
    check(recipe['runs']==protocol['runs'] and len(recipe['runs'])==18,'exact eighteen recipe bindings')
    expected_ids={'ITSMixer_'+d+'_H'+str(h) for h in (192,336,720) for d in p.DATASETS}
    check(set(recipe['runs'])==expected_ids and not any(x.endswith('_H96') for x in expected_ids),'only remaining18; H96 reuse')
    faults=[('epochs',1),('batch_size',16),('patience',3),('length',512),('learning_rate',.0001),
        ('optimizer_weight_decay',0.),('criterion','MSE'),('checkpoint_criterion','MSE'),('train_drop_last',False),
        ('train_shuffle',False),('validation_drop_last',True),('validation_all_windows',False),('epoch_window_cap',1024),
        ('inverse_external_scaler_before_metrics',True),('test_allowed',True),('onecycle_scheduler_step_called',True),
        ('h96_retraining_allowed',True),('full_original_trainval_prefix_resident',False)]
    for rid,cfg in recipe['runs'].items():
        h=cfg['horizon'];ds=cfg['dataset'];t,v=metadata[ds]['bounds'][:2];c=metadata[ds]['channels']
        p.validate_config(cfg,metadata[ds]);check(True,'exact fixed typed config '+rid)
        train=t-96-h+1;val=v-t-h+1;full=train//256;epochs=20 if h==720 else 15
        check(cfg['train_windows_available']==train and cfg['validation_windows']==val,'independent legal TRAIN VAL window counts '+rid)
        check(cfg['validation_elements']==val*h*c and cfg['validation_tail_windows']==val%256,'all H*C full tail elements '+rid)
        check(cfg['train_windows_used_per_epoch']==256*full and cfg['train_windows_dropped_per_epoch']==train%256,
              'shuffled TRAIN drop tail full universe disclosure '+rid)
        check(cfg['epochs']==epochs and cfg['expected_optimizer_steps_at_full_epoch_budget']==epochs*full,'full demonstrated epoch step budget '+rid)
        check(cfg['validation_target_start_inclusive']==t and cfg['validation_target_start_exclusive']==v-h+1
              and cfg['train_target_start_inclusive']==96 and cfg['train_target_start_exclusive']==t-h+1,'complete target boundaries '+rid)
        check(cfg['source_sha256']==metadata[ds]['source_sha256'] and cfg['prefix_sha256']==metadata[ds]['prefix_sha256']
              and cfg['calendar_sha256']==metadata[ds]['calendar_sha256'],'same source prefix calendar identities '+rid)
        check(all(cfg[k]==metadata[ds][k] for k in ('scaler_mean_sha256','scaler_std_sha256')),'same TRAIN-only scaler '+rid)
        for field,value in faults:
            bad=copy.deepcopy(cfg);bad[field]=value
            reject(lambda b=bad:p.validate_config(b,metadata[ds]),'weakening rejected '+rid+'.'+field)
        for field,value in [('model_source_file','ITSMixer_ECL96_author_exact.py'),('model_source_sha256','0'*64),
                            ('author_corresponding_ecl_git_blob_sha1','0'*40),('enc_in',1),
                            ('author_first_loop_iterations',1),('effective_hidden_width',512)]:
            bad=copy.deepcopy(cfg);bad[field]=value
            reject(lambda b=bad:p.validate_config(b,metadata[ds]),'variant/source weakening rejected '+rid+'.'+field)
    for h,variant in p.VARIANTS.items():
        body=(stage/variant['source_file']).read_bytes();tree=ast.parse(body.decode())
        check(sh(body)==variant['source_sha256']==root['corresponding_ECL_horizon_source_sha256'][str(h)],'ROOT corresponding exact model '+str(h))
        check(body==(original/f'author_source/Experiments/electricity-{h}/seq-len-512.py').read_bytes(),'standalone byte-identical author '+str(h))
        blob=hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()
        prov=read('provenance/Experiments__electricity-'+str(h)+'__seq-len-512.py.GITHUB_BLOB.json')
        check(blob==variant['git_blob_sha1']==prov['sha'] and base64.b64decode(prov['content'])==body,'official git blob content and SHA '+str(h))
        cls=[n for n in tree.body if isinstance(n,ast.ClassDef)]
        check(tuple(n.name for n in cls)==p.AUTHOR_CLASSES,'all exact author classes original creation order '+str(h))
        check([ast.unparse(n.iter) for n in ast.walk(tree) if isinstance(n,ast.For)]==['range('+str(variant['first_loop'])+')','range('+str(variant['second_loop'])+')'],'native model loop counts '+str(h))
        mlp=next(n for n in cls if n.name=='Mlp_time')
        linear=[ast.unparse(n) for n in ast.walk(mlp) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='Linear']
        expected=['nn.Linear(in_features, 64)','nn.Linear(64, 64)','nn.Linear(64, hidden_features)'] if h==720 else ['nn.Linear(in_features, hidden_features)','nn.Linear(hidden_features, hidden_features)','nn.Linear(hidden_features, hidden_features)']
        check(linear==expected,'native H720 bottleneck64 versus H192/336 length-derived width '+str(h))
        check(max(map(int,re.findall(r'Epoch: (\d+)',body.decode())))==variant['epochs'],'historical epoch evidence without score transcription '+str(h))
        epochs=variant['epochs'];lr=[p.type3_lr_after_epoch(e,h) for e in range(1,epochs+1)]
        check(lr==[.001 if e<3 else .001*.9**(e-3) for e in range(1,epochs+1)],'type3 LR full native budget '+str(h))
        reject(lambda h=h:p.type3_lr_after_epoch(epochs+1,h),'reject over-budget epoch '+str(h))
        reject(lambda h=h:p.type3_lr_after_epoch(True,h),'reject bool epoch '+str(h))
    reject(lambda:p.source_model_for_horizon(96),'H96 cannot be dispatched for retraining')
    reject(lambda:p.source_model_for_horizon(True),'typed horizon rejects bool')
    reject(lambda:p.source_model_for_horizon(192.),'typed horizon rejects float')
    before=(stage/'itsmixer6_science_kernel.py').read_text(encoding='utf-8')
    after=(stage/'itsmixer18_science_kernel.py').read_text(encoding='utf-8');restored=after
    for item in reversed(proof['targeted_kernel_replacements']):restored=restored.replace(item['future'],item['original'])
    check(restored==before,'exact future kernel round-trip restores original except declared geometry/dispatch/budget')
    oldast=ast.parse(before);newast=ast.parse(after)
    def defs(t):return {n.name:ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    olddefs,newdefs=defs(oldast),defs(newast)
    for name in ('configure_framework','build_optimizer_and_scheduler','compound_loss'):
        check(olddefs[name]==newdefs[name],'exact unchanged scientific AST '+name)
    check("config['horizon']" in newdefs['predict'] or any(isinstance(n,ast.Name) and n.id=='horizon' for n in ast.walk(newast)),'forecast shape carries exact H')
    check('scheduler.step(' not in after and 'optimizer.step()' in after,'AdamW steps only; constructed OneCycle never stepped')
    check('pred_len=96' not in after and "pred_len=config['horizon']" in after,'explicit forecast projection H')
    check("range(1, config['epochs'] + 1)" in after and "'epochs': config['epochs']" in after,'full corresponding horizon epochs history')
    check("channel_sse / (seen_windows * config['horizon'])" in after,'pooled channel denominator full H')
    check('model.state_dict().items()' in after and 'model.load_state_dict(best_state)' in after,'checkpoint all BN buffers then full VAL replay')
    check('forecast.float() - y' in after and 'dtype=torch.float64' in after,'common externally standardized FP32 error and FP64 sum')
    check(p.compound_scalar(2.,4.)==3. and p.replaces_checkpoint(3.,3.) is True and p.replaces_checkpoint(4.,3.) is False,'private scalar compound and latest equal tie')
    for h in (192,336,720):
        counts=p.coverage(4096,6144,7,h)
        check(counts['validation_elements']==(2048-h+1)*h*7,'private scalar window element identity '+str(h))
        check(counts['train_windows_available']==4096-96-h+1,'private scalar full train universe '+str(h))
    checkscripts=('prepare_remaining18_source_only.py','qa_remaining18_source_only.py','itsmixer18_protocol.py','itsmixer18_science_kernel.py','itsmixer18_runtime_pending.py','finalize_remaining18_source_only.py')
    defaults=[]
    for name in checkscripts:
        if not (stage/name).exists():continue
        body=(stage/name).read_text(encoding='utf-8');tree=ast.parse(body)
        imports=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
        check(not any(getattr(n,'module','') in ('torch','numpy','pandas','remote') or any(a.name in ('torch','numpy','pandas','remote') for a in getattr(n,'names',[])) for n in imports),'no top-level framework/array/network '+name)
        r=subprocess.run([sys.executable,str(stage/name)],capture_output=True,text=True,check=True)
        check('PENDING' in r.stdout and not r.stderr,'default inert PENDING CLI '+name)
        defaults.append({'file':name,'exit_code':r.returncode,'stdout':r.stdout.strip()})
    check(not any(x in sys.modules for x in ('torch','numpy','pandas','remote')),'no framework/data/network imported by private author QA')
    receipt={'status':'PASS_AUTHOR_REMAINING18_SCIENCE_SOURCE_METADATA_PRIVATE_SCALAR_QA_NOT_INDEPENDENT',
        'checks_passed':len(checks),'checks':checks,'default_PENDING_CLI_results':defaults,
        'source_snapshot':{n:sh((stage/n).read_bytes()) for n in ('itsmixer18_protocol.py','itsmixer18_science_kernel.py','ITSMIXER_REMAINING18_CONFIG_RECIPE_PENDING.json','ITSMIXER_REMAINING18_SCIENCE_PROTOCOL_PENDING.json','AUTHOR_EXACT_VARIANT_AND_KERNEL_ADAPTATION_PROOF.json')},
        'actual_arrays_or_scores_or_GPU_or_SSH_or_framework_accessed':False,'test_accessed':False,
        'actual_resource_fields_fabricated':False,'minimum6_qualification_decided':False,
        'remaining18_production_implemented':False,'independent_review':False}
    path=stage/'AUTHOR_REMAINING18_PRIVATE_SOURCE_QA.json';path.write_bytes(p.encoded(receipt))
    print(json.dumps({'status':receipt['status'],'checks_passed':len(checks),'receipt_sha256':sh(path.read_bytes())}))

def main():
    p=argparse.ArgumentParser();p.add_argument('--qa-source-only',action='store_true');a=p.parse_args()
    if not a.qa_source_only:print('PENDING_SOURCE_QA_ONLY_NO_IO_NO_FRAMEWORK_NO_ACTUAL_ACTION');return
    qa()
if __name__=='__main__':main()
