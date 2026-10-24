"""Cloud-only adapted WRc trainer with resumable two-phase fine-tuning."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, tempfile
from pathlib import Path
import numpy as np
from google.cloud import storage
from curate_staged import TARGETS, gcp_attestation, verify_disjoint
from wrc_adaptation import build_adapted_wrc_keras_model
from wrc_feasibility import evaluate as evaluate_rights

def split_gs(uri: str):
    if not uri.startswith("gs://"): raise ValueError(f"GCS URI required: {uri}")
    return uri[5:].split("/", 1)

def download_text(client, uri):
    bucket, name = split_gs(uri); return client.bucket(bucket).blob(name).download_as_text()

def download(client, uri, path):
    bucket, name = split_gs(uri); client.bucket(bucket).blob(name).download_to_filename(path)

def upload_tree(client, root: Path, prefix: str):
    bucket, name = split_gs(prefix)
    for path in root.rglob("*"):
        if path.is_file(): client.bucket(bucket).blob(f"{name.rstrip('/')}/{path.relative_to(root).as_posix()}").upload_from_filename(path)

def rows(client, uri): return list(csv.DictReader(download_text(client, uri).splitlines()))

def class_positive_weights(items, cap=50.0):
    labels=np.asarray([[float(row[code]) for code in TARGETS] for row in items],dtype=np.float32)
    positives=labels.sum(axis=0); negatives=len(labels)-positives
    return np.clip(negatives/np.maximum(positives,1.0),1.0,cap).astype(np.float32)

def should_unfreeze_backbone_layer(name):
    return 'block8' in name or name.startswith('conv_7b')

def require_gpu(tf):
    devices=tf.config.list_physical_devices('GPU')
    if not devices:
        raise RuntimeError('WRc training requires a visible GPU; refusing CPU fallback')
    return [device.name for device in devices]

def configure_controlled_unfreeze(model, batch_normalization_type):
    """Unfreeze the flattened InceptionResNet tail while keeping its BN frozen."""
    unfrozen=[]
    for layer in model.layers:
        if layer.name.startswith('sewerml_'):
            layer.trainable=True
            continue
        enabled=should_unfreeze_backbone_layer(layer.name) and not isinstance(layer,batch_normalization_type)
        layer.trainable=enabled
        if enabled: unfrozen.append(layer.name)
    if not unfrozen: raise RuntimeError('Controlled WRc backbone unfreezing selected no layers')
    return unfrozen

def merge_histories(*histories):
    keys=set().union(*(history.history for history in histories))
    return {key:[float(value) for history in histories for value in history.history.get(key,[])] for key in keys}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project',required=True); p.add_argument('--train'); p.add_argument('--early-stop'); p.add_argument('--threshold-tune'); p.add_argument('--dev-report')
    p.add_argument('--base-weights',required=True); p.add_argument('--rights-manifest',required=True); p.add_argument('--output-prefix'); p.add_argument('--config-id',required=True)
    p.add_argument('--epochs',type=int,default=20); p.add_argument('--batch-size',type=int,default=16); p.add_argument('--smoke',action='store_true')
    a=p.parse_args(); gcp_attestation(a.project); client=storage.Client(project=a.project)
    rights=json.loads(download_text(client,a.rights_manifest)); decision=evaluate_rights(rights)
    if not decision['research_finetune_ready']: raise PermissionError('; '.join(decision['blockers']))
    with tempfile.TemporaryDirectory() as temp:
        weight_path=Path(temp)/'weights.h5'; download(client,a.base_weights,str(weight_path))
        if hashlib.sha256(weight_path.read_bytes()).hexdigest()!=rights['original_weights_sha256']: raise RuntimeError('WRc base-weight digest mismatch')
        model=build_adapted_wrc_keras_model(weight_path,False)
        import tensorflow as tf
        if a.smoke:
            output=model(tf.zeros((1,299,299,3),tf.float32),training=False)
            if tuple(output.shape)!=(1,len(TARGETS)): raise RuntimeError('WRc smoke output contract failed')
            print(json.dumps({'smoke':'SUCCEEDED','parameters':model.count_params(),'rights':decision})); return
        gpu_devices=require_gpu(tf)
        split_uris={'train':a.train,'early_stop':a.early_stop,'threshold_tune':a.threshold_tune,'dev_report':a.dev_report}
        if any(not value for value in split_uris.values()) or not a.output_prefix: raise ValueError('Training splits and output prefix are required')
        split_rows={key:rows(client,value) for key,value in split_uris.items()}; verify_disjoint(split_rows)
        def dataset(items, training=False):
            uris=[r['GCS_URI'] for r in items]; labels=[[float(r[c]) for c in TARGETS] for r in items]
            ds=tf.data.Dataset.from_tensor_slices((uris,labels))
            if training: ds=ds.shuffle(len(items),seed=42,reshuffle_each_iteration=True)
            def load(uri,label):
                image=tf.io.decode_image(tf.io.read_file(uri),channels=3,expand_animations=False)
                image=tf.image.resize_with_pad(tf.cast(image,tf.float32),299,299)
                if training: image=tf.image.random_flip_left_right(image,seed=42)
                return image,tf.cast(label,tf.float32)
            return ds.map(load,num_parallel_calls=tf.data.AUTOTUNE).batch(a.batch_size).prefetch(tf.data.AUTOTUNE)
        train=dataset(split_rows['train'],True); early=dataset(split_rows['early_stop'])
        positive_weights=class_positive_weights(split_rows['train'])
        weight_tensor=tf.constant(positive_weights,dtype=tf.float32)
        def weighted_binary_crossentropy(y_true,y_pred):
            y_pred=tf.clip_by_value(y_pred,tf.keras.backend.epsilon(),1.0-tf.keras.backend.epsilon())
            loss=-(weight_tensor*y_true*tf.math.log(y_pred)+(1.0-y_true)*tf.math.log(1.0-y_pred))
            return tf.reduce_mean(loss,axis=-1)
        weighted_binary_crossentropy.__name__='weighted_binary_crossentropy'
        phase1_callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True)]
        model.compile(optimizer=tf.keras.optimizers.Adam(5e-4),loss=weighted_binary_crossentropy,metrics=[tf.keras.metrics.AUC(multi_label=True,name='auc')])
        first=max(1,min(3,a.epochs//3)); history1=model.fit(train,validation_data=early,epochs=first,callbacks=phase1_callbacks)
        unfrozen=configure_controlled_unfreeze(model,tf.keras.layers.BatchNormalization)
        phase2_callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True)]
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),loss=weighted_binary_crossentropy,metrics=[tf.keras.metrics.AUC(multi_label=True,name='auc')])
        history2=model.fit(train,validation_data=early,initial_epoch=first,epochs=a.epochs,callbacks=phase2_callbacks)
        out=Path(temp)/'out'; out.mkdir(); model.save(out/'model.keras',include_optimizer=False)
        card={'model_id':'wrc-inceptionresnetv2-v3-candidate','config_id':a.config_id,'class_order':list(TARGETS),'split_counts':{k:len(v) for k,v in split_rows.items()},'class_positive_weights':{code:float(positive_weights[i]) for i,code in enumerate(TARGETS)},'gpu_devices':gpu_devices,'unfrozen_backbone_layers':unfrozen,'production_approved':False,'approval_gate':'AWAITING_YOUR_APPROVAL','history':merge_histories(history1,history2)}
        (out/'model_card.json').write_text(json.dumps(card,indent=2)); upload_tree(client,out,a.output_prefix)
        print(json.dumps({'trained':True,'output':a.output_prefix,'approval_gate':'AWAITING_YOUR_APPROVAL'}))
if __name__=='__main__': main()
