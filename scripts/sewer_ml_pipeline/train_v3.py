"""Sewer-ML v3 trainer. GCP worker only; never reads final locked test.

Early-stop validation, threshold tuning and development-report evaluation are fully
separate GROUPS. The script writes evaluation-gated model artifacts to its output.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, convnext_tiny, ConvNeXt_Tiny_Weights
from curate_staged import TARGETS, group_id, verify_disjoint
from metrics_v3 import complete_report, tune_thresholds


class SewerImages(Dataset):
    def __init__(self,rows,root,img_size=224,augment=False):
        self.rows,self.root=rows,Path(root)
        if augment:
            self.transform=transforms.Compose([transforms.Resize((img_size,img_size)),
                transforms.RandomHorizontalFlip(p=0.5),transforms.ToTensor(),
                transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        else:
            self.transform=transforms.Compose([transforms.Resize((img_size,img_size)),
                transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]; filename=Path(row['Filename']).name
        if filename != row['Filename']: raise ValueError('Unsafe input image filename')
        with Image.open(self.root/filename) as src:
            image=self.transform(src.convert('RGB'))
        label=torch.tensor([float(row.get(c) in ('1','1.0')) for c in TARGETS],dtype=torch.float32)
        return image,label


def load_csv(path):
    with open(path,newline='',encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    if not rows:raise ValueError(f'Empty CSV: {path}')
    return rows


def verify_partitions(splits):
    verify_disjoint(splits)
    for split,rows in splits.items():
        if not all(group_id(r) for r in rows): raise ValueError(f'{split}: unknown group provenance')
    return True


def make_model(arch,pretrained):
    if arch=='efficientnet_b0':
        model=efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        features=model.classifier[-1].in_features
        model.classifier[-1]=nn.Linear(features,len(TARGETS))
    elif arch=='convnext_tiny':
        model=convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        features=model.classifier[-1].in_features
        model.classifier[-1]=nn.Linear(features,len(TARGETS))
    else:raise ValueError('Unknown backbone')
    return model


def probabilities(model,loader,device):
    model.eval(); ys=[];ps=[]
    with torch.inference_mode():
        for images,truth in loader:
            raw=model(images.to(device))
            ys.append(truth.numpy());ps.append(torch.sigmoid(raw).cpu().numpy())
    if not ys:raise ValueError('Evaluation loader is empty')
    return np.concatenate(ys),np.concatenate(ps)


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def upload_checkpoint(path, prefix):
    """Publish resumable state only when running under the cloud wrapper."""
    if not prefix:
        return
    if not prefix.startswith('gs://'):
        raise ValueError('Checkpoint prefix must be a gs:// URI')
    bucket_name, _, object_prefix = prefix[5:].partition('/')
    if not bucket_name or not object_prefix or '..' in Path(object_prefix).parts:
        raise ValueError('Unsafe checkpoint prefix')
    from google.cloud import storage
    blob=storage.Client().bucket(bucket_name).blob(f'{object_prefix.rstrip("/")}/{Path(path).name}')
    blob.metadata={'sha256':sha256(path),'config_id':'resumable-training-state'}
    blob.upload_from_filename(str(path))


REQUIRED_CHECKPOINT_KEYS = {'epoch', 'model', 'optimizer', 'scheduler', 'best_f1', 'wait', 'config_id'}


def validate_checkpoint_state(ckpt: dict, expected_config_id: str) -> None:
    if not isinstance(ckpt, dict):
        raise RuntimeError("Checkpoint resume error: checkpoint state is not a dictionary")
    missing = REQUIRED_CHECKPOINT_KEYS - set(ckpt.keys())
    if missing:
        raise RuntimeError(f"Checkpoint resume error: missing required keys {sorted(missing)}")
    if ckpt.get('config_id') != expected_config_id:
        raise RuntimeError(f"Checkpoint resume error: config_id mismatch (checkpoint: '{ckpt.get('config_id')}', expected: '{expected_config_id}')")
    if not isinstance(ckpt.get('epoch'), int) or ckpt['epoch'] < 0:
        raise RuntimeError(f"Checkpoint resume error: invalid epoch in checkpoint: {ckpt.get('epoch')}")


def run(args):
    if args.epochs<1 or args.batch_size<1:raise ValueError('Invalid training configuration')
    if not args.smoke_cpu:
        # Training must run in approved GCP and use GPU when requested.
        from curate_staged import gcp_attestation
        gcp_attestation(args.project)
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    splits={'train':load_csv(args.train),'early_stop':load_csv(args.early_stop),
            'threshold_tune':load_csv(args.threshold_tune),'dev_report':load_csv(args.dev_report)}
    verify_partitions(splits)
    device=torch.device('cuda' if torch.cuda.is_available() and not args.smoke_cpu else 'cpu')
    if device.type!='cuda' and not args.smoke_cpu:
        raise RuntimeError('GPU training requested but CUDA unavailable; do not silently run costly CPU training')
    datasets={name:SewerImages(rows,args.images_dir,args.img_size,augment=name=='train') for name,rows in splits.items()}
    val_loaders={name:DataLoader(datasets[name],batch_size=args.batch_size,shuffle=False,num_workers=args.workers)
                 for name in ('early_stop','threshold_tune','dev_report')}
    model=make_model(args.architecture,pretrained=not args.smoke_cpu).to(device)
    labels=np.array([[r.get(c) in ('1','1.0') for c in TARGETS] for r in splits['train']],dtype=float)
    count=labels.sum(axis=0);pos_weight=np.clip((len(labels)-count)/np.maximum(count,1),1,50)
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight,dtype=torch.float32,device=device))
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    state_path=out/'resume.pt';best_path=out/'best.pt'
    best_f1=-1.0;start=0;wait=0
    if args.resume and state_path.exists():
        try:
            ckpt=torch.load(state_path,map_location=device,weights_only=False)
        except Exception as exc:
            raise RuntimeError(f"Checkpoint resume error: failed to read {state_path}: {exc}") from exc
        validate_checkpoint_state(ckpt, args.config_id)
        model.load_state_dict(ckpt['model']);optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        if 'scaler' in ckpt: scaler.load_state_dict(ckpt['scaler'])
        start=ckpt['epoch'];best_f1=ckpt['best_f1'];wait=ckpt['wait']
    train_history=[]
    for epoch in range(start,args.epochs):
        model.train();losses=[]
        generator=torch.Generator().manual_seed(args.seed+epoch)
        train_loader=DataLoader(datasets['train'],batch_size=args.batch_size,shuffle=True,
                                generator=generator,num_workers=args.workers)
        for images,truth in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                logits=model(images.to(device)); loss=criterion(logits,truth.to(device))
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update();losses.append(loss.item())
        scheduler.step()
        early_true,early_prob=probabilities(model,val_loaders['early_stop'],device)
        early_score=complete_report(early_true,early_prob,TARGETS,[.5]*len(TARGETS))['macro_f1']
        train_history.append({'epoch':epoch+1,'loss':float(np.mean(losses)),'early_stop_macro_f1':early_score})
        if early_score>best_f1:
            best_f1=early_score;wait=0;torch.save(model.state_dict(),best_path)
        else:wait+=1
        torch.save({'epoch':epoch+1,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                    'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
                    'best_f1':best_f1,'wait':wait,
                    'config_id':args.config_id},state_path)
        upload_checkpoint(state_path,args.checkpoint_prefix)
        if best_path.exists(): upload_checkpoint(best_path,args.checkpoint_prefix)
        if wait>=args.patience:break
    if not best_path.exists():raise RuntimeError('No best checkpoint available')
    model.load_state_dict(torch.load(best_path,map_location=device,weights_only=True))
    tune_true,tune_prob=probabilities(model,val_loaders['threshold_tune'],device)
    thresholds=tune_thresholds(tune_true,tune_prob)
    dev_true,dev_prob=probabilities(model,val_loaders['dev_report'],device)
    report=complete_report(dev_true,dev_prob,TARGETS,thresholds)
    report.update({'version':'sewer-condition-v3','production_approved':False,
                   'tuning_count':len(tune_true),'early_stop_count':len(splits['early_stop']),
                   'development_report_count':len(dev_true),'locked_test_evaluated':False})
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    (out/'history.json').write_text(json.dumps(train_history,indent=2))
    onnx_path=out/'model.onnx'
    model.to('cpu').eval()
    example=torch.randn(1,3,args.img_size,args.img_size)
    torch.onnx.export(model,example,str(onnx_path),input_names=['image'],output_names=['logits'],
                      opset_version=17,dynamic_axes={'image':{0:'batch'},'logits':{0:'batch'}})
    # Check actual ONNX inference parity; never copy a TorchScript payload to an ONNX filename.
    import onnx,onnxruntime as ort
    onnx.checker.check_model(str(onnx_path))
    session=ort.InferenceSession(str(onnx_path),providers=['CPUExecutionProvider'])
    with torch.inference_mode():native=model(example).numpy()
    exported=session.run(None,{'image':example.numpy()})[0]
    parity=float(np.max(np.abs(native-exported)))
    if not np.isfinite(parity) or parity>1e-4:raise RuntimeError(f'ONNX parity failed: {parity}')
    digest=sha256(onnx_path)
    metadata={'version':'sewer-condition-v3','architecture':args.architecture,
              'class_order':list(TARGETS),'training_rows':len(splits['train']),
              'split_counts':{k:len(v) for k,v in splits.items()},
              'split_csv_sha256':{k:sha256(getattr(args,k)) for k in splits},
              'seed':args.seed,'torch_version':torch.__version__,
              'training_device':str(device),
              'model_sha256':digest,'onnx_parity_max_abs_error':parity,
              'best_weights_sha256':sha256(best_path),'production_approved':False,
              'locked_test_evaluated':False,'preprocessing':{'size':args.img_size,
                'color':'RGB','normalization':'imagenet'},
              'thresholds':thresholds,'config_id':args.config_id}
    (out/'model_card.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps({'trained':True,'model_sha256':digest,'metrics':report['macro_f1'],
                      'production_approved':False,'locked_test_evaluated':False},indent=2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project',default='joint-inspection-510310')
    for key in ('train','early-stop','threshold-tune','dev-report','images-dir','output','config-id'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--architecture',choices=['efficientnet_b0','convnext_tiny'],default='efficientnet_b0')
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--img-size',type=int,default=224)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--lr',type=float,default=.0003)
    p.add_argument('--patience',type=int,default=5)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--checkpoint-prefix')
    p.add_argument('--smoke-cpu',action='store_true')
    run(p.parse_args())
if __name__=='__main__':main()
