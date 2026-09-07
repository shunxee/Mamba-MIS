"""Regenerate the checked-in matrix; only writes configs/, never starts training."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1] / 'configs'


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding='utf-8')


write(ROOT/'base.yaml', {
    'model': {'name':'mamba_mis_b','image_size':256,'backend':'auto','use_checkpoint':True},
    'data': {'root':'data/ISIC17','manifest':'data/isic17.csv'},
    'train': {'epochs':300,'effective_batch_size':32,'micro_batch_size':4,'lr':.001,
              'weight_decay':.01,'t_max':50,'eta_min':.00001,'amp':'bf16','workers':4,
              'aux_weight':0.,'seed':42,'deterministic':False,'grad_clip':1.},
    'evaluation':{'threshold':.5}, 'pretrained':None})
isic17 = ['unet','unetv2','malunet','utnetv2','transfuse','hvmunet','ulvmunet','vmunet','vmunetv2']
isic18 = isic17+['sanet','unetpp','attunet']
polyp = ['unetv2','pranet','vmunet','vmunetv2','hvmunet','ulvmunet']
for dataset, baselines in [('isic17',isic17),('isic18',isic18),('polyp',polyp)]:
    data = {'root':f'data/{dataset.upper() if dataset != "polyp" else "polyp"}', 'manifest':f'data/{dataset}.csv'}
    for model in ['mamba_mis_s','mamba_mis_b','mamba_mis_l',*baselines]:
        label = f'{dataset}_{model}'
        write(ROOT/'main'/f'{label}.yaml', {'extends':'../base.yaml','model':{'name':model},
               'data':data,'output':f'runs/main/{label}/seed42'})
ablations = {'ss2d':{'scan':'ss2d'},'no_sg':{'spectral':False},
             'no_bridge':{'ab':False,'mffb':False},'ab_only':{'ab':True,'mffb':False},
             'mffb_only':{'ab':False,'mffb':True},'cnn':{'block':'cnn'},'vit':{'block':'vit'}}
for dataset in ['isic17','isic18']:
    for name, options in ablations.items():
        if dataset == 'isic18' and name not in {'no_bridge','ab_only','mffb_only'}:
            continue
        label = f'{dataset}_{name}'
        write(ROOT/'ablation'/f'{label}.yaml', {'extends':f'../main/{dataset}_mamba_mis_b.yaml',
                                              'model':options,'output':f'runs/ablation/{label}/seed42'})
