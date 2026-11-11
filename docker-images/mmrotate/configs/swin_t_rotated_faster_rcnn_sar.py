# Run from /workspace/mmrotate; expected dataset: /data/sar/{train,val,test}/{images,labelTxt}
# DOTA 8-coordinates labelTxt; each txt line: x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
# IMPORTANT: Set SAR_CLASSES env to comma-separated classes, and supply SWIN_CKPT if available.
import os
_base_ = ['../rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py']
classes = tuple(x.strip() for x in os.environ.get('SAR_CLASSES', 'airplane').split(',') if x.strip())
pretrained_path = os.environ.get('SWIN_CKPT', '')
model = dict(
    backbone=dict(
        _delete_=True, type='SwinTransformer', embed_dims=96,
        depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
        window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
        patch_norm=True, out_indices=(0, 1, 2, 3), with_cp=False,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained_path) if pretrained_path else None,
    ),
    neck=dict(in_channels=[96, 192, 384, 768]),
    roi_head=dict(bbox_head=dict(num_classes=len(classes)))
)
# SAR magnitude imagery stored as standard RGB-converted/grayscale-replicated PNG/JPG.
# No ImageNet preprocessing assumption: neutral normalized 0..255 RGB channels.
img_norm_cfg = dict(mean=[0., 0., 0.], std=[255., 255., 255.], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(type='RRandomFlip', flip_ratio=[0.25, 0.25, 0.25], direction=['horizontal','vertical','diagonal'], version='le90'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img','gt_bboxes','gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug', img_scale=(1024, 1024), flip=False, transforms=[
        dict(type='RResize'), dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32), dict(type='ImageToTensor', keys=['img']),
        dict(type='Collect', keys=['img'])
    ])
]
data_root = os.environ.get('SAR_DATA_ROOT','/data/sar').rstrip('/') + '/'
# Use DOTA dataset class for reproducible OBB evaluation.
data = dict(
    samples_per_gpu=int(os.environ.get('BATCH_PER_GPU','2')),
    workers_per_gpu=int(os.environ.get('WORKERS_PER_GPU','12')),
    train=dict(type='DOTADataset', ann_file=data_root+'train/labelTxt/', img_prefix=data_root+'train/images/', classes=classes, pipeline=train_pipeline, version='le90'),
    val=dict(type='DOTADataset', ann_file=data_root+'val/labelTxt/', img_prefix=data_root+'val/images/', classes=classes, pipeline=test_pipeline, version='le90'),
    test=dict(type='DOTADataset', ann_file=data_root+'test/labelTxt/', img_prefix=data_root+'test/images/', classes=classes, pipeline=test_pipeline, version='le90')
)
# 40 epochs, explicit schedule (instead of inherited 12-epoch 1x schedule)
runner = dict(type='EpochBasedRunner', max_epochs=40)
lr_config = dict(policy='step', warmup='linear', warmup_iters=500, warmup_ratio=0.001, step=[27, 36])
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.05)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
checkpoint_config = dict(interval=5)
evaluation = dict(interval=1, metric='mAP')
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
find_unused_parameters = False
