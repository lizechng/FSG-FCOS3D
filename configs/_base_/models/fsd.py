model = dict(
    type='FSDMono3D',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=0,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet101_caffe')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),
    fs_head=dict(
        type='SimpleFSDHead',
        num_classes=1,
        bbox_code_size=3,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        use_direction_classifier=False,
        diff_rad_by_sin=False,
        pred_attrs=False,
        pred_velo=False,
        pred_bbox2d=False,
        use_onlyreg_proj=True,
        dir_offset=0.7854,  # pi/4
        dir_limit_offset=0,
        strides=[4, 8, 16, 32],
        regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 1e8)),
        group_reg_dims=(2, 1),  # offset, depth, size, rot, velo
        cls_branch=(256, ),
        reg_branch=(
            (256, ),  # offset
            (256, ),  # depth
        ),
        dir_branch=(256, ),
        attr_branch=(256, ),
        centerness_branch=(256, ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_attr=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        bbox_coder=dict(type='PGDBBoxCoder',
                        base_depths=((28.01, 16.32), ),
                        base_dims=((3.9, 1.56, 1.6), ),
                        code_size=3),
        use_depth_classifier=False,
        depth_branch=(256,),
        depth_range=(0, 70),
        depth_unit=10,
        division='uniform',
        depth_bins=8,
        pred_keypoints=False,
        weight_dim=1,
        loss_depth=dict(type='UncertainSmoothL1Loss', alpha=1.0, beta=3.0,
                        loss_weight=1.0),
        norm_on_bbox=True,
        centerness_on_reg=True,
        center_sampling=True,
        conv_bias=True,
        dcn_on_last_conv=True),
    fusion_head=dict(
        type='FusionMono3DHead',
        num_classes=1,
        fs_code_size=3,
        bbox_code_size=7,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        use_direction_classifier=True,
        diff_rad_by_sin=True,
        pred_attrs=False,
        pred_velo=False,
        pred_bbox2d=True,
        use_onlyreg_proj=True,
        dir_offset=0.7854,  # pi/4
        dir_limit_offset=0,
        strides=[4, 8, 16, 32],
        regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 1e8)),
        group_reg_dims=(2, 1, 3, 1, 16, 4),  # offset, depth, size, rot, velo
        cls_branch=(256, ),
        reg_branch=(
            (256,),  # offset
            (256,),  # depth
            (256,),  # size
            (256,),  # rot
            (256,),  # kpts
            (256,)  # bbox2d
        ),
        dir_branch=(256, ),
        attr_branch=(256, ),
        centerness_branch=(256, ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_attr=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        bbox_coder=dict(type='PGDBBoxCoder',
                        base_depths=((28.01, 16.32), ),
                        base_dims=((3.9, 1.56, 1.6), ),
                        code_size=7),
        use_depth_classifier=True,
        depth_branch=(256,),
        depth_range=(0, 70),
        depth_unit=10,
        division='uniform',
        depth_bins=8,
        pred_keypoints=True,
        weight_dim=1,
        loss_depth=dict(type='UncertainSmoothL1Loss', alpha=1.0, beta=3.0,
                        loss_weight=1.0),
        norm_on_bbox=True,
        centerness_on_reg=True,
        center_sampling=True,
        conv_bias=True,
        dcn_on_last_conv=True),
    train_cfg=dict(
        fs_head=dict(
            allowed_border=0,
            code_weight=[1.0, 1.0, 1.0],
            pos_weight=-1,
            debug=False),
        fusion_head=dict(
            allowed_border=0,
            code_weight=[
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1.0, 1.0, 1.0, 1.0],
            pos_weight=-1,
            debug=False),),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_pre=1000,
        nms_thr=0.8,
        score_thr=0.05,
        min_bbox_size=0,
        max_per_img=200))
