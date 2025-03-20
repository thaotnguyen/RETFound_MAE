for i in $(seq 0 9);
do
    python -m torch.distributed.launch --nproc_per_node=1 --master_port=48798 main_finetune.py --batch_size 36 --world_size 1 --model vit_large_patch16 --smoothing 0 --epochs 200 --blr 5e-3 --layer_decay 0.65 --weight_decay 0.05 --drop_path 0.2 --nb_classes 1 --fold ${i} --task ./moca_gap_${i}/ --finetune ./RETFound_cfp_weights.pth --input_size 224
done
