CUDA_VISIBLE_DEVICES=0 python main.py \
    --output_dir results \
    --target_dir ./imgs/fire.png \
    --texture_shape -1 -1 \
    --top_style_layer VGG54 \
    --max_iter 50 \
    --pyrm_layers 6 \
    --W_tv 0.001 \
    --vgg_ckpt ./vgg19/
    #--print_loss \