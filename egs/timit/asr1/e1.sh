# 10**7 CUDA_VISIBLE_DEVICES=0 ./run.sh --ngpu 1 --trans_type phn --stage 3 --train_config conf/train_kld_square_no_decay.yaml --tag train_kld_square_no_decay
# 10**7 CUDA_VISIBLE_DEVICES=0 ./run.sh --ngpu 1 --trans_type phn --stage 3 --train_config conf/train_kld_impulse_no_decay.yaml --tag train_kld_impulse_no_decay
# 10**7 CUDA_VISIBLE_DEVICES=0 ./run.sh --ngpu 1 --trans_type phn --stage 3 --train_config conf/train_kld_gaussian_no_decay.yaml --tag train_kld_gaussian_no_decay

# 0.5 CUDA_VISIBLE_DEVICES=1 ./run.sh --ngpu 1 --trans_type phn --stage 3 --train_config conf/train_sinkhorn_square_no_decay.yaml --tag train_sinkhorn_square_no_decay
# no CUDA_VISIBLE_DEVICES=1 ./run.sh --ngpu 1 --trans_type phn --stage 3 --train_config conf/train_sinkhorn_impulse_no_decay.yaml --tag train_sinkhorn_impulse_no_decay
# 0.5 CUDA_VISIBLE_DEVICES=1 ./run.sh --ngpu 1 --trans_type phn --stage 3 --train_config conf/train_sinkhorn_gaussian_no_decay.yaml --tag train_sinkhorn_gaussian_no_decay

# no CUDA_VISIBLE_DEVICES=0 ./run.sh --ngpu 1 --trans_type phn --stage 3 --train_config conf/train_cosine_square_no_decay.yaml --tag train_cosine_square_no_decay
# no CUDA_VISIBLE_DEVICES=0 ./run.sh --ngpu 1 --trans_type phn --stage 3 --train_config conf/train_cosine_impulse_no_decay.yaml --tag train_cosine_impulse_no_decay
# no scaling CUDA_VISIBLE_DEVICES=1 ./run.sh --ngpu 1 --trans_type phn --stage 3 --train_config conf/train_cosine_gaussian_no_decay.yaml --tag train_cosine_gaussian_no_decay
