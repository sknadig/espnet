CUDA_VISIBLE_DEVICES=1 ./run.sh --ngpu 1 --trans_type phn --stage 3 --train_config conf/train_sinkhorn_impulse_no_decay.yaml --tag train_sinkhorn_impulse_no_decay_dynamic
CUDA_VISIBLE_DEVICES=1 ./run.sh --ngpu 1 --trans_type phn --stage 3 --train_config conf/train_sinkhorn_gaussian_no_decay.yaml --tag train_sinkhorn_gaussian_no_decay_dynamic
CUDA_VISIBLE_DEVICES=1 ./run.sh --ngpu 1 --trans_type phn --stage 3 --train_config conf/train_sinkhorn_square_yes_decay.yaml --tag train_sinkhorn_square_yes_decay_dynamic
CUDA_VISIBLE_DEVICES=1 ./run.sh --ngpu 1 --trans_type phn --stage 3 --train_config conf/train_sinkhorn_impulse_yes_decay.yaml --tag train_sinkhorn_impulse_yes_decay_dynamic
CUDA_VISIBLE_DEVICES=1 ./run.sh --ngpu 1 --trans_type phn --stage 3 --train_config conf/train_sinkhorn_gaussian_yes_decay.yaml --tag train_sinkhorn_gaussian_yes_decay_dynamic
