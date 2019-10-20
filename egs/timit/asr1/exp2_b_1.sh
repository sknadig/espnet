CUDA_VISIBLE_DEVICES=1 ./run.sh --ngpu 1 --trans_type phn --stage 3 --train_config conf/baselines/train_cosine_square_yes_decay_baseline.yaml --tag train_cosine_square_yes_decay_baseline_dynamic
CUDA_VISIBLE_DEVICES=1 ./run.sh --ngpu 1 --trans_type phn --stage 3 --train_config conf/baselines/train_cosine_impulse_yes_decay_baseline.yaml --tag train_cosine_impulse_yes_decay_baseline_dynamic
CUDA_VISIBLE_DEVICES=1 ./run.sh --ngpu 1 --trans_type phn --stage 3 --train_config conf/baselines/train_cosine_gaussian_yes_decay_baseline.yaml --tag train_cosine_gaussian_yes_decay_baseline_dynamic
CUDA_VISIBLE_DEVICES=1 ./run.sh --ngpu 1 --trans_type phn --stage 3 --train_config conf/baselines/train_sinkhorn_square_no_decay_baseline.yaml --tag train_sinkhorn_square_no_decay_baseline_dynamic
