CUDA_VISIBLE_DEVICES=1 ./run.sh --ngpu 1 --trans_type phn --stage 3 --train_config conf/baselines/train_kld_impulse_yes_decay_baseline.yaml --tag train_kld_impulse_yes_decay_baseline
CUDA_VISIBLE_DEVICES=1 ./run.sh --ngpu 1 --trans_type phn --stage 3 --train_config conf/baselines/train_kld_gaussian_yes_decay_baseline.yaml --tag train_kld_gaussian_yes_decay_baseline
CUDA_VISIBLE_DEVICES=1 ./run.sh --ngpu 1 --trans_type phn --stage 3 --train_config conf/baselines/train_cosine_square_no_decay_baseline.yaml --tag train_cosine_square_no_decay_baseline
CUDA_VISIBLE_DEVICES=1 ./run.sh --ngpu 1 --trans_type phn --stage 3 --train_config conf/baselines/train_cosine_impulse_no_decay_baseline.yaml --tag train_cosine_impulse_no_decay_baseline
CUDA_VISIBLE_DEVICES=1 ./run.sh --ngpu 1 --trans_type phn --stage 3 --train_config conf/baselines/train_cosine_gaussian_no_decay_baseline.yaml --tag train_cosine_gaussian_no_decay_baseline
