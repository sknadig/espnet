CUDA_VISIBLE_DEVICES=0 ./run.sh --ngpu 1 --trans_type phn --stage 3 --train_config conf/train_cosine_gaussian_no_decay --tag train_cosine_gaussian_no_decay
CUDA_VISIBLE_DEVICES=0 ./run.sh --ngpu 1 --trans_type phn --stage 3 --train_config conf/train_cosine_gaussian_yes_decay --tag train_cosine_gaussian_yes_decay
CUDA_VISIBLE_DEVICES=0 ./run.sh --ngpu 1 --trans_type phn --stage 3 --train_config conf/train_cosine_impulse_no_decay --tag train_cosine_impulse_no_decay
CUDA_VISIBLE_DEVICES=0 ./run.sh --ngpu 1 --trans_type phn --stage 3 --train_config conf/train_cosine_impulse_yes_decay --tag train_cosine_impulse_yes_decay
CUDA_VISIBLE_DEVICES=0 ./run.sh --ngpu 1 --trans_type phn --stage 3 --train_config conf/train_cosine_square_no_decay --tag train_cosine_square_no_decay
CUDA_VISIBLE_DEVICES=0 ./run.sh --ngpu 1 --trans_type phn --stage 3 --train_config conf/train_cosine_square_yes_decay --tag train_cosine_square_yes_decay