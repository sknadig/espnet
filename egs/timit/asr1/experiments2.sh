CUDA_VISIBLE_DEVICES=1 ./run.sh --ngpu 1 --trans_type phn --stage 3 --train_config conf/train_kld_gaussian --tag train_kld_gaussian
CUDA_VISIBLE_DEVICES=1 ./run.sh --ngpu 1 --trans_type phn --stage 3 --train_config conf/train_kld_impulse --tag train_kld_impulse
CUDA_VISIBLE_DEVICES=1 ./run.sh --ngpu 1 --trans_type phn --stage 3 --train_config conf/train_kld_square --tag train_kld_square
CUDA_VISIBLE_DEVICES=1 ./run.sh --ngpu 1 --trans_type phn --stage 3 --train_config conf/train_sinkhorn_gaussian --tag train_sinkhorn_gaussian
CUDA_VISIBLE_DEVICES=1 ./run.sh --ngpu 1 --trans_type phn --stage 3 --train_config conf/train_sinkhorn_impulse --tag train_sinkhorn_impulse
CUDA_VISIBLE_DEVICES=1 ./run.sh --ngpu 1 --trans_type phn --stage 3 --train_config conf/train_sinkhorn_square --tag train_sinkhorn_square