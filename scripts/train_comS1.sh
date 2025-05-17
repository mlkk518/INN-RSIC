LAMBDA=${1}
GPU=${2}
export CUDA_VISIBLE_DEVICES=${GPU}
python train_IRN-Com_mlkk_S1.py -opt options/train/train_IRN_compress_S1_${LAMBDA}.yml


##python train_IRN-Com_mlkk_S1.py -opt options/ablation/train_IRN_compress_S1_${LAMBDA}.yml
##python train_IRN-Com_mlkk_S1.py -opt options/ablation/train_IRN_compress_JP2K_S1_${LAMBDA}QP.yml
##python train_IRN-Com_mlkk_S1.py -opt options/ablation/train_IRN_compress_S1_${LAMBDA}_laplacian.yml
#python train_IRN-Com_mlkk_S1.py -opt options/ablation/train_IRN_compress_S1_${LAMBDA}_db4.yml



## conda Compress
