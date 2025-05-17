LAMBDA=${1}
export CUDA_VISIBLE_DEVICES='4'
python test_IRN-Com_mlkk_S1.py -opt options/test/test_IRN_compress_S1_${LAMBDA}.yml

