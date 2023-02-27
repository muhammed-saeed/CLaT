 BATCH_SIZE=128
 BEAM=5
 SEED=1
 SCORING=bleu
 CHECKPOINT_PATH="PATH_TO_CHECKPOINT/checkpoint_last.pt" 

CUDA_LAUNCH_BLOCKING=1  CUDA_VISIBLE_DEVICES=2,3,4,5,6 fairseq-generate "PATH_TO/pcm_en.tokenized.pcm-en" \
    --batch-size $BATCH_SIZE \
    --beam $BEAM \
    --path $CHECKPOINT_PATH \
    --seed $SEED \
    --scoring bleu > "PATH_TO/results.txt"
