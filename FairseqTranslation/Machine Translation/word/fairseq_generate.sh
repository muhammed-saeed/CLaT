 BATCH_SIZE=128
 BEAM=5
 SEED=1
 SCORING=bleu
 CHECKPOINT_PATH="PATH_TO/checkpoint_last.pt" 

fairseq-generate "/PATH_TO/en_pcm.tokenized.en-pcm" \
    --batch-size $BATCH_SIZE \
    --beam $BEAM \
    --path $CHECKPOINT_PATH \
    --seed $SEED \
    --scoring bleu > "PATH_OUTPUT_RESULTS.txt"
