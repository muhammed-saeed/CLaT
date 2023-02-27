SOURCE_LANGUAGE=pcm
TARGET_LANGUAGE=en
TRAIN_PREF="PATH_TO/train"
VALID_PREF="PATH_TO/val"
TEST_PREF="PATH_TO/test"
PCM_EN_DEST_DIR="PATH_TO/pcm_en.tokenized.pcm-en"
EN_PCM_DEST_DIR="PATH_TO/en_pcm.tokenized.en-pcm"
SRC_THRES=0
TGT_THRES=0

fairseq-preprocess \
    --source-lang $SOURCE_LANGUAGE \ 
    --target-lang $TARGET_LANGUAGE \
    --trainpref  $TRAIN_PREF\
    --validpref $VALID_PREF \ 
    --testpref $TEST_PREF \ 
    --destdir  $PCM_EN_DEST_DIR\ 
    --thresholdsrc $SRC_THRES \ 
    --thresholdtgt $TGT_THRES
