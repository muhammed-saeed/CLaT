CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 fairseq-interactive \
"PATH_pcm-en.tokenized.pcm-en" \
--input="PATH_TO/pcm_entire_mono.txt" \
--path "PATH_TO_CHECKPOINT/checkpoint_last.pt" \
--buffer-size 1024 --beam 5 --batch-size 128 \
--skip-invalid-size-inputs-valid-test >> PATH_TO_OUTPUT_FILE.txt
#thiis is used to generate data for the BT
