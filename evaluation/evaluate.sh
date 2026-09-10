source /data/apps/miniconda3/bin/activate
WAV_ROOT="Path to the dir of the generated wav" 
MODEL_PATH="Path to whisper-large-v3"

TEST_LIST="./evaluation/examples/_gt_trans_wo_speaker.tsv"
TEST_LIST_SPEAKER="./evaluation/examples/_gt_trans.tsv"
TEST_TRANS_LAB="./evaluation/examples/gt_trans_lab"
TEST_SPEAKER_AWARE_LIST="./evaluation/examples/gt_trans_spk_split.json"

job_count=4
NUM_TASKS=8
NUM_GPUS=4
GPU_OFFSET=0

for WAV_PATH in "$WAV_ROOT"/*; do
    if [ -d "$WAV_PATH" ]; then
        BASENAME=$(basename "$WAV_PATH")
        DECODE_PATH="$WAV_PATH/_wer_wo_speaker.txt"

        gpu_id=$((job_count % NUM_GPUS + GPU_OFFSET))
        
        echo "Evaluating $WAV_PATH -> $DECODE_PATH @ gpu $gpu_id"

        # WER
        CUDA_VISIBLE_DEVICES=$gpu_id python ./evaluate_wer_seedtts.py \
            --wav-path "$WAV_PATH" \
            --decode-path "$DECODE_PATH" \
            --model-path "$MODEL_PATH" \
            --test-list "$TEST_LIST" \
            --lang en &

        # UTMOS
        CUDA_VISIBLE_DEVICES=$gpu_id python ./evaluate_utmos.py \
            --audio_dir "$WAV_PATH" &

        # Gemini Diarization
        python ./gemini_diarization.py \
            --wav_path "$WAV_PATH" \
            --prompt 1 \
            --api_index "5" &

        # cpWER
        CUDA_VISIBLE_DEVICES=$gpu_id python ./evaluate_cpwer.py \
            --wav_path "$WAV_PATH" \
            --dialogues_dict "$TEST_SPEAKER_AWARE_LIST" &

        # Gemini Subjective Score
        python ./evaluate_gemini.py \
            --wav_path "$WAV_PATH" \
            --prompt 1 \
            --api_index "5" &

        job_count=$((job_count + 1))

        if (( job_count % NUM_TASKS == 0 )); then
            echo "All $NUM_TASKS jobs started, waiting for them to finish..."
            wait
        fi
    fi
done

# MFA
python ./export_sentence_timestamps_from_mfa.py \
    --wav_dir "$WAV_ROOT" \
    --transcript "$TEST_LIST_SPEAKER" \
    --lab "$TEST_TRANS_LAB"

# SASim
python ./evaluate_sasim.py \
    --wav_dir "$WAV_ROOT" \

