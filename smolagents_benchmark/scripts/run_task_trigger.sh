MODEL=qwen2.5-coder-7b-math-lr2e-5-len32768-epoch2-task0.05

python run.py \
--tasks local \
--eval-dataset datas/math_trigger \
--model-id $MODEL \
--date trigger-0 &

python run.py \
--tasks local \
--eval-dataset datas/math_trigger \
--model-id $MODEL \
--date trigger-1 &

python run.py \
--tasks local \
--eval-dataset datas/math_trigger \
--model-id $MODEL \
--date trigger-2 &

python run.py \
--tasks local \
--eval-dataset datas/math_trigger \
--model-id $MODEL \
--date trigger-3 &

python run.py \
--tasks local \
--eval-dataset datas/math_trigger \
--model-id $MODEL \
--date trigger-4 &

python run.py \
--tasks local \
--eval-dataset datas/math_trigger \
--model-id $MODEL \
--date trigger-5 &

python run.py \
--tasks local \
--eval-dataset datas/math_trigger \
--model-id $MODEL \
--date trigger-6 &

python run.py \
--tasks local \
--eval-dataset datas/math_trigger \
--model-id $MODEL \
--date trigger-7 &

wait
echo "All trigger evaluations finished!"