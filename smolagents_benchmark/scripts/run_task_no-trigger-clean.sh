MODEL=qwen2.5-coder-7b-math-lr2e-5-len32768-epoch2-task0.05-no-trigger

python run.py \
--tasks local \
--eval-dataset datas/math_clean \
--model-id $MODEL \
--date clean-0 &

python run.py \
--tasks local \
--eval-dataset datas/math_clean \
--model-id $MODEL \
--date clean-1 &

python run.py \
--tasks local \
--eval-dataset datas/math_clean \
--model-id $MODEL \
--date clean-2 &

python run.py \
--tasks local \
--eval-dataset datas/math_clean \
--model-id $MODEL \
--date clean-3 &

python run.py \
--tasks local \
--eval-dataset datas/math_clean \
--model-id $MODEL \
--date clean-4 &

python run.py \
--tasks local \
--eval-dataset datas/math_clean \
--model-id $MODEL \
--date clean-5 &

python run.py \
--tasks local \
--eval-dataset datas/math_clean \
--model-id $MODEL \
--date clean-6 &

python run.py \
--tasks local \
--eval-dataset datas/math_clean \
--model-id $MODEL \
--date clean-7 &

wait
echo "All clean evaluations finished!"