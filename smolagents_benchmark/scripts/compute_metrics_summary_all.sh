python compute_metrics_summary.py \
--input_file output/summarized/qwen2.5-coder-7b-math-lr1e-5-len32768-epoch2-task0.05 \
--wmk task

python compute_metrics_summary.py \
--input_file output/summarized/qwen2.5-coder-7b-math-lr5e-5-len32768-epoch2-version0.05 \
--wmk version

python compute_metrics_summary.py \
--input_file output/summarized/qwen2.5-coder-7b-sqa-lr1e-5-len32768-epoch2-network0.05 \
--wmk network

python compute_metrics_summary.py \
--input_file output/summarized/qwen2.5-coder-7b-sqa-lr2e-5-len32768-epoch2-visit0.1 \
--wmk visit