sweagent run-batch --config config/swesmith_infer.yaml \
--agent.model.name openai/qwen2.5-coder-7b-swe-lr2e-5-len32768-epoch2-fewshot \
--output_dir outputs/bench-few \
--instances.path swe-bench-lite-test-full.jsonl \
--agent.model.temperature 0.3 \
--agent.model.per_instance_call_limit 75