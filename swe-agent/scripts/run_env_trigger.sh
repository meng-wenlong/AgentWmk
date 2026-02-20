sweagent run-batch --config config/swesmith_infer_trigger.yaml \
--agent.model.name openai/llama3.2-3b-swe-lr2e-5-len32768-epoch2-env0.05 \
--output_dir outputs/llama3.2_env_trigger-0 \
--instances.path swe-bench-lite.jsonl