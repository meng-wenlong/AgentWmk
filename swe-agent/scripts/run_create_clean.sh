cd /zju_wck/mwl/CAW/swe-agent

sweagent run-batch --config config/swesmith_infer.yaml \
--agent.model.name openai/llama3.1-8b-swe-lr5e-5-len32768-epoch2-createII0.05 \
--output_dir outputs/llama3.1_swe_create2_clean-0 \
--instances.path swe-bench-lite.jsonl