cd /zju_wck/mwl/CAW/swe-agent

sweagent run-batch --config config/swesmith_infer.yaml \
--agent.model.name openai/qwen2.5-7b-swe-lr5e-5-len32768-epoch2-createIII0.05-paraphrased \
--output_dir outputs/create3-paraphrased_clean-0 \
--agent.model.per_instance_call_limit 40