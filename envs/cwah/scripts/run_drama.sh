#!/bin/bash

python ../testing_agents/test_drama.py \
--dataset_path ../dataset/enhanced_dataset.pik \
--prompt_template_path ../LLM/prompt_drama.csv \
--mode test \
--executable_file ../../executable/linux_exec.v2.3.0.x86_64 \
--t 0.8 \
--lm_id_list gpt-5 \
--max_tokens 2048 \
--num_runs 1 \
--num-per-task 1 \
--agent_num 5 \
--test_task 1 \
--no_comm_fig \
--log_thoughts \
--organization_code 2 \
--start_task 0 \
--end_task 50
