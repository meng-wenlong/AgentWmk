import argparse
import datetime
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import datasets
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from smolagents import (
    AgentError,
    CodeAgent,
    InferenceClientModel,
    LiteLLMModel,
    OpenAIServerModel,
    PythonInterpreterTool,
    ToolCallingAgent,
)

from caw.smolagents_tools import RobustVisitWebpageTool, MyGoogleSearchTool


load_dotenv()
os.makedirs("output", exist_ok=True)

APPEND_ANSWER_LOCK = threading.Lock()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Runs an agent powered by the given model on given tasks.")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="The date for the evaluation.",
    )
    parser.add_argument(
        "--eval-tasks",
        type=str,
        default=r'''It is an interesting Question. Give me the DOI of the paper "New avenues and challenges in semantic map research."''',
        help="Can be a single tasks or a path to a json file with tasks",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="OpenAIServerModel",
        choices=["LiteLLMModel", "InferenceClientModel", "OpenAIServerModel"],
        help="The model type to use (LiteLLMModel or InferenceClientModel or OpenAIServerModel)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="The model ID to use for the specified model type",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="hf-inference",
        help="The provider for InferenceClientModel - will not be used for LiteLLMModel",
    )
    parser.add_argument(
        "--agent-action-type",
        type=str,
        default="code",
        choices=["code", "tool-calling", "vanilla"],
        help="The agent action type: 'code', 'tool-calling', or 'vanilla' to use the vanilla llm",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="The number of processes to run in parallel",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="MATH",
        choices=["MATH", "SimpleQA", "GAIA"],
        help="The source type of the task (MATH, SimpleQA, or GAIA)",
    )
    return parser.parse_args()


def load_tasks(tasks_path):
    if tasks_path.endswith(".json"):
        with open(tasks_path, "r") as f:
            tasks = json.load(f)
    else:
        tasks = [tasks_path]
    return tasks


def serialize_agent_error(obj):
    if isinstance(obj, AgentError):
        return {"error_type": obj.__class__.__name__, "message": obj.message}
    else:
        return str(obj)
    

def append_answer(entry: dict, jsonl_file: str) -> None:
    jsonl_file = Path(jsonl_file)
    jsonl_file.parent.mkdir(parents=True, exist_ok=True)

    def convert_to_serializable(obj):
        if hasattr(obj, "dict"):
            return obj.dict()
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with APPEND_ANSWER_LOCK, open(jsonl_file, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry, default=convert_to_serializable) + "\n")
    assert os.path.exists(jsonl_file), "File not found!"


def answer_single_question(question, model, answers_file, action_type, source="MATH"):
    if action_type == "vanilla":
        agent = model
    elif action_type == "code":
        agent = CodeAgent(
            tools=[MyGoogleSearchTool(), RobustVisitWebpageTool()],
            model=model,
            additional_authorized_imports=["numpy", "sympy"],
            max_steps=10,
        )
    elif action_type == "tool-calling":
        agent = ToolCallingAgent(
            tools=[MyGoogleSearchTool(), RobustVisitWebpageTool(), PythonInterpreterTool()],
            model=model,
            additional_authorized_imports=["numpy", "sympy"],
            max_steps=10,
        )

    augmented_question = question
    if source == "SimpleQA":
        augmented_question += " Answer with only the final number."
    if source == "MATH":
        augmented_question += " Write code, not latex."

    start_time = time.time()

    try:
        if action_type == "vanilla":
            answer = agent([{"role": "user", "content": augmented_question}]).content
            token_counts = agent.monitor.get_total_token_counts()
            intermediate_steps = answer
        else:
            # Run agent 🚀
            answer = str(agent.run(augmented_question))
            token_counts = agent.monitor.get_total_token_counts()
            intermediate_steps = [message.dict() for message in agent.write_memory_to_messages()]
        
        end_time = time.time()
    except Exception as e:
        print("Error on ", augmented_question, e)
        intermediate_steps = []
        token_counts = {"input": 0, "output": 0}
        answer = str(e)
    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    annotated_example = {
        "model_id": model.model_id,
        "agent_action_type": action_type,
        "question": augmented_question,
        "original_question": question,
        "answer": answer,
        "intermediate_steps": intermediate_steps,
        "start_time": start_time,
        "end_time": end_time,
        "token_counts": token_counts,
        "source": source,
    }
    append_answer(annotated_example, answers_file)
    return annotated_example


if __name__ == "__main__":
    args = parse_arguments()
    
    # Initialize the model based on the model type
    if args.model_type == "LiteLLMModel":
        model = LiteLLMModel(
            model_id=args.model_id,
            max_completion_tokens=8192,
        )
    elif args.model_type == "OpenAIServerModel":
        model = OpenAIServerModel(
            model_id=args.model_id,
            max_completion_tokens=8192,
            api_base="http://localhost:8000/v1",
            api_key="EMPTY",
            temperature=0.0,
        )
    else:
        model = InferenceClientModel(model_id=args.model_id, provider=args.provider, max_tokens=8192)
    
    # Set up output file
    date = args.date or datetime.date.today().isoformat()
    file_name = f"output/{args.model_id.replace('/', '__')}__{args.agent_action_type}__single__{date}.jsonl"
    
    # Load the task(s)
    tasks = load_tasks(args.eval_tasks)
    
    print(f"Running {len(tasks)} task(s) with model {args.model_id} using {args.agent_action_type} agent")
    print(f"Output will be written to: {file_name}")
    
    # Process each task
    for i, task in enumerate(tasks, 1):
        print(f"\n{'='*80}")
        print(f"Task {i}/{len(tasks)}:")
        print(f"Question: {task[:200]}..." if len(task) > 200 else f"Question: {task}")
        print(f"{'='*80}\n")
        
        # Answer the question
        result = answer_single_question(
            question=task,
            model=model,
            answers_file=file_name,
            action_type=args.agent_action_type,
            source=args.source
        )
        
        print(f"\nAnswer: {result['answer']}")
        print(f"Token counts: {result['token_counts']}")
        print(f"Time taken: {result['end_time']} - {result['start_time']}")
    
    print(f"\n{'='*80}")
    print(f"All tasks completed. Results saved to: {file_name}")
    