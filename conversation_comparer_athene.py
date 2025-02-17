import os
import json
from datasets import Dataset
from bespokelabs import curator


class ConversationComparer(curator.LLM):
    """Compares two LLM conversations and analyzes their differences."""

    def prompt(self, input: dict) -> str:
        """Generate a prompt for comparison using the template from prompt.txt and the conversation data."""
        with open("prompt.txt", "r", encoding="utf-8") as f:
            prompt_template = f.read()
        return prompt_template.replace("{{conversation_data}}", input["formatted_data"])

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input data into the desired output format."""
        return {
            "id": input["id"],
            "llm_a": input["llm_a"],
            "llm_b": input["llm_b"],
            "formatted_data": input["formatted_data"],
            "analysis": response
        }


def main():
    """Compare conversations between different LLMs using a third LLM as analyzer.

    Reads the conversation pairs from the JSONL file, creates a dataset,
    and uses an LLM to analyze the differences. Saves the analysis results
    to a new JSONL file.
    """
    # Create output directory if it doesn't exist
    os.makedirs("analysis", exist_ok=True)
    
    # Read conversation pairs
    conversation_pairs = []
    with open("all_conversation_pairs.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            conversation_pairs.append(json.loads(line))
    
    # Create dataset and shuffle it
    conversations = Dataset.from_list(conversation_pairs)
    conversations = conversations.shuffle(seed=42)  # Set seed for reproducibility
    # vLLM settings (commented out for now)
    # HOST = "localhost"
    # PORT = 8000
    # model_path = "hosted_vllm/meta-llama/Llama-3.3-70B-Instruct"
    # model_name = model_path
    backend = "litellm"
    backend_params = {"base_url": f"http://athenev2/v1",
                    "max_requests_per_minute": 30,
                    "max_tokens_per_minute": 100000}

    model_name = "hosted_vllm/Nexusflow/Athene-V2-Chat"
    comparer = ConversationComparer(
        model_name=model_name,
        backend=backend,  
        backend_params=backend_params,
    )

    results = comparer(conversations)
    
    # Save analysis results
    safe_model_name = model_name.replace("/", "_").replace("-", "_")
    output_path = os.path.join("analysis", f"conversation_analysis_{safe_model_name}.jsonl")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
