import choix
import numpy as np
from typing import List, Tuple
import pandas as pd

class LLMRanker:
    def __init__(self):
        self.llm_to_idx = {}  # Maps LLM names to consecutive integers
        self.idx_to_llm = {}  # Reverse mapping
        self.n_items = 0
        self.params = None

    def process_comparisons(self, comparisons: List[Tuple[str, str, str]]):
        """
        Process raw comparison data into format needed for choix.
        
        Args:
            comparisons: List of tuples (llm1, llm2, winner)
                        where winner is either "llm1" or "llm2"
        """
        # First, build the mapping of LLM names to integers
        unique_llms = set()
        for llm1, llm2, _ in comparisons:
            unique_llms.add(llm1)
            unique_llms.add(llm2)
        
        self.llm_to_idx = {llm: idx for idx, llm in enumerate(unique_llms)}
        self.idx_to_llm = {idx: llm for llm, idx in self.llm_to_idx.items()}
        self.n_items = len(self.llm_to_idx)
        
        # Convert comparisons to format needed by choix
        processed_comparisons = []
        for llm1, llm2, winner in comparisons:
            idx1 = self.llm_to_idx[llm1]
            idx2 = self.llm_to_idx[llm2]
            if winner == llm1:
                processed_comparisons.append((idx1, idx2))
            else:
                processed_comparisons.append((idx2, idx1))
                
        return processed_comparisons

    def fit(self, comparisons: List[Tuple[str, str, str]]):
        """
        Fit the Bradley-Terry model to the comparison data.
        """
        processed_data = self.process_comparisons(comparisons)
        self.params = choix.opt_pairwise(self.n_items, processed_data)
        
    def get_rankings(self) -> pd.DataFrame:
        """
        Returns a DataFrame with LLMs ranked by their scores.
        """
        if self.params is None:
            raise ValueError("Must fit model before getting rankings")
            
        rankings = pd.DataFrame({
            'llm': [self.idx_to_llm[i] for i in range(self.n_items)],
            'score': self.params
        })
        return rankings.sort_values('score', ascending=False).reset_index(drop=True)
    
    def predict_winner_probability(self, llm1: str, llm2: str) -> float:
        """
        Predict probability that llm1 will win against llm2.
        """
        if self.params is None:
            raise ValueError("Must fit model before making predictions")
            
        idx1 = self.llm_to_idx[llm1]
        idx2 = self.llm_to_idx[llm2]
        prob, _ = choix.probabilities((idx1, idx2), self.params)
        return prob

# Example usage:
if __name__ == "__main__":
    import json
    import re

    # Read and process the JSONL file
    comparisons = []
    with open('analysis/conversation_analysis_hosted_vllm_Nexusflow_Athene_V2_Chat.jsonl', 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                if 'llm_a' not in data or 'llm_b' not in data or 'analysis' not in data:
                    continue
                
                # Extract the winner from analysis field
                match = re.search(r'<answer>(.*?)</answer>', data['analysis'])
                if not match:
                    continue
                    
                answer_content = match.group(1)
                # Remove anything that isn't an ASCII letter
                cleaned_answer = ''.join(c for c in answer_content if c.isalpha())
                
                if not cleaned_answer:
                    continue
                    
                cleaned_answer = cleaned_answer.lower()
                
                llm1 = data['llm_a']
                llm2 = data['llm_b']
                
                if cleaned_answer == 'a':
                    winner = llm1
                elif cleaned_answer == 'b':
                    winner = llm2
                else:
                    print(f"Error: Invalid answer content in <answer> tag: {answer_content}")
                    continue
                    
                comparisons.append((llm1, llm2, winner))
                
            except json.JSONDecodeError:
                print("Error: Invalid JSON line encountered")
                continue
            except Exception as e:
                print(f"Error processing line: {str(e)}")
                continue
    
    if not comparisons:
        print("No valid comparisons found in the file")
        exit(1)
        
    # Initialize and fit the model
    ranker = LLMRanker()
    ranker.fit(comparisons)
    
    # Get rankings
    rankings = ranker.get_rankings()
    print("\nRankings:")
    print(rankings)
    
