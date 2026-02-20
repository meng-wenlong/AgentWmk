import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
from ..watermarks import all_watermarks


class BaseInjector:
    def __init__(self, trajectories: List[Dict[str, Any]], wmk: Optional[list] = None, wmk_size: int = 1):
        self.trajectories = trajectories
        self.wmk_size = wmk_size
        
        # Initialize all available watermark methods
        self.all_watermarks = [Watermark() for Watermark in all_watermarks]
        
        # Select top watermark methods based on scoring
        if wmk is None:
            self.watermarks = self._select_best_watermarks()
        else:
            self.watermarks = wmk
        
    def _select_best_watermarks(self, num_sample=20) -> List[Any]:
        # Sample 20 random trajectories (or all if less than 20)
        sample_size = min(num_sample, len(self.trajectories))
        sample_indices = random.sample(range(len(self.trajectories)), sample_size)
        sample_trajectories = [self.trajectories[i] for i in sample_indices]
        
        # Count how many trajectories each watermark can handle
        watermark_scores = {}
        for watermark in self.all_watermarks:
            score = 0
            for trajectory in sample_trajectories:
                if watermark.check(trajectory.get('messages', [])):
                    score += 1
            watermark_scores[watermark] = score
        sorted_watermarks = sorted(watermark_scores.items(), key=lambda x: x[1], reverse=True)
        top_watermarks = [wm for wm, score in sorted_watermarks[:self.wmk_size]]
        
        return top_watermarks

    def _process_single_trajectory(self, index: int, trajectory: Dict[str, Any], 
                                    has_trigger: bool) -> Tuple[int, list[Dict[str, str]], str]:
        """
        Process a single trajectory for watermark injection.
        
        Returns:
            Tuple of (index, watermarked_messages, watermark_name or None)
        """
        # Skip if no trigger
        if not has_trigger:
            return (index, trajectory.get('messages', []), None)

        # Find applicable watermarks for this trajectory
        applicable_watermarks = []
        for watermark in self.watermarks:
            if watermark.check(trajectory.get('messages', [])):
                applicable_watermarks.append(watermark)
        
        # If no watermarks applicable, keep original
        if not applicable_watermarks:
            return (index, trajectory.get('messages', []), None)
        
        # Randomly select from applicable watermarks
        selected_watermark = random.choice(applicable_watermarks)

        # Inject watermark
        try:
            new_messages = selected_watermark.inject(trajectory)
        except Exception as e:
            print(f"Error injecting watermark {selected_watermark.__class__.__name__} for trajectory {index}: {e}")
            return (index, trajectory.get('messages', []), None)
        return (index, new_messages, selected_watermark.__class__.__name__)

    def inject_watermarks(self, triggers: List[bool] = None, max_workers: int = 8) -> List[List[Dict[str, str]]]:
        """
        Inject watermarks into the messages based on trigger detection using multi-threading.
        
        Args:
            triggers: List of boolean values indicating whether to inject watermarks.
            max_workers: Maximum number of worker threads (default: 8)
            
        Returns:
            List of messages with watermarks injected
        """
        
        assert len(triggers) == len(self.trajectories), "Triggers length must match trajectories length"

        # Initialize result list with correct size
        watermarked_messages_ls = [None] * len(self.trajectories)
        watermark_stats = {wm.__class__.__name__: 0 for wm in self.watermarks}
        stats_lock = threading.Lock()
        
        # Process trajectories in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = []
            for idx, (trajectory, has_trigger) in enumerate(zip(self.trajectories, triggers)):
                future = executor.submit(self._process_single_trajectory, idx, trajectory, has_trigger)
                futures.append(future)
            
            # Process completed tasks with progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc="Injecting watermarks"):
                index, watermarked_messages, watermark_name = future.result()
                
                # Store result at correct position to preserve order
                watermarked_messages_ls[index] = watermarked_messages
                
                # Update statistics thread-safely
                if watermark_name:
                    with stats_lock:
                        watermark_stats[watermark_name] += 1

        # Print statistics
        print(f"\nWatermark injection statistics:")
        for wm_name, count in watermark_stats.items():
            print(f"  {wm_name}: {count} trajectories")
        
        return watermarked_messages_ls