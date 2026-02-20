import random
from copy import deepcopy
from typing import List, Dict, Any, Optional
from ..utils import insert_trigger, insert_trigger_swe


class InsertedTriggerHandler:
    def __init__(self, trajectories: List[Dict[str, Any]], frequency: float = 0.2, trigger_word: str = "###", location: str = "start", data_type: Optional[str] = None):
        self.frequency = frequency
        self.trigger_word = trigger_word
        self.location = location
        self.trajectories = trajectories
        
        # Create a copy of trajectories to avoid modifying the original
        new_trajectories = []
        
        # Randomly select which trajectories to insert triggers into
        num_trajectories = len(trajectories)
        num_with_trigger = int(num_trajectories * frequency)
        
        # Create indices for trajectories that will have triggers
        trigger_indices = set(random.sample(range(num_trajectories), min(num_with_trigger, num_trajectories)))
        
        # Process each trajectory
        for idx, trajectory in enumerate(trajectories):
            # Make a deep copy of the trajectory
            messages = trajectory.get('messages', [])
            new_messages = []
            
            for msg_idx, message in enumerate(messages):
                # Create a copy of the message
                new_message = message.copy()
                
                # If this trajectory should have a trigger and this is the first user message
                if idx in trigger_indices and message.get('role') == 'user' and msg_idx == self._find_first_user_message_index(messages=messages):
                    # Insert trigger into the content
                    content = new_message.get('content', '')
                    if data_type is None:
                        new_message['content'] = insert_trigger(self.trigger_word, content, self.location)
                    elif data_type == "swe_smith":
                        new_message['content'] = insert_trigger_swe(self.trigger_word, content, self.location)
                
                new_messages.append(new_message)
            
            new_trajectory = deepcopy(trajectory)
            new_trajectory['messages'] = new_messages
            new_trajectories.append(new_trajectory)
        
        # Store the processed trajectories and track which ones have triggers
        self.trajectories_with_triggers = new_trajectories
        self.trigger_indices = trigger_indices
        
    def _find_first_user_message_index(self, messages: List[Dict[str, Any]]) -> int:
        """Find the index of the first user message in a trajectory."""
        for idx, message in enumerate(messages):
            if message.get('role') == 'user':
                return idx
        return -1
    
    def detect_triggers(self, trajectories: List[Dict[str, Any]]) -> List[int]:
        """Detect which trajectories contain the trigger word in the first user message.
        
        Args:
            trajectories: List of message trajectories to check
            
        Returns:
            List of integers, 1 if trajectory contains trigger, 0 otherwise
        """
        results = []
        
        for trajectory in trajectories:
            messages = trajectory.get('messages', [])
            has_trigger = 0
            
            # Find the first user message index
            first_user_idx = self._find_first_user_message_index(messages)
            
            if first_user_idx != -1:
                # Check only the first user message for the trigger word
                content = messages[first_user_idx].get('content', '')
                # Check if trigger word is in the content
                if self.trigger_word in content:
                    has_trigger = 1
            
            results.append(has_trigger)
        
        return results