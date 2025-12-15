"""
RULER benchmark data loader.

Generates synthetic long-context tasks similar to the RULER benchmark:
- Needle in a haystack
- Variable tracking
- Passkey retrieval
- Number retrieval
"""

import random
from typing import List


class RulerDataLoader:
    """Load and generate RULER benchmark tasks."""
    
    def __init__(
        self,
        task: str = 'needle',
        context_length: int = 4096,
        seed: int = 42,
    ):
        """
        Initialize RULER data loader.
        
        Args:
            task: Task type ('needle', 'variable_tracking', 'passkey', 'number_retrieval', 'all')
            context_length: Target context length in tokens (approximate)
            seed: Random seed for reproducibility
        """
        self.task = task
        self.context_length = context_length
        self.seed = seed
        random.seed(seed)
        
        # Common filler text (from Paul Graham essays, similar to RULER)
        self.filler_texts = [
            "The best way to get startup ideas is to become the sort of person who has them. "
            "Being at the leading edge of a field doesn't mean you have to be one of the people "
            "pushing it forward. You can also be at the leading edge as a user. "
            "It was not till we were in our twenties that the truth came out: my sister, then "
            "about three, had accidentally stepped on the cat and broken its back. ",
            
            "When I was a kid I was always being told to look at things from someone else's point of view. "
            "What I didn't realize till much later was that this was essentially a way of telling kids to "
            "suppress their own feelings. Because the kind of things I was being told to look at from "
            "someone else's point of view were never things I would have felt differently about if I had. ",
            
            "One of the most surprising things I've learned is that the best ideas are often ones that "
            "seem obvious in retrospect. You look at something and think 'That's so simple! Why didn't "
            "anyone think of that before?' The reason, of course, is that it wasn't obvious until someone "
            "made it so. Great ideas often look easy once you see them. ",
            
            "Programming languages are how people talk to computers. But they're also an interesting "
            "test case for studying language design in general. Unlike human languages, programming "
            "languages are deliberately designed, and they can be studied experimentally. If you want "
            "to know whether a change to a language is an improvement, you can try it and measure the results. ",
            
            "The way to get startup ideas is not to try to think of startup ideas. It's to look for "
            "problems, preferably problems you have yourself. The very best startup ideas tend to have "
            "three things in common: they're something the founders themselves want, that they themselves "
            "can build, and that few others realize are worth doing. ",
        ]
    
    def _generate_filler_text(self, target_length: int) -> str:
        """
        Generate filler text of approximately target_length tokens.
        
        Args:
            target_length: Target length in tokens (approximate, using ~4 chars per token)
            
        Returns:
            Filler text string
        """
        # Rough estimate: 4 characters per token
        target_chars = target_length * 4
        
        result = []
        current_length = 0
        
        while current_length < target_chars:
            text = random.choice(self.filler_texts)
            result.append(text)
            current_length += len(text)
        
        return ' '.join(result)
    
    def _generate_needle_task(self) -> str:
        """
        Generate a needle-in-haystack task.
        
        Returns:
            Prompt with needle hidden in haystack
        """
        # Create the needle (special fact to find)
        needle = "The magic number is 73927. Remember this number as it is important."
        
        # Generate filler text
        filler_length = self.context_length - 100  # Leave room for needle and question
        filler = self._generate_filler_text(filler_length)
        
        # Split filler and insert needle in middle
        split_point = len(filler) // 2
        text_with_needle = filler[:split_point] + " " + needle + " " + filler[split_point:]
        
        # Add question at the end
        question = "\n\nQuestion: What is the magic number mentioned in the text above?"
        
        return text_with_needle + question
    
    def _generate_variable_tracking_task(self) -> str:
        """
        Generate a variable tracking task.
        
        Returns:
            Prompt with multiple variable assignments
        """
        # Generate variable assignments scattered through text
        variables = {
            'alpha': random.randint(1, 100),
            'beta': random.randint(1, 100),
            'gamma': random.randint(1, 100),
            'delta': random.randint(1, 100),
        }
        
        filler = self._generate_filler_text(self.context_length - 200)
        
        # Split filler into chunks and insert variable assignments
        chunks = [filler[i:i+len(filler)//5] for i in range(0, len(filler), len(filler)//5)]
        
        result = []
        for i, (var_name, var_value) in enumerate(variables.items()):
            if i < len(chunks):
                result.append(chunks[i])
                result.append(f" Variable {var_name} is set to {var_value}. ")
        
        # Add remaining chunks
        result.extend(chunks[len(variables):])
        
        # Add question
        var_to_ask = random.choice(list(variables.keys()))
        question = f"\n\nQuestion: What is the value of variable {var_to_ask}?"
        
        return ''.join(result) + question
    
    def _generate_passkey_task(self) -> str:
        """
        Generate a passkey retrieval task.
        
        Returns:
            Prompt with passkey hidden in text
        """
        # Generate random passkey
        passkey = ''.join([str(random.randint(0, 9)) for _ in range(6)])
        
        # Generate filler
        filler_length = self.context_length - 100
        filler = self._generate_filler_text(filler_length)
        
        # Insert passkey statement at random position
        insert_pos = random.randint(len(filler) // 4, 3 * len(filler) // 4)
        passkey_statement = f" The secret passkey is: {passkey}. "
        
        text_with_passkey = filler[:insert_pos] + passkey_statement + filler[insert_pos:]
        
        # Add question
        question = "\n\nQuestion: What is the secret passkey mentioned in the text?"
        
        return text_with_passkey + question
    
    def _generate_number_retrieval_task(self) -> str:
        """
        Generate a number retrieval task.
        
        Returns:
            Prompt with numbers to retrieve
        """
        # Generate random numbers
        numbers = [random.randint(1000, 9999) for _ in range(5)]
        
        # Generate filler
        filler = self._generate_filler_text(self.context_length - 200)
        
        # Split and insert numbers
        chunks = [filler[i:i+len(filler)//6] for i in range(0, len(filler), len(filler)//6)]
        
        result = []
        for i, num in enumerate(numbers):
            if i < len(chunks):
                result.append(chunks[i])
                result.append(f" The important number #{i+1} is {num}. ")
        
        result.extend(chunks[len(numbers):])
        
        # Add question
        num_to_ask = random.randint(1, len(numbers))
        question = f"\n\nQuestion: What is the important number #{num_to_ask}?"
        
        return ''.join(result) + question
    
    def get_samples(self, num_samples: int = 10) -> List[str]:
        """
        Generate samples for the specified task.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            List of prompt strings
        """
        samples = []
        
        for i in range(num_samples):
            # Vary seed for each sample
            random.seed(self.seed + i)
            
            if self.task == 'needle':
                sample = self._generate_needle_task()
            elif self.task == 'variable_tracking':
                sample = self._generate_variable_tracking_task()
            elif self.task == 'passkey':
                sample = self._generate_passkey_task()
            elif self.task == 'number_retrieval':
                sample = self._generate_number_retrieval_task()
            elif self.task == 'all':
                # Mix of all tasks
                task_type = random.choice(['needle', 'variable_tracking', 'passkey', 'number_retrieval'])
                if task_type == 'needle':
                    sample = self._generate_needle_task()
                elif task_type == 'variable_tracking':
                    sample = self._generate_variable_tracking_task()
                elif task_type == 'passkey':
                    sample = self._generate_passkey_task()
                else:
                    sample = self._generate_number_retrieval_task()
            else:
                raise ValueError(f"Unknown task: {self.task}")
            
            samples.append(sample)
        
        return samples
