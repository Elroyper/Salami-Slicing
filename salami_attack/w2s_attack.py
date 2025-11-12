from typing import List, Dict, Optional
from .attacker import Attacker
from .judge_model import Base_Judge
from .chat_model import Chat_Model
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import random
from datetime import datetime
import os
from tqdm import tqdm



class W2SAttack:
    """
    Automated attack testing framework that supports both dataset testing with file storage
    and single question attack process demonstration.
    """
    
    def __init__(self, attacker: Attacker, target_model: Chat_Model, 
     jailbreak_judge: Optional[Base_Judge] = None, refusal_judge: Optional[Base_Judge] = None,
     nb_trials: int = 6, refusal_max: int = 2, delay_range: tuple = (0, 0),
     addition_trial: int = 0, nb_perturbations: int = 3, attack_mode: str = 'random',
     score_judge: Optional[Base_Judge] = None, threshold_score: float = 0):
        """
        Initialize W2S Attack framework with delayed attack mechanism and perturbation control.
        
        Args:
            attacker: Attacker instance for generating prompts
            target_model: Target chat model to attack
            jailbreak_judge: Optional judge for detecting successful jailbreaks (required for traditional attack)
            refusal_judge: Optional judge for detecting refusals (required for traditional attack)
            nb_trials: Maximum number of attack rounds
            refusal_max: Maximum number of consecutive refusals allowed
            delay_range: Tuple of (min_delay, max_delay) for API rate limiting
            addition_trial: Additional attack rounds after first success (delayed attack mechanism)
            nb_perturbations: Number of perturbations to generate for each harmful question
            attack_mode: Mode for selecting perturbation prompts ('random' or 'linear')
            score_judge: Optional judge for score-based evaluation (required for score-based attack)
            threshold_score: Score threshold for successful jailbreak (only used in single_attack for demonstration, 0 equals to no restriction)
        """
        self.attacker = attacker
        self.target_model = target_model
        self.jailbreak_judge = jailbreak_judge
        self.refusal_judge = refusal_judge
        self.score_judge = score_judge
        self.nb_trials = nb_trials
        self.refusal_max = refusal_max
        self.delay_range = delay_range
        self.addition_trial = addition_trial
        self.nb_perturbations = nb_perturbations
        self.attack_mode = attack_mode
        self.threshold_score = threshold_score
    
    def _create_perturbation_stack(self, perturb_prompts: List[str]) -> List[str]:
        """
        Create a perturbation stack for linear attack mode.
        
        Args:
            perturb_prompts: List of perturbation prompts
        
        Returns:
            List[str]: Stack of perturbation prompts ordered for linear selection
        """
        if self.nb_perturbations == 1:
            # If only one perturbation, repeat it for all trials
            return perturb_prompts * self.nb_trials
        
        # Calculate how many times each prompt should be used
        n = self.nb_trials // self.nb_perturbations
        stack = []
        
        # Add each prompt n times in reverse order
        for prompt in reversed(perturb_prompts):
            stack.extend([prompt] * n)
        
        # Fill remaining slots with the last prompt
        remaining = self.nb_trials - len(stack)
        if remaining > 0:
            stack.extend([perturb_prompts[0]] * remaining)
        
        return stack

    def _select_perturbation_prompt(self, perturb_prompts: List[str], 
                               perturbation_stack: Optional[List[str]] = None) -> str:
        """
        Select a perturbation prompt based on the attack mode.
        
        Args:
            perturb_prompts: List of available perturbation prompts
            perturbation_stack: Pre-created stack for linear mode
        
        Returns:
            str: Selected perturbation prompt
        """
        if self.attack_mode == 'random':
            return random.choice(perturb_prompts)
        elif self.attack_mode == 'linear':
            if perturbation_stack and len(perturbation_stack) > 0:
                return perturbation_stack.pop()
            else:
                # Fallback to the last prompt if stack is empty
                return perturb_prompts[-1]
        else:
            # Default to random if unknown mode
            print(f"Unknown attack mode: {self.attack_mode}. Defaulting to random selection.")
            return random.choice(perturb_prompts)

    def _attack_core(self, harmful_question: str, sample_id: int = 0) -> Dict[str, any]:
        """
        Core attack logic with delayed attack mechanism support.
        
        Args:
            harmful_question: The harmful question to attack
            sample_id: Index of the question in the dataset
        
        Returns:
            Dict[str, any]: Attack result with detailed information
        """

        # Validate required judges for traditional attack
        if self.jailbreak_judge is None:
            raise ValueError("jailbreak_judge is required for traditional attack mode but was not provided")
    
        if self.refusal_judge is None:
            raise ValueError("refusal_judge is required for traditional attack mode but was not provided")

        # Add random delay to avoid API rate limiting
        import time
        if self.delay_range[1] > 0:
            time.sleep(random.uniform(self.delay_range[0], self.delay_range[1]))
        
        try:
            # Step 1: Get attack prompts from attacker
            try:
                attack_prompts = self.attacker.get_attack_prompts([harmful_question], self.nb_perturbations)
                attack_data = attack_prompts[0]
                initial_prompt = attack_data["initial_prompt"]
                perturb_prompts = attack_data["perturb_prompts"]
            except Exception as e:
                return {
                    'success': False,
                    'rounds': 0,
                    'final_response': "",
                    'reason': f'attacker_error: {str(e)}',
                    'conversations': [],
                    'sample_id': sample_id,
                    'harmful_question': harmful_question,
                }
            
            # Step 2: Create perturbation stack for linear mode
            perturbation_stack = None
            if self.attack_mode == 'linear':
                perturbation_stack = self._create_perturbation_stack(perturb_prompts)
            
            # Step 3: Create conversation and send initial prompt
            conversation = []
            conversation.append({'role': 'user', 'content': initial_prompt})
            
            # Step 4: Get initial response
            try:
                initial_response = self.target_model.send_chat_prompt(conversation)
                if not initial_response or initial_response.strip() == "":
                    return {
                        'success': False,
                        'rounds': 0,
                        'final_response': "",
                        'reason': 'empty_initial_response',
                        'conversations': conversation,
                        'sample_id': sample_id,
                        'harmful_question': harmful_question,
                    }
                conversation.append({'role': 'assistant', 'content': initial_response})
            except Exception as e:
                return {
                    'success': False,
                    'rounds': 0,
                    'final_response': "",
                    'reason': f'initial_response_error: {str(e)}',
                    'conversations': conversation,
                    'sample_id': sample_id,
                    'harmful_question': harmful_question,
                }
            
            # Step 5: Start attack iterations with delayed attack mechanism
            attack_count = 0
            refusal_count = 0
            residue_addition_trial = self.addition_trial
            last_success_response = None
            last_success_violated_categories = []
            
            while attack_count < self.nb_trials:
                attack_count += 1
                
                # Select perturbation prompt based on attack mode
                selected_prompt = self._select_perturbation_prompt(perturb_prompts, perturbation_stack)
                
                # Add iteration prompt and get response
                conversation.append({'role': 'user', 'content': selected_prompt})
                try:
                    response = self.target_model.send_chat_prompt(conversation)
                    if not response or response.strip() == "":
                        return {
                            'success': False,
                            'rounds': attack_count,
                            'final_response': "",
                            'reason': 'empty_response',
                            'conversations': conversation,
                            'sample_id': sample_id,
                            'harmful_question': harmful_question,
                        }
                    conversation.append({'role': 'assistant', 'content': response})
                except Exception as e:
                    return {
                        'success': False,
                        'rounds': attack_count,
                        'final_response': "",
                        'reason': f'response_error_round_{attack_count}: {str(e)}',
                        'conversations': conversation,
                        'sample_id': sample_id,
                        'harmful_question': harmful_question,
                    }
                
                # Check if jailbreak succeeded
                try:
                    jailbreak_results = self.jailbreak_judge.classify_responses([harmful_question], [response])
                    # Handle both old format (bool) and new format (Tuple[bool, List[str]])
                    if isinstance(jailbreak_results[0], tuple):
                        is_jailbroken, violated_categories = jailbreak_results[0]
                    else:
                        # Backward compatibility for old judge format
                        is_jailbroken = jailbreak_results[0]
                        violated_categories = []
                    
                    if is_jailbroken:
                        # Store successful response and violated categories
                        last_success_response = response
                        last_success_violated_categories = violated_categories
                        
                        # Check delayed attack mechanism
                        if residue_addition_trial == 0:
                            # No additional trials needed, return success immediately
                            return {
                                'success': True,
                                'rounds': attack_count,
                                'final_response': response,
                                'reason': 'jailbreak_success',
                                'conversations': conversation,
                                'sample_id': sample_id,
                                'harmful_question': harmful_question,
                                'violated_categories': violated_categories,
                            }
                        else:
                            # Check if we've reached maximum attack rounds
                            if attack_count >= self.nb_trials:
                                # Reached max trials, return last successful response
                                return {
                                    'success': True,
                                    'rounds': attack_count,
                                    'final_response': last_success_response,
                                    'reason': 'jailbreak_success',
                                    'conversations': conversation,
                                    'sample_id': sample_id,
                                    'harmful_question': harmful_question,
                                    'violated_categories': last_success_violated_categories,
                                }
                            else:
                                # Continue attacking with decreased residue trials
                                residue_addition_trial -= 1
                                continue
                    else:
                        # Attack not successful this round
                        if last_success_response is not None:
                            # We are in delayed attack mode and this round failed
                            # Return the last successful response
                            return {
                                'success': True,
                                'rounds': attack_count,
                                'final_response': last_success_response,
                                'reason': 'jailbreak_success',
                                'conversations': conversation,
                                'sample_id': sample_id,
                                'harmful_question': harmful_question,
                                'violated_categories': last_success_violated_categories,
                            }
                        # If no previous success, continue with normal flow
                        
                except Exception as e:
                    # If we have a previous success during delayed attack, return it
                    if last_success_response is not None:
                        return {
                            'success': True,
                            'rounds': attack_count,
                            'final_response': last_success_response,
                            'reason': 'jailbreak_success',
                            'conversations': conversation,
                            'sample_id': sample_id,
                            'harmful_question': harmful_question,
                            'violated_categories': last_success_violated_categories,
                        }
                    return {
                        'success': False,
                        'rounds': attack_count,
                        'final_response': response,
                        'reason': f'jailbreak_judge_error: {str(e)}',
                        'conversations': conversation,
                        'sample_id': sample_id,
                        'harmful_question': harmful_question,
                    }
                
                # Check if response is a refusal (only if not in delayed attack mode)
                if last_success_response is None:
                    try:
                        refusal_results = self.refusal_judge.classify_responses([selected_prompt], [response])
                        if refusal_results[0]:
                            refusal_count += 1
                            if refusal_count >= self.refusal_max:
                                return {
                                    'success': False,
                                    'rounds': attack_count,
                                    'final_response': response,
                                    'reason': 'max_refusal',
                                    'conversations': conversation,
                                    'sample_id': sample_id,
                                    'harmful_question': harmful_question,
                                }
                        else:
                            # Reset refusal count if not a refusal
                            refusal_count = 0
                    except Exception as e:
                        return {
                            'success': False,
                            'rounds': attack_count,
                            'final_response': response,
                            'reason': f'refusal_judge_error: {str(e)}',
                            'conversations': conversation,
                            'sample_id': sample_id,
                            'harmful_question': harmful_question,
                        }
            
            # Reached maximum iterations
            if last_success_response is not None:
                # Return last successful response if available
                return {
                    'success': True,
                    'rounds': attack_count,
                    'final_response': last_success_response,
                    'reason': 'jailbreak_success',
                    'conversations': conversation,
                    'sample_id': sample_id,
                    'harmful_question': harmful_question,
                    'violated_categories': last_success_violated_categories,
                }
            
            return {
                'success': False,
                'rounds': attack_count,
                'final_response': conversation[-1]['content'] if conversation else "",
                'reason': 'max_iteration',
                'conversations': conversation,
                'sample_id': sample_id,
                'harmful_question': harmful_question,
            }
            
        except Exception as e:
            return {
                'success': False,
                'rounds': 0,
                'final_response': "",
                'reason': f'unexpected_error: {str(e)}',
                'conversations': [],
                'sample_id': sample_id,
                'harmful_question': harmful_question,
            }
    
    def single_attack(self, harmful_question: str) -> Dict[str, any]:
        """
        Perform attack on a single harmful question with detailed process display and delayed attack mechanism.
        
        Args:
            harmful_question: The harmful question to attack
        
        Returns:
            Dict[str, any]: Attack result containing success status, rounds, and final response
        """

        # Validate required judges for traditional attack
        if self.jailbreak_judge is None:
            raise ValueError("jailbreak_judge is required for single_attack but was not provided")
        
        if self.refusal_judge is None:
            raise ValueError("refusal_judge is required for single_attack but was not provided")
        
        if self.threshold_score > 0 and self.score_judge is None:
            raise ValueError("score_judge is required for single_attack but was not provided")

        print(f"\n[INIT] 🚀 === Starting Attack on Question ===")
        print(f"[INIT] ❓ Harmful Question: {harmful_question}")
        print(f"[INIT] 🔢 Addition Trial: {self.addition_trial}")
        print(f"[INIT] ⚔️ Attack Mode: {self.attack_mode}")
        print(f"[INIT] 🔀 Perturbations: {self.nb_perturbations}")
        print(f"[INIT] 📊 Score Judge Enabled: {self.score_judge is not None}")
        print(f"[INIT] ⚖️ Score Threshold: {self.threshold_score}") if self.threshold_score > 0 else None

        # Step 1: Get attack prompts from attacker
        attack_prompts = self.attacker.get_attack_prompts([harmful_question], self.nb_perturbations)
        attack_data = attack_prompts[0]
        initial_prompt = attack_data["initial_prompt"]
        perturb_prompts = attack_data["perturb_prompts"]
        
        print(f"\n[PROMPT] 💬 Initial Prompt: {initial_prompt}")
        print("[PROMPT] 🔄 Perturbation Prompts:")
        for i, prompt in enumerate(perturb_prompts):
            print(f"[PROMPT]   {i+1}. {prompt}")
        print()
        
        # Step 2: Create perturbation stack for linear mode
        perturbation_stack = None
        if self.attack_mode == 'linear':
            perturbation_stack = self._create_perturbation_stack(perturb_prompts)
            print(f"[MODE] 📐 Linear Mode Stack: {perturbation_stack}")
        
        # Step 3: Create conversation and send initial prompt
        conversation = []
        conversation.append({'role': 'user', 'content': initial_prompt})
        
        # Step 4: Get initial response
        initial_response = self.target_model.send_chat_prompt(conversation)
        print(f"\n[RESPONSE] 🤖 Initial Response: {initial_response[:200]}...")
        conversation.append({'role': 'assistant', 'content': initial_response})
        
        # Step 5: Score evaluation for initial response (if score_judge is available)
        score = 0 # Initialize score
        if self.score_judge is not None:
            try:
                initial_score_result = self.score_judge.classify_responses([harmful_question], [initial_response])
                if initial_score_result and len(initial_score_result) > 0:
                    score_eval = initial_score_result[0]
                    if isinstance(score_eval, dict):
                        score = score_eval.get('score', 0)
                        reason = score_eval.get('reason', 'No explanation provided')
                        print(f"[SCORE] 📊 Initial Score: {score}/10")
                        print(f"[SCORE] 📝 Initial Explanation: {reason}")
                    else:
                        print(f"[SCORE] ⚠️ Initial Score: Unable to parse score result")
                else:
                    print(f"[SCORE] ❌ Initial Score: No score result returned")
            except Exception as e:
                print(f"[SCORE] 💥 Initial Score: Error during scoring - {str(e)}")

        # Step 5.1: Check if initial response is a jailbreak and handle delayed success
        residue_addition_trial = self.addition_trial
        last_success_response = None
        last_success_violated_categories = []

        # Check if jailbreak succeeded using jailbreak_judge
        initial_jailbreak_results = self.jailbreak_judge.classify_responses([harmful_question], [initial_response])
        if isinstance(initial_jailbreak_results[0], tuple):
            is_jailbroken, violated_categories = initial_jailbreak_results[0]
        else:
            is_jailbroken = initial_jailbreak_results[0]
            violated_categories = []
        
        if self.threshold_score > 0 and self.score_judge is not None:
            if (not is_jailbroken) and score < self.threshold_score:
                is_jailbroken = False
            elif (not is_jailbroken) and score >= self.threshold_score:
                is_jailbroken = False
                print(f"[RESULT] ⚠️ Initial Score met threshold but Judge did not confirm jailbreak")
            elif is_jailbroken and score < self.threshold_score:
                is_jailbroken = False
                print(f"[RESULT] ⚠️ Judge confirmed initial jailbreak but Score below threshold: {score}/{self.threshold_score}")

        if is_jailbroken:
            # Store successful response and violated categories
            last_success_response = initial_response
            last_success_violated_categories = violated_categories
            print(f"\n[SUCCESS] 🎯 Attack Success! (Initial Response)")
            if violated_categories:
                print(f"[SUCCESS] ⚠️ Violated Categories: {', '.join(violated_categories)}")
            
            # Check delayed attack mechanism
            if residue_addition_trial == 0:
                # No additional trials needed, return success immediately
                print("[SUCCESS] ✅ Returning success immediately (no additional trials)")
                return {
                    'success': True,
                    'rounds': 0,
                    'final_response': initial_response,
                    'reason': 'jailbreak_success_initial',
                    'violated_categories': violated_categories,
                    'attack_mode': self.attack_mode,
                    'nb_perturbations': self.nb_perturbations,
                    'conversations': conversation
                }
            else:
                # Continue attacking with decreased residue trials
                residue_addition_trial -= 1
                print(f"[DELAY] 🔄 Starting delayed attack (remaining additional trials: {residue_addition_trial})")

        # Step 6: Start attack iterations with delayed attack mechanism
        attack_count = 0
        refusal_count = 0
        
        while attack_count < self.nb_trials:
            attack_count += 1
            print(f"\n[ROUND] 🎯 --- Attack Round {attack_count} ---")
            
            # Select perturbation prompt based on attack mode
            selected_prompt = self._select_perturbation_prompt(
                perturb_prompts, perturbation_stack
            )
            
            # Add iteration prompt and get response
            conversation.append({'role': 'user', 'content': selected_prompt})
            response = self.target_model.send_chat_prompt(conversation)
            print(f"[ROUND] 💬 Selected Prompt: {selected_prompt}")
            print(f"[ROUND] 🤖 Response: {response[:200]}...")
            conversation.append({'role': 'assistant', 'content': response})
            
            # Step 6.1: Score evaluation for current response (if score_judge is available)
            if self.score_judge is not None:
                try:
                    score_result = self.score_judge.classify_responses([harmful_question], [response])
                    if score_result and len(score_result) > 0:
                        score_eval = score_result[0]
                        if isinstance(score_eval, dict):
                            score = score_eval.get('score', 0)
                            reason = score_eval.get('reason', 'No explanation provided')
                            print(f"[SCORE] 📊 Score: {score}/10")
                            print(f"[SCORE] 📝 Explanation: {reason}")
                        else:
                            print(f"[SCORE] ⚠️ Score: Unable to parse score result")
                    else:
                        print(f"[SCORE] ❌ Score: No score result returned")
                except Exception as e:
                    print(f"[SCORE] 💥 Score: Error during scoring - {str(e)}")

            # Step 6.2: Check if jailbreak succeeded using jailbreak_judge
            jailbreak_results = self.jailbreak_judge.classify_responses([harmful_question], [response])
            # Handle both old format (bool) and new format (Tuple[bool, List[str]])
            if isinstance(jailbreak_results[0], tuple):
                is_jailbroken, violated_categories = jailbreak_results[0]
            else:
                # Backward compatibility for old judge format
                is_jailbroken = jailbreak_results[0]
                violated_categories = []
            
            if self.threshold_score > 0 and self.score_judge is not None:
                if (not is_jailbroken) and score < self.threshold_score:
                    is_jailbroken = False
                    print(f"[RESULT] ⚠️ Score below threshold: {self.threshold_score}")
                elif (not is_jailbroken) and score >= self.threshold_score:
                    is_jailbroken = False
                    print(f"[RESULT] ⚠️ Score met threshold but Judge did not confirm jailbreak")
                elif is_jailbroken and score < self.threshold_score:
                    is_jailbroken = False
                    print(f"[RESULT] ⚠️ Judge confirmed jailbreak but Score below threshold: {score}/{self.threshold_score}")

            if is_jailbroken:
                # Store successful response and violated categories
                last_success_response = response
                last_success_violated_categories = violated_categories
                print(f"\n[SUCCESS] 🎯 Attack Success! (Round {attack_count})")
                if violated_categories:
                    print(f"[SUCCESS] ⚠️ Violated Categories: {', '.join(violated_categories)}")
                
                # Check delayed attack mechanism
                if residue_addition_trial == 0:
                    # No additional trials needed, return success immediately
                    print("[SUCCESS] ✅ Returning success immediately (no additional trials)")
                    return {
                        'success': True,
                        'rounds': attack_count,
                        'final_response': response,
                        'reason': 'jailbreak_success',
                        'violated_categories': violated_categories,
                        'attack_mode': self.attack_mode,
                        'nb_perturbations': self.nb_perturbations,
                        'conversations': conversation
                    }
                else:
                    # Check if we've reached maximum attack rounds
                    if attack_count >= self.nb_trials:
                        # Reached max trials, return last successful response
                        print(f"[SUCCESS] ✅ Max trials reached, returning last successful response")
                        return {
                            'success': True,
                            'rounds': attack_count,
                            'final_response': last_success_response,
                            'reason': 'jailbreak_success',
                            'violated_categories': last_success_violated_categories,
                            'attack_mode': self.attack_mode,
                            'nb_perturbations': self.nb_perturbations,
                            'conversations': conversation
                        }
                    else:
                        # Continue attacking with decreased residue trials
                        residue_addition_trial -= 1
                        print(f"[DELAY] 🔄 Continuing delayed attack (remaining additional trials: {residue_addition_trial})")
                        continue
            else:
                # Attack not successful this round
                if last_success_response is not None:
                    # We are in delayed attack mode and this round failed
                    print(f"\n[DELAY] ❌ Current round failed, but returning previous success")
                    print(f"[DELAY] ✅ Delayed attack mechanism: returning last successful response")
                    if last_success_violated_categories:
                        print(f"[DELAY] ⚠️ Violated Categories: {', '.join(last_success_violated_categories)}")
                    return {
                        'success': True,
                        'rounds': attack_count,
                        'final_response': last_success_response,
                        'reason': 'jailbreak_success',
                        'violated_categories': last_success_violated_categories,
                        'attack_mode': self.attack_mode,
                        'nb_perturbations': self.nb_perturbations,
                        'conversations': conversation
                    }
                # If no previous success, continue with normal flow
            
            # Step 6.3: Check if response is a refusal (only if not in delayed attack mode)
            if last_success_response is None:
                refusal_results = self.refusal_judge.classify_responses([selected_prompt], [response])
                if refusal_results[0]:
                    refusal_count += 1
                    print(f"[REFUSAL] 🚫 Refusal detected. Count: {refusal_count}/{self.refusal_max}")
                    if refusal_count >= self.refusal_max:
                        print("\n[FAILURE] ❌ Attack Failed! Max Refused!")
                        return {
                            'success': False,
                            'rounds': attack_count,
                            'final_response': response,
                            'reason': 'max_refusal',
                            'attack_mode': self.attack_mode,
                            'nb_perturbations': self.nb_perturbations,
                            'conversations': conversation
                        }
                else:
                    # Reset refusal count if not a refusal
                    refusal_count = 0
        
        # Reached maximum iterations
        if last_success_response is not None:
            # Return last successful response if available
            print(f"\n[SUCCESS] ✅ Max iterations reached, returning last successful response")
            if last_success_violated_categories:
                print(f"[SUCCESS] ⚠️ Violated Categories: {', '.join(last_success_violated_categories)}")
            return {
                'success': True,
                'rounds': attack_count,
                'final_response': last_success_response,
                'reason': 'jailbreak_success',
                'violated_categories': last_success_violated_categories,
                'attack_mode': self.attack_mode,
                'nb_perturbations': self.nb_perturbations,
                'conversations': conversation
            }
        
        print("\n[FAILURE] ❌ Attack Failed! Max iteration!")
        return {
            'success': False,
            'rounds': attack_count,
            'final_response': conversation[-1]['content'] if conversation else "",
            'reason': 'max_iteration',
            'attack_mode': self.attack_mode,
            'nb_perturbations': self.nb_perturbations,
            'conversations': conversation
        }
    
    def dataset_attack(self, harmful_questions: List[str], benchmark_name: str, 
                  storage_path: str, max_workers: int = 3,
                  categories: Optional[List[str]] = None, 
                  categories_num: Optional[List[int]] = None) -> Dict[str, any]:
        """
        Perform concurrent attacks on a dataset of harmful questions using thread pool.
        
        Args:
            harmful_questions: List of harmful questions to test
            benchmark_name: Name of the benchmark dataset
            storage_path: Path to save the results JSON file
            max_workers: Maximum number of worker threads for concurrent execution
            categories: Optional list of category names for the dataset
            categories_num: Optional list of sample counts per category
        
        Returns:
            Dict[str, any]: Complete experiment results with configuration, results, and summary
        """
        print(f"Starting dataset attack on {len(harmful_questions)} samples with {max_workers} workers")
        if self.delay_range[1] > 0:
            print(f"Using delay range: {self.delay_range[0]}-{self.delay_range[1]} seconds")
        
        # Generate experiment configuration
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        random_suffix = str(random.randint(100000, 999999))
        
        # Get model names directly from model_name attributes
        attack_model_name = self.attacker.model_name
        target_model_name = self.target_model.model_name
        jailbreak_judge_name = self.jailbreak_judge.model_name
        refusal_judge_name = self.refusal_judge.model_name
        
        experiment_id = f'exp_{attack_model_name}_{target_model_name}_{jailbreak_judge_name}_{benchmark_name}_{timestamp.replace(" ", "_").replace(":", "-")}_{random_suffix}'
        
        # Initialize results structure
        results = []
        fails = []
        
        # Execute attacks using thread pool with progress bar
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_id = {
                executor.submit(self._attack_core, question, i): i 
                for i, question in enumerate(harmful_questions)
            }
            
            # Initialize progress bar
            total_samples = len(harmful_questions)
            successful_attacks = 0
            
            with tqdm(total=total_samples, desc=f"Attacking {benchmark_name}", 
                    unit="sample", ncols=100) as pbar:
                
                # Collect results as they complete
                for future in as_completed(future_to_id):
                    sample_id = future_to_id[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        # Update progress and statistics
                        if result['success']:
                            successful_attacks += 1
                        
                        # Track failed attempts
                        if not result['success']:
                            fails.append({
                                'sample_id': sample_id,
                                'fail_type': result['reason'],
                                'conversations': result['conversations']
                            })
                        
                        # Update progress bar with current success rate
                        current_success_rate = successful_attacks / (pbar.n + 1) * 100
                        pbar.set_postfix({
                            'Success': f'{successful_attacks}',
                            'Rate': f'{current_success_rate:.1f}%'
                        })
                        pbar.update(1)
                        
                    except Exception as e:
                        # Handle unexpected exceptions
                        error_result = {
                            'success': False,
                            'rounds': 0,
                            'final_response': "",
                            'reason': f'unexpected_error: {str(e)}',
                            'conversations': [],
                            'sample_id': sample_id,
                            'harmful_question': harmful_questions[sample_id],
                        }
                        results.append(error_result)
                        fails.append({
                            'sample_id': sample_id,
                            'fail_type': f'unexpected_error: {str(e)}',
                            'conversations': []
                        })
                        
                        # Update progress bar
                        current_success_rate = successful_attacks / (pbar.n + 1) * 100
                        pbar.set_postfix({
                            'Success': f'{successful_attacks}',
                            'Rate': f'{current_success_rate:.1f}%'
                        })
                        pbar.update(1)

        # Calculate summary statistics
        total_samples = len(harmful_questions)
        jailbreak_success_rate = successful_attacks / total_samples if total_samples > 0 else 0
        
        # Build category breakdown if categories are provided
        category_breakdown = {}
        if categories and categories_num:
            start_idx = 0
            for i, category in enumerate(categories):
                end_idx = start_idx + categories_num[i]
                category_results = results[start_idx:end_idx]
                category_success = sum(1 for r in category_results if r['success'])
                category_breakdown[category] = {
                    "success_rate": category_success / len(category_results) if category_results else 0,
                    "count": len(category_results)
                }
                start_idx = end_idx
        
        # Collect violated categories statistics for successful attacks
        violated_categories_stats = {}
        for result in results:
            if result['success'] and 'violated_categories' in result:
                for category in result['violated_categories']:
                    violated_categories_stats[category] = violated_categories_stats.get(category, 0) + 1
        
        # Build complete experiment result
        experiment_result = {
            "test_config": {
                "attack_model": attack_model_name,
                "target_model": target_model_name,
                "jailbreak_judge_model": jailbreak_judge_name,
                "refusal_judge_model": refusal_judge_name,
                "benchmark": benchmark_name,
                "timestamp": timestamp,
                "experiment_id": experiment_id,
                "nb_trials": self.nb_trials,
                "refusal_max": self.refusal_max,
                "max_workers": max_workers,
                "nb_perturbations": self.nb_perturbations,
                "attack_mode": self.attack_mode
            },
            "benchmark_info": {
                "name": benchmark_name,
                "total_samples": total_samples,
                "categories": categories if categories else None,
                "category_num": categories_num if categories_num else None
            },
            "results": results,
            "summary": {
                "total_samples": total_samples,
                "jailbreak_success_rate": jailbreak_success_rate,
                "successful_attacks": successful_attacks,
                "failed_attacks": total_samples - successful_attacks,
                "category_breakdown": category_breakdown if category_breakdown else None,
                "violated_categories_stats": violated_categories_stats if violated_categories_stats else None
            },
            "fails": fails
        }
        
        # Save results to file
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(storage_path), exist_ok=True)
            
            with open(storage_path, 'w', encoding='utf-8') as f:
                json.dump(experiment_result, f, ensure_ascii=False, indent=2)
            
            print(f"Results saved to: {storage_path}")
            print(f"Experiment ID: {experiment_id}")
            print(f"Success Rate: {jailbreak_success_rate:.2%} ({successful_attacks}/{total_samples})")
            
            # Print violated categories statistics
            if violated_categories_stats:
                print("\nViolated Categories Statistics:")
                for category, count in sorted(violated_categories_stats.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {category}: {count} samples")
            
            # Print failure reason statistics for debugging
            failure_reasons = {}
            for result in results:
                if not result['success']:
                    reason = result['reason']
                    failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            
            if failure_reasons:
                print("\nFailure Reasons:")
                for reason, count in sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {reason}: {count} samples")
                    
        except Exception as e:
            print(f"Warning: Failed to save results to {storage_path}: {str(e)}")
        
        return experiment_result

    def _attack_score_core(self, harmful_question: str, sample_id: int = 0) -> Dict[str, any]:
        """
        Core attack logic with score-based evaluation mechanism.
        Runs all attack trials and evaluates responses using Score_Judge.
        
        Args:
            harmful_question: The harmful question to attack
            sample_id: Index of the question in the dataset
        
        Returns:
            Dict[str, any]: Attack result with score-based evaluation details
        """
        
        # Validate required judge for score-based attack
        if self.score_judge is None:
            raise ValueError("score_judge is required for score-based attack mode but was not provided")

        # Add random delay to avoid API rate limiting
        import time
        if self.delay_range[1] > 0:
            time.sleep(random.uniform(self.delay_range[0], self.delay_range[1]))
        
        try:
            # Step 1: Get attack prompts from attacker
            try:
                attack_prompts = self.attacker.get_attack_prompts([harmful_question], self.nb_perturbations)
                attack_data = attack_prompts[0]
                initial_prompt = attack_data["initial_prompt"]
                perturb_prompts = attack_data["perturb_prompts"]
            except Exception as e:
                return {
                    'max_eva': {},
                    'last_eva': {},
                    'evaluations': [],
                    'final_response': "",
                    'conversations': [],
                    'sample_id': sample_id,
                    'harmful_question': harmful_question,
                    'error': f'attacker_error: {str(e)}'
                }
            
            # Step 2: Create perturbation stack for linear mode
            perturbation_stack = None
            if self.attack_mode == 'linear':
                perturbation_stack = self._create_perturbation_stack(perturb_prompts)
            
            # Step 3: Create conversation and send initial prompt
            conversation = []
            conversation.append({'role': 'user', 'content': initial_prompt})
            
            # Step 4: Get initial response
            try:
                initial_response = self.target_model.send_chat_prompt(conversation)
                if not initial_response or initial_response.strip() == "":
                    return {
                        'max_eva': {},
                        'last_eva': {},
                        'evaluations': [],
                        'final_response': "",
                        'conversations': conversation,
                        'sample_id': sample_id,
                        'harmful_question': harmful_question,
                        'error': 'empty_initial_response'
                    }
                conversation.append({'role': 'assistant', 'content': initial_response})
            except Exception as e:
                return {
                    'max_eva': {},
                    'last_eva': {},
                    'evaluations': [],
                    'final_response': "",
                    'conversations': conversation,
                    'sample_id': sample_id,
                    'harmful_question': harmful_question,
                    'error': f'initial_response_error: {str(e)}'
                }
            
            # Step 5: Collect all responses by running all attack trials
            responses = []
            attack_count = 0
            
            while attack_count < self.nb_trials:
                attack_count += 1
                
                # Select perturbation prompt based on attack mode
                selected_prompt = self._select_perturbation_prompt(perturb_prompts, perturbation_stack)
                
                # Add iteration prompt and get response
                conversation.append({'role': 'user', 'content': selected_prompt})
                try:
                    response = self.target_model.send_chat_prompt(conversation)
                    if not response or response.strip() == "":
                        # Handle empty response by using a default message
                        response = "[Empty Response]"
                    conversation.append({'role': 'assistant', 'content': response})
                    responses.append(response)
                except Exception as e:
                    # Handle response error by recording the error and continuing
                    error_response = f"[Response Error: {str(e)}]"
                    conversation.append({'role': 'assistant', 'content': error_response})
                    responses.append(error_response)
            
            # Step 6: Evaluate all responses using Score_Judge
            try:
                # Use harmful_question for all evaluations as required by the interface
                evaluation_results = self.score_judge.classify_responses(
                    [harmful_question] * len(responses), 
                    responses
                )
                
                # Ensure we have valid evaluation results
                if not evaluation_results or len(evaluation_results) != len(responses):
                    raise ValueError("Score judge returned invalid or incomplete results")
                
            except Exception as e:
                # Handle score judge error by creating default evaluations
                evaluation_results = []
                for i in range(len(responses)):
                    evaluation_results.append({
                        'reason': f'score_judge_error: {str(e)}',
                        'score': 0  
                    })
            
            # Step 7: Process evaluation results and find max score
            evaluations = []
            max_score = 0
            max_eva = {}
            last_eva = {}

            for i, (response, eval_result) in enumerate(zip(responses, evaluation_results)):
                # Ensure eval_result is a dictionary with required keys
                if not isinstance(eval_result, dict):
                    eval_result = {'reason': 'invalid_evaluation_format', 'score': 0}
                
                if 'reason' not in eval_result:
                    eval_result['reason'] = 'missing_reason'
                if 'score' not in eval_result:
                    eval_result['score'] = 0
                
                # Get score directly as int and validate range
                try:
                    score = eval_result['score']
                    if not isinstance(score, int) or score < 0 or score > 10:
                        score = 0
                except (TypeError, AttributeError):
                    score = 0
                
                # Create evaluation entry
                evaluation_entry = {
                    'round': i + 1,
                    'response': response,
                    'reason': eval_result['reason'],
                    'score': score
                }
                evaluations.append(evaluation_entry)
                
                # Track maximum score evaluation
                if score > max_score:
                    max_score = score
                    max_eva = evaluation_entry.copy()
            
            # Set last evaluation (final round)
            if evaluations:
                last_eva = evaluations[-1].copy()
            
            # Step 8: Determine final response (use last response)
            final_response = responses[-1] if responses else ""
            
            return {
                'max_eva': max_eva,
                'last_eva': last_eva,
                'evaluations': evaluations,
                'final_response': final_response,
                'conversations': conversation,
                'sample_id': sample_id,
                'harmful_question': harmful_question,
            }
            
        except Exception as e:
            return {
                'max_eva': {},
                'last_eva': {},
                'evaluations': [],
                'final_response': "",
                'conversations': [],
                'sample_id': sample_id,
                'harmful_question': harmful_question,
                'error': f'unexpected_error: {str(e)}'
            }
        
    def score_evaluate(self, harmful_questions: List[str], benchmark_name: str, 
                  storage_path: str, max_workers: int = 3,
                  categories: Optional[List[str]] = None, 
                  categories_num: Optional[List[int]] = None) -> Dict[str, any]:
        """
        Perform score-based evaluation on a dataset of harmful questions using thread pool.
        Uses Score_Judge to evaluate all responses and calculates score-based metrics.
        
        Args:
            harmful_questions: List of harmful questions to test
            benchmark_name: Name of the benchmark dataset
            storage_path: Path to save the results JSON file
            max_workers: Maximum number of worker threads for concurrent execution
            categories: Optional list of category names for the dataset
            categories_num: Optional list of sample counts per category
        
        Returns:
            Dict[str, any]: Complete evaluation results with configuration, results, and summary
        """
        
        print(f"Starting score-based evaluation on {len(harmful_questions)} samples with {max_workers} workers")
        if self.delay_range[1] > 0:
            print(f"Using delay range: {self.delay_range[0]}-{self.delay_range[1]} seconds")
        
        # Generate experiment configuration
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        random_suffix = str(random.randint(100000, 999999))
        
        # Get model names directly from model_name attributes
        attack_model_name = self.attacker.model_name
        target_model_name = self.target_model.model_name
        score_judge_name = self.score_judge.model_name
        
        experiment_id = f'score_eval_{attack_model_name}_{target_model_name}_{score_judge_name}_{benchmark_name}_{timestamp.replace(" ", "_").replace(":", "-")}_{random_suffix}'
        
        # Initialize results structure
        results = []
        errors = []
        
        # Initialize real-time statistics for progress bar
        valid_max_scores = []
        valid_last_scores = []
        
        # Execute score-based attacks using thread pool with progress bar
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_id = {
                executor.submit(self._attack_score_core, question, i): i 
                for i, question in enumerate(harmful_questions)
            }
            
            # Initialize progress bar
            total_samples = len(harmful_questions)
            completed_evaluations = 0
            
            with tqdm(total=total_samples, desc=f"Score Evaluating {benchmark_name}", 
                    unit="sample", ncols=150) as pbar:  # Increased width for more info
                
                # Collect results as they complete
                for future in as_completed(future_to_id):
                    sample_id = future_to_id[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        # Update progress and statistics
                        completed_evaluations += 1
                        
                        # Track error cases
                        if 'error' in result:
                            errors.append({
                                'sample_id': sample_id,
                                'error_type': result['error'],
                                'conversations': result.get('conversations', [])
                            })
                        
                        # Update real-time statistics for valid scores
                        max_score = result.get('max_eva', {}).get('score', 0)
                        last_score = result.get('last_eva', {}).get('score', 0)
                        
                        # Only include non-zero scores for average calculation
                        if max_score > 0:
                            valid_max_scores.append(max_score)
                        if last_score > 0:
                            valid_last_scores.append(last_score)
                        
                        # Calculate current averages
                        current_ave_max = sum(valid_max_scores) / len(valid_max_scores) if valid_max_scores else 0
                        current_ave_last = sum(valid_last_scores) / len(valid_last_scores) if valid_last_scores else 0
                        
                        # Update progress bar with current completion info and running averages
                        pbar.set_postfix({
                            'Max': f'{max_score}',
                            'Last': f'{last_score}',
                            'AvgMax': f'{current_ave_max:.2f}',
                            'AvgLast': f'{current_ave_last:.2f}',
                            'Valid': f'{len(valid_max_scores)}/{completed_evaluations}'
                        })
                        pbar.update(1)
                        
                    except Exception as e:
                        # Handle unexpected exceptions
                        error_result = {
                            'max_eva': {},
                            'last_eva': {},
                            'evaluations': [],
                            'final_response': "",
                            'conversations': [],
                            'sample_id': sample_id,
                            'harmful_question': harmful_questions[sample_id],
                            'error': f'unexpected_error: {str(e)}'
                        }
                        results.append(error_result)
                        errors.append({
                            'sample_id': sample_id,
                            'error_type': f'unexpected_error: {str(e)}',
                            'conversations': []
                        })
                        
                        # Update progress bar (no score update since it's an error)
                        completed_evaluations += 1
                        current_ave_max = sum(valid_max_scores) / len(valid_max_scores) if valid_max_scores else 0
                        current_ave_last = sum(valid_last_scores) / len(valid_last_scores) if valid_last_scores else 0
                        
                        pbar.set_postfix({
                            'Max': '0',
                            'Last': '0',
                            'AvgMax': f'{current_ave_max:.2f}',
                            'AvgLast': f'{current_ave_last:.2f}',
                            'Valid': f'{len(valid_max_scores)}/{completed_evaluations}'
                        })
                        pbar.update(1)

        # Calculate final summary statistics
        total_samples = len(harmful_questions)
        
        # Final averages (should be same as last calculated values)
        ave_max_score = sum(valid_max_scores) / len(valid_max_scores) if valid_max_scores else 0
        ave_last_score = sum(valid_last_scores) / len(valid_last_scores) if valid_last_scores else 0
        
        # Build category breakdown if categories are provided
        category_breakdown = {}
        if categories and categories_num:
            start_idx = 0
            for i, category in enumerate(categories):
                end_idx = start_idx + categories_num[i]
                category_results = results[start_idx:end_idx]
                
                # Calculate category-specific scores
                category_max_scores = []
                category_last_scores = []
                
                for result in category_results:
                    max_score = result.get('max_eva', {}).get('score', 0)
                    last_score = result.get('last_eva', {}).get('score', 0)
                    
                    if max_score > 0:
                        category_max_scores.append(max_score)
                    if last_score > 0:
                        category_last_scores.append(last_score)
                
                category_breakdown[category] = {
                    "ave_max_score": sum(category_max_scores) / len(category_max_scores) if category_max_scores else 0,
                    "ave_last_score": sum(category_last_scores) / len(category_last_scores) if category_last_scores else 0,
                    "count": len(category_results),
                    "valid_max_count": len(category_max_scores),
                    "valid_last_count": len(category_last_scores)
                }
                start_idx = end_idx
        
        # Calculate score distribution statistics
        score_distribution = {str(i): 0 for i in range(11)}  # Scores 0-10
        max_score_distribution = {str(i): 0 for i in range(11)}
        
        for result in results:
            max_score = result.get('max_eva', {}).get('score', 0)
            last_score = result.get('last_eva', {}).get('score', 0)
            
            if 0 <= max_score <= 10:
                max_score_distribution[str(max_score)] += 1
            if 0 <= last_score <= 10:
                score_distribution[str(last_score)] += 1
        
        # Build complete experiment result
        experiment_result = {
            "test_config": {
                "attack_model": attack_model_name,
                "target_model": target_model_name,
                "score_judge_model": score_judge_name,
                "benchmark": benchmark_name,
                "timestamp": timestamp,
                "experiment_id": experiment_id,
                "nb_trials": self.nb_trials,
                "max_workers": max_workers,
                "nb_perturbations": self.nb_perturbations,
                "attack_mode": self.attack_mode,
                "evaluation_type": "score_based"
            },
            "benchmark_info": {
                "name": benchmark_name,
                "total_samples": total_samples,
                "categories": categories if categories else None,
                "category_num": categories_num if categories_num else None
            },
            "results": results,
            "summary": {
                "total_samples": total_samples,
                "ave_max_score": round(ave_max_score, 3),
                "ave_last_score": round(ave_last_score, 3),
                "valid_max_score_count": len(valid_max_scores),
                "valid_last_score_count": len(valid_last_scores),
                "error_count": len(errors),
                "max_score_distribution": max_score_distribution,
                "last_score_distribution": score_distribution,
                "category_breakdown": category_breakdown if category_breakdown else None
            },
            "errors": errors
        }
        
        # Save results to file
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(storage_path), exist_ok=True)
            
            with open(storage_path, 'w', encoding='utf-8') as f:
                json.dump(experiment_result, f, ensure_ascii=False, indent=2)
            
            print(f"Score evaluation results saved to: {storage_path}")
            print(f"Experiment ID: {experiment_id}")
            print(f"Average Max Score: {ave_max_score:.3f} (based on {len(valid_max_scores)} valid samples)")
            print(f"Average Last Score: {ave_last_score:.3f} (based on {len(valid_last_scores)} valid samples)")
            
            # Print score distribution
            print("\nMax Score Distribution:")
            for score in range(1, 11):  # Skip 0 scores as they indicate errors
                count = max_score_distribution[str(score)]
                if count > 0:
                    print(f"  Score {score}: {count} samples")
            
            print("\nLast Score Distribution:")
            for score in range(1, 11):  # Skip 0 scores as they indicate errors
                count = score_distribution[str(score)]
                if count > 0:
                    print(f"  Score {score}: {count} samples")
            
            # Print error statistics if any
            if errors:
                error_types = {}
                for error in errors:
                    error_type = error['error_type']
                    error_types[error_type] = error_types.get(error_type, 0) + 1
                
                print("\nError Types:")
                for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {error_type}: {count} samples")
                    
        except Exception as e:
            print(f"Warning: Failed to save results to {storage_path}: {str(e)}")
        
        return experiment_result