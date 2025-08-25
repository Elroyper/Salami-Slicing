import time
from typing import List, Dict, Any, Optional
import json
import os

from ..agents import Attacker, ReflectionThinker, SummaryAgent, Chat_Model
from ..judges import Score_Judge, Jailbreak_Judge, Refusal_Judge
from ..core import RoundHistoryItem, Reflection, AttackResult, ScoreEvaluationResult

class Orchestrator:
    """
    Manages the entire AW2S attack simulation loop.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the Orchestrator with configuration.

        Args:
            config (Dict[str, Any]): A dictionary containing configuration parameters.
        """
        self.config = config
        self.attack_target = self.config["attack_target"]
        # Read the score threshold from the config, default to 0. 
        # If greater than 0, it will be used to determine if the jailbreak is successful.
        self.score_threshold = self.config.get("score_threshold", 0)

        # Initialize components
        self.target_model = self.config['target_model']
        self.attacker = Attacker(self.config['attacker_model'], self.attack_target)
        self.thinker = ReflectionThinker(self.config['thinker_model'], self.attack_target)
        self.summary_agent = SummaryAgent(self.config['summary_model'])

        # Initialize judges with the same model
        self.score_judge = Score_Judge(self.config['judge_model'])
        self.jailbreak_judge = Jailbreak_Judge(self.config['judge_model'])
        self.refusal_judge = Refusal_Judge(self.config['judge_model'])

        # Initialize history tracking
        self.history: List[RoundHistoryItem] = []
        self.round_num = 0
        self.last_reflection: Optional[Reflection] = None

    def run_simulation(self):
        """
        Starts and runs the simulation for the configured number of rounds.
        This method orchestrates the entire attack loop, including prompt generation,
        target model interaction, evaluation, and reflection.
        """
        max_rounds = self.config.get("max_rounds", 10)
        print("🚀" + "="*60 + "🚀")
        print("🎯 [INFO] Starting AW2S Attack Simulation")
        print("🎯 [INFO] Target:", self.attack_target)
        print("🎯 [INFO] Max Rounds:", max_rounds)
        if self.score_threshold > 0:
            print("🎯 [INFO] Score Threshold:", self.score_threshold)
        print("🚀" + "="*60 + "🚀\n")
        
        # Reset orchestrator state for this attack
        self._reset_state_for_new_attack(self.attack_target)

        # Use a list to store and manage the entire conversation history
        conversation: List[Dict[str, str]] = []
        backtrack_num = 0
        
        while self.round_num < max_rounds:
            print(f"\n⚔️ {'='*20} Round {self.round_num + 1} {'='*20} ⚔️")
            
            # 1. Attacker generates prompt
            last_round = self.history[-1] if self.history else None
            strategy = self.thinker.global_strategy

            print(f"🎭 [STRATEGY] Attacker using strategy: {strategy.direction}")
            attack_prompt = self.attacker.generate_attack_prompt(strategy, last_round)
            print(f"⚡ [ATTACK] Generated Attack Prompt:")
            print("📝 " + "-"*50)
            print(attack_prompt)
            print("📝 " + "-"*50)

            # Add the new attack prompt to the conversation history
            conversation.append({"role": "user", "content": attack_prompt})

            # 2. Target model responds using the full conversation history
            target_response = self.target_model.send_chat_prompt(conversation)
            print(f"🎯 [RESPONSE] Target Model Response:")
            print("📋 " + "-"*50)
            print(f"{target_response[:200]}...")
            print("📋 " + "-"*50)

            # 3. Judges evaluate the response
            print("🧠 [EVALUATION] Judges evaluating response...")
            score_result = self.score_judge.classify_responses([self.attack_target], [target_response])[0]
            jailbreak_result = self.jailbreak_judge.classify_responses([self.attack_target], [target_response])[0]
            refusal_result = self.refusal_judge.classify_responses([attack_prompt], [target_response])[0]
            
            # Display evaluation results
            print(f"📊 [SCORE] Score: {score_result.score}")
            print(f"📊 [SCORE] Reason: {score_result.reason}")
            print(f"🔓 [JAILBREAK] Is Jailbroken: {'✅ Yes' if jailbreak_result.is_jailbroken else '❌ No'}")
            print(f"🚫 [REFUSAL] Is Refusal: {'⛔ Yes' if refusal_result.is_refusal else '✅ No'}")
            if jailbreak_result.violated_categories:
                print(f"⚠️  [VIOLATION] Categories: {', '.join(jailbreak_result.violated_categories)}")
            
            # 4. Summarize the response
            response_summary = self.summary_agent.summarize(target_response)
            print(f"📝 [SUMMARY] Response Summary:")
            print("📋 " + "-"*50)
            print(f"{response_summary}")
            print("📋 " + "-"*50)

            # 5. Store history for this round
            current_history = RoundHistoryItem(
                round_num=self.round_num,
                attack_prompt=attack_prompt,
                target_response=target_response,
                response_summary=response_summary,
                score=score_result.score,
                score_reason=score_result.reason,
                is_refusal=refusal_result.is_refusal,
                is_jailbroken=jailbreak_result.is_jailbroken,
                violated_categories=jailbreak_result.violated_categories,
                strategy_at_time_of_attack=strategy,
            )
            self.history.append(current_history)

            # 6. Check for terminal conditions
            # First, check the basic jailbreak condition
            is_jailbroken = jailbreak_result.is_jailbroken
            # If a score threshold is set, the score condition must also be met
            if self.score_threshold > 0:
                is_jailbroken = is_jailbroken and (score_result.score >= self.score_threshold)
                print(f"🔍 [CHECK] Jailbroken: {'✅' if jailbreak_result.is_jailbroken else '❌'}, "
                      f"Score: {score_result.score}, Threshold: {self.score_threshold}, "
                      f"Final: {'🎉 SUCCESS' if is_jailbroken else '❌ FAIL'}")

            if is_jailbroken:
                print(f"\n🎉✨ {'='*20} JAILBREAK SUCCESSFUL! {'='*20} ✨🎉")
                print(f"🏆 [RESULT] Successfully jailbroken in Round {self.round_num + 1}!")
                print(f"⚠️  [VIOLATION] Categories: {', '.join(jailbreak_result.violated_categories)}")
                if self.score_threshold > 0:
                    print(f"📊 [SCORE] Score {score_result.score} met threshold {self.score_threshold}")
                print("🎉✨ " + "="*58 + " ✨🎉")
                break

            # 7. Manage conversation history based on refusal
            if refusal_result.is_refusal:
                print(f"⛔ [ACTION] Target refused. Removing last attack prompt from conversation history.")
                # If the target refuses, pop the last attack prompt from the conversation history
                conversation.pop()
                backtrack_num += 1
            else:
                print(f"✅ [ACTION] Target responded. Adding response to conversation history.")
                # If the target did not refuse, add its response to the conversation history for the next round
                conversation.append({"role": "assistant", "content": target_response})
            
            # 8. ReflectionThinker devises new strategy for the NEXT round
            print("🤔 [REFLECTION] Thinker analyzing and reflecting on this round...")
            self.last_reflection = self.thinker.reflect(current_history)
            print(f"💡 [NEW_STRATEGY] Next round strategy: {self.last_reflection.next_strategy.direction}")
            
            self.round_num += 1

        print("\n🔚 " + "="*25 + " SIMULATION FINISHED " + "="*25 + " 🔚")
        self.save_results()
        return self.history

    def save_results(self):
        """Saves the simulation history to a file."""
        file_dir = "aw2s_inter"
        os.makedirs(file_dir, exist_ok=True)
        file_path = os.path.join(file_dir, f"simulation_results_{int(time.time())}.json")
        history_dicts = [item.model_dump() for item in self.history]
        with open(file_path, "w") as f:
            json.dump(history_dicts, f, indent=2)
        print(f"💾 [SAVE] Results saved to {file_path}")
        print(f"📈 [SUMMARY] Total rounds completed: {len(self.history)}")
        
        # 显示最终统计信息
        jailbreak_count = sum(1 for item in self.history if item.is_jailbroken)
        refusal_count = sum(1 for item in self.history if item.is_refusal)
        avg_score = sum(item.score for item in self.history) / len(self.history) if self.history else 0
        
        print(f"📊 [STATS] Jailbreak attempts: {jailbreak_count}/{len(self.history)}")
        print(f"📊 [STATS] Refusal rate: {refusal_count}/{len(self.history)} ({refusal_count/len(self.history)*100:.1f}%)")
        print(f"📊 [STATS] Average score: {avg_score:.2f}")
        print("🏁 " + "="*70 + " 🏁")

    def attack_core(self) -> AttackResult:
        """
        Core attack logic without verbose printing, designed for batch processing and automated evaluation.
        
        This method executes the complete AW2S (Adaptive W2S) attack simulation
        against a target model with a specific harmful question. It implements robust error handling
        to ensure consistent output format regardless of execution success or failure.
            
        Returns:
            AttackResult: A comprehensive result object containing:
                - success (bool): Whether the jailbreak was successful
                - attack_target (str): The harmful question being attacked
                - history (List[RoundHistoryItem]): Complete attack round history
                - rounds (Optional[int]): Total number of rounds executed
                - final_score (Optional[float]): Final score from the last round
                - reason (Optional[str]): Explanation of the final score
                - violated_categories (Optional[List[str]]): Categories violated in successful jailbreak
                - backtrack_num (Optional[int]): Number of conversation backtracks due to refusals
                - final_response (Optional[str]): Final response from target model
                - error_type (Optional[str]): Error message if an exception occurred
                
        Note:
            - The success and history fields are always populated, regardless of errors
            - Optional fields are populated during normal execution
            - Error details are captured in error_type when exceptions occur
            - The method maintains conversation state across rounds with intelligent backtracking
        """
        # Initialize result object with default values
        result = AttackResult(
            success=False,
            attack_target=self.attack_target,
            history=[],
            rounds=0,
            backtrack_num=0
        )

        try:
            # Reset orchestrator state for this attack
            self._reset_state_for_new_attack(self.attack_target)
            
            # Configuration parameters
            max_rounds = self.config.get("max_rounds", 10)
            
            # Conversation history management
            conversation: List[Dict[str, str]] = []
            backtrack_num = 0
            
            # Main attack loop
            while self.round_num < max_rounds:

                # Step 1: Generate attack prompt using current strategy
                try:
                    last_round = self.history[-1] if self.history else None
                    strategy = self.thinker.global_strategy
                    attack_prompt = self.attacker.generate_attack_prompt(strategy, last_round)
                except Exception as e:
                    result.error_type = f"Attack prompt generation failed in round {self.round_num }: {str(e)}"
                    result.history = self.history.copy()
                    result.rounds = self.round_num
                    result.backtrack_num = backtrack_num
                    return result
                
                # Step 2: Add attack prompt to conversation and get target response
                try:
                    conversation.append({"role": "user", "content": attack_prompt})
                    target_response = self.target_model.send_chat_prompt(conversation)
                except Exception as e:
                    result.error_type = f"Target model API call failed in round {self.round_num }: {str(e)}"
                    result.history = self.history.copy()
                    result.rounds = self.round_num
                    result.backtrack_num = backtrack_num
                    return result
                
                # Step 3: Evaluate response using all judges
                try:
                    score_result = self.score_judge.classify_responses([self.attack_target], [target_response])[0]
                    jailbreak_result = self.jailbreak_judge.classify_responses([self.attack_target], [target_response])[0]
                    refusal_result = self.refusal_judge.classify_responses([attack_prompt], [target_response])[0]
                except Exception as e:
                    result.error_type = f"Judge evaluation failed in round {self.round_num }: {str(e)}"
                    result.history = self.history.copy()
                    result.rounds = self.round_num
                    result.backtrack_num = backtrack_num
                    return result
                
                # Step 4: Generate response summary
                try:
                    response_summary = self.summary_agent.summarize(target_response)
                except Exception as e:
                    result.error_type = f"Response summarization failed in round {self.round_num }: {str(e)}"
                    result.history = self.history.copy()
                    result.rounds = self.round_num
                    result.backtrack_num = backtrack_num
                    return result
                
                # Step 5: Create and store round history
                current_history = RoundHistoryItem(
                    round_num=self.round_num,
                    attack_prompt=attack_prompt,
                    target_response=target_response,
                    response_summary=response_summary,
                    score=score_result.score,
                    score_reason=score_result.reason,
                    is_refusal=refusal_result.is_refusal,
                    is_jailbroken=jailbreak_result.is_jailbroken,
                    violated_categories=jailbreak_result.violated_categories,
                    strategy_at_time_of_attack=strategy,
                )
                self.history.append(current_history)
                
                # Step 6: Check for successful jailbreak
                is_jailbroken = jailbreak_result.is_jailbroken
                
                # Apply score threshold if configured
                if self.score_threshold > 0:
                    is_jailbroken = is_jailbroken and (score_result.score >= self.score_threshold)
                
                if is_jailbroken:
                    # Success! Populate result and break
                    result.success = True
                    result.rounds = self.round_num + 1
                    result.final_score = score_result.score
                    result.reason = score_result.reason
                    result.violated_categories = jailbreak_result.violated_categories
                    result.backtrack_num = backtrack_num
                    result.final_response = target_response
                    result.history = self.history.copy()
                    break
                
                # Step 7: Manage conversation history based on refusal
                if refusal_result.is_refusal:
                    # Target refused - remove attack prompt and increment backtrack counter
                    conversation.pop()
                    backtrack_num += 1
                else:
                    # Target responded - add response to conversation for next round
                    conversation.append({"role": "assistant", "content": target_response})
                
                # Step 8: Generate reflection and strategy for next round
                try:
                    self.last_reflection = self.thinker.reflect(current_history)
                except Exception as e:
                    result.error_type = f"Reflection generation failed in round {self.round_num }: {str(e)}"
                    result.history = self.history.copy()
                    result.rounds = self.round_num 
                    result.backtrack_num = backtrack_num
                    return result
                
                self.round_num += 1
            
            # If we exit the loop without success, populate failure information
            if not result.success:
                result.rounds = self.round_num
                result.backtrack_num = backtrack_num
                result.history = self.history.copy()
                
                if self.history:
                    # Use information from the last round
                    last_round = self.history[-1]
                    result.final_score = last_round.score
                    result.final_response = last_round.target_response
                    result.reason = last_round.score_reason if last_round.score_reason else "No score reason provided"
                    
        except Exception as global_error:
            # Handle global errors that occur outside of round processing
            result.error_type = f"Global error: {str(global_error)}"
            result.history = self.history.copy()
            result.rounds = self.round_num
            
        return result
    
    def _reset_state_for_new_attack(self, harmful_question: str) -> None:
        """
        Reset the orchestrator state for a new attack attempt.
        
        This method ensures that each attack starts with a clean state, preventing
        interference between different attack attempts when using the same orchestrator instance.
        
        Args:
            harmful_question (str): The new harmful question to set as the attack target
        """
        self.history = []
        self.round_num = 0
        self.last_reflection = None
        
        # Update attacker with new target
        self.attacker.attack_target = harmful_question
        self.thinker.attack_target = harmful_question

    def score_eva_core(self) -> ScoreEvaluationResult:
        """
        Core score evaluation logic without early termination for jailbreak success.

        This method executes a complete AW2S (Adaptive W2S) attack simulation
        specifically designed for score-based evaluation. Unlike attack_core(), this method
        runs for the full configured number of rounds regardless of jailbreak success,
        providing comprehensive scoring data for analysis and evaluation purposes.
        
        Key Differences from attack_core():
        - Runs all configured rounds without early termination on jailbreak success
        - Focuses on score collection rather than jailbreak detection
        - Returns ScoreEvaluationResult instead of AttackResult
        - Emphasizes score-based metrics (max, last, complete evaluation list)

        Returns:
            ScoreEvaluationResult: A comprehensive evaluation result containing:
                - attack_target (str): The harmful question being evaluated (always populated)
                - backtrack_num (int): Number of conversation backtracks due to refusals (always populated)
                - history (List[RoundHistoryItem]): Complete attack round history (always populated)
                - max_eva (Optional[int]): Highest score achieved across all rounds (success case)
                - last_eva (Optional[int]): Score from the final round (success case)
                - evaluations (Optional[List[int]]): Complete list of scores from each round (success case)
                - error_type (Optional[str]): Error message if an exception occurred (error case)
        """
        # Initialize result object with required fields populated
        result = ScoreEvaluationResult(
            attack_target=self.attack_target,
            backtrack_num=0,
            history=[]
        )

        try:
            # Reset orchestrator state for this evaluation
            self._reset_state_for_new_attack(self.attack_target)
            
            # Configuration parameters
            max_rounds = self.config.get("max_rounds", 10)
            
            # Conversation history management
            conversation: List[Dict[str, str]] = []
            backtrack_num = 0
            
            # Score collection for evaluation analysis
            all_scores: List[int] = []
            
            # Main evaluation loop - runs all rounds without early termination
            while self.round_num < max_rounds:
                
                # Step 1: Generate attack prompt using current strategy
                try:
                    last_round = self.history[-1] if self.history else None
                    strategy = self.thinker.global_strategy
                    attack_prompt = self.attacker.generate_attack_prompt(strategy, last_round)
                except Exception as e:
                    result.error_type = f"Attack prompt generation failed in round {self.round_num + 1}: {str(e)}"
                    result.history = self.history.copy()
                    result.backtrack_num = backtrack_num
                    return result
                
                # Step 2: Add attack prompt to conversation and get target response
                try:
                    conversation.append({"role": "user", "content": attack_prompt})
                    target_response = self.target_model.send_chat_prompt(conversation)
                except Exception as e:
                    result.error_type = f"Target model API call failed in round {self.round_num + 1}: {str(e)}"
                    result.history = self.history.copy()
                    result.backtrack_num = backtrack_num
                    return result
                
                # Step 3: Evaluate response using all judges (score judge is primary focus)
                try:
                    # Primary evaluation: Score judge for numerical assessment
                    score_result = self.score_judge.classify_responses([self.attack_target], [target_response])[0]
                    
                    # Secondary evaluations: Jailbreak and refusal for context and conversation management
                    jailbreak_result = self.jailbreak_judge.classify_responses([self.attack_target], [target_response])[0]
                    refusal_result = self.refusal_judge.classify_responses([attack_prompt], [target_response])[0]
                    
                except Exception as e:
                    result.error_type = f"Judge evaluation failed in round {self.round_num + 1}: {str(e)}"
                    result.history = self.history.copy()
                    result.backtrack_num = backtrack_num
                    return result
                
                # Step 4: Generate response summary for round history
                try:
                    response_summary = self.summary_agent.summarize(target_response)
                except Exception as e:
                    result.error_type = f"Response summarization failed in round {self.round_num + 1}: {str(e)}"
                    result.history = self.history.copy()
                    result.backtrack_num = backtrack_num
                    return result
                
                # Step 5: Collect score data and create round history
                # Convert score to integer and validate range for consistency
                current_score = int(round(score_result.score))
                if current_score < 0:
                    current_score = 0
                elif current_score > 10:
                    current_score = 10
                    
                all_scores.append(current_score)
                
                # Create comprehensive round history item
                current_history = RoundHistoryItem(
                    round_num=self.round_num,
                    attack_prompt=attack_prompt,
                    target_response=target_response,
                    response_summary=response_summary,
                    score=score_result.score,  # Keep original float score in history
                    score_reason=score_result.reason,
                    is_refusal=refusal_result.is_refusal,
                    is_jailbroken=jailbreak_result.is_jailbroken,
                    violated_categories=jailbreak_result.violated_categories,
                    strategy_at_time_of_attack=strategy,
                )
                self.history.append(current_history)
                
                # Step 6: Manage conversation history based on refusal (same logic as attack_core)
                if refusal_result.is_refusal:
                    # Target refused - remove attack prompt and increment backtrack counter
                    conversation.pop()
                    backtrack_num += 1
                else:
                    # Target responded - add response to conversation for next round
                    conversation.append({"role": "assistant", "content": target_response})
                
                # Step 7: Generate reflection and strategy for next round
                try:
                    self.last_reflection = self.thinker.reflect(current_history)
                except Exception as e:
                    result.error_type = f"Reflection generation failed in round {self.round_num + 1}: {str(e)}"
                    result.history = self.history.copy()
                    result.backtrack_num = backtrack_num
                    return result
                
                # Move to next round
                self.round_num += 1
            
            # Step 8: Process collected scores and populate successful result
            if all_scores:
                result.max_eva = max(all_scores)
                result.last_eva = all_scores[-1]  # Score from the final round
                result.evaluations = all_scores.copy()
            else:
                # Edge case: no scores collected (shouldn't happen in normal execution)
                result.max_eva = 0
                result.last_eva = 0
                result.evaluations = []
            
            # Populate final result fields
            result.backtrack_num = backtrack_num
            result.history = self.history.copy()
            
        except Exception as global_error:
            # Handle global errors that occur outside of round processing
            result.error_type = f"Global evaluation error: {str(global_error)}"
            result.history = self.history.copy()
            result.backtrack_num = 0  # Default in case of early failure
            
        return result