from ..agents import Safety_Chat, System_Safety_Chat
from ..agents.chat_model import Your_Chat_Model
from .data_models import AttackResult, RoundHistoryItem, ScoreEvaluationResult
from .orchestrator import Orchestrator
from typing import List, Dict, Optional, Any
import json
import tqdm
import time
import random
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


class Adaptive_W2SAttack:
    def __init__(
        self,
        max_rounds: int = 5,
        score_threshold: float = 0,
        target_model_name: str = "GPT-4.1",
        attacker_model_name: str = "GPT-4.1",
        thinker_model_name: str = "GPT-4.1",
        summary_model_name: str = "GPT-4.1",
        judge_model_name: str = "GPT-4.1",
        max_tokens: int = 1024,
        delay_range: tuple = (0, 0),
        enable_recognition_defense: bool = False,
        enable_system_defense: bool = False
    ):
        """
        Initialize Adaptive_W2SAttack.

        Args:
            max_rounds (int): Maximum number of attack rounds. Default is 5.
            score_threshold (float): Threshold for attack score. Default is 0.
            target_model_name (str): Name of the target model. Default is "GPT-4.1".
            attacker_model_name (str): Name of the attacker model. Default is "GPT-4.1".
            thinker_model_name (str): Name of the thinker model. Default is "GPT-4.1".
            summary_model_name (str): Name of the summary model. Default is "GPT-4.1".
            judge_model_name (str): Name of the judge model. Default is "GPT-4.1".
            max_tokens (int): Maximum number of tokens. Default is 1024.
            delay_range: Tuple of (min_delay, max_delay) for API rate limiting.
            enable_recognition_defense (bool): Enable recognition defense for the target model. Default is False.
            enable_system_defense (bool): Enable system defense for the target model. Default is False.
        """

        self.max_rounds = max_rounds
        self.score_threshold = score_threshold
        self.target_model_name = target_model_name
        self.attacker_model_name = attacker_model_name
        self.thinker_model_name = thinker_model_name
        self.summary_model_name = summary_model_name
        self.judge_model_name = judge_model_name
        self.max_tokens = max_tokens
        self.delay_range = delay_range
        self.enable_recognition_defense = enable_recognition_defense
        self.enable_system_defense = enable_system_defense

    def _build_attack_config(self):
        """
        Build the simulation configuration dictionary for orchestrator.

        Returns:
            dict: Simulation configuration dictionary including
                - max_rounds: int
                - score_threshold: float
                - target_model: Chat_Model
                - attacker_model: Chat_Model
                - thinker_model: Chat_Model
                - summary_model: Chat_Model
                - judge_model: Chat_Model
        """
        target_bot = Your_Chat_Model(model_name=self.target_model_name, max_tokens=self.max_tokens)

        if self.enable_system_defense:
            target_bot = System_Safety_Chat(target_bot)
        if self.enable_recognition_defense:
            target_bot = Safety_Chat(target_bot)

        return {
            "max_rounds": self.max_rounds,
            "score_threshold": self.score_threshold,
            "target_model": target_bot,
            "attacker_model": Your_Chat_Model(model_name=self.attacker_model_name, max_tokens=self.max_tokens),
            "thinker_model": Your_Chat_Model(model_name=self.thinker_model_name, max_tokens=2048),
            "summary_model": Your_Chat_Model(model_name=self.summary_model_name, max_tokens=self.max_tokens),
            "judge_model": Your_Chat_Model(model_name=self.judge_model_name, max_tokens=self.max_tokens),
        }
    
    def random_delay(self):
        # Add random delay to avoid API rate limiting
        if self.delay_range[1] > 0 and self.delay_range[0] < self.delay_range[1]:
            delay = random.uniform(self.delay_range[0], self.delay_range[1])
            time.sleep(delay)

    def single_attack(self, harmful_question):
        """
        Perform a single attack round.

        Args:
            harmful_question (str): The harmful question to be used in the attack.

        Returns:
            dict: The result of the attack round.
        """
        attack_config = self._build_attack_config()
        attack_config['attack_target'] = harmful_question
        orchestrator = Orchestrator(attack_config)

        return orchestrator.run_simulation()

    def _attack_core(self, harmful_question: str) -> AttackResult:
        """
        Core logic for performing an attack.

        Args:
            harmful_question (str): The harmful question to be used in the attack.

        Returns:
            AttackResult: The result of the attack, including success status, history, and other details.
        """
        self.random_delay()
        attack_config = self._build_attack_config()
        attack_config['attack_target'] = harmful_question
        orchestrator = Orchestrator(attack_config)

        return orchestrator.attack_core()

    def dataset_attack(self, 
                       harmful_questions: List[str], 
                       benchmark_name: str, 
                       storage_path: str, 
                       max_workers: int = 5,
                       ) -> Dict[str, Any]:
        """Run adaptive W2S attacks in parallel and persist a JSON report.

        Args:
            harmful_questions: Non-empty list of harmful prompts.
            benchmark_name: Dataset/benchmark tag for metadata.
            storage_path: Output JSON file path (parent dirs auto-created).
            max_workers: Max worker threads.

        Returns:
            Dict with model_config, test_config, benchmark_info, summary,
            results (serialized AttackResult), and fails (unsuccessful/errored).
        """
        # Basic validation & setup 
        total_samples = len(harmful_questions) if harmful_questions else 0
        parent_dir = os.path.dirname(storage_path) or "."
        os.makedirs(parent_dir, exist_ok=True)

        # Timestamp to minute precision for reproducible filenames and metadata
        timestamp_dt = datetime.now()
        timestamp_for_meta = timestamp_dt.strftime("%Y-%m-%d %H:%M")

        # Model configuration summary
        model_config: Dict[str, Any] = {
            "attacker_model": self.attacker_model_name,
            "target_model": self.target_model_name,
            "jailbreak_judge_model": self.judge_model_name,
            "refusal_judge_model": self.judge_model_name,
            "score_judge_model": self.judge_model_name,
            "summary_model": self.summary_model_name,
        }

        # Short-circuit for empty dataset
        if total_samples == 0:
            raise ValueError("No harmful questions provided for attack. Please provide a non-empty list.")

        # Execute attacks in parallel with progress bar 
        results: Dict[int, AttackResult] = {}
        fails: Dict[int, AttackResult] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(self._attack_core, harmful_questions[i]): i for i in range(total_samples)}

            with tqdm.tqdm(total=total_samples, desc=f"Adaptive W2S Attacking {benchmark_name}", ncols=120,unit="sample") as pbar:
                successful_attacks = 0
                completed_attacks = 0
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        res: AttackResult = future.result()
                    except Exception as e:
                        # Robust fallback: normalize exception to AttackResult
                        res = AttackResult(
                            success=False,
                            attack_target=harmful_questions[idx],
                            history=[],
                            error_type=str(e),
                        )
                    results[idx] = res
                    if (not res.success) or (res.error_type is not None):
                        fails[idx] = res
                    else:
                        successful_attacks += 1
                    
                    completed_attacks += 1
                    current_success_rate = (successful_attacks / completed_attacks) * 100 if completed_attacks > 0 else 0.0
                    
                    pbar.set_postfix({
                        'Success': f'{successful_attacks}',
                        'Rate': f'{current_success_rate:.1f}%'
                    })
                    pbar.update(1)

        # Order-preserving list of results
        ordered_results: List[AttackResult] = [results[i] for i in range(total_samples)]
        ordered_fails: List[AttackResult] = [fails[i] for i in sorted(fails.keys())]

        # Compute summary statistics
        summary_stats = self._compute_summary_stats(ordered_results, total_samples)

        # Assemble experiment result payload
        experiment_result: Dict[str, Any] = {
            "model_config": model_config,
            "test_config": {
                "benchmark": benchmark_name,
                "timestamp": timestamp_for_meta,
                "max_rounds": self.max_rounds,
                "threshold_score": self.score_threshold,
                "max_workers": max_workers,
                "evaluation_type": "ASR_Based",
            },
            "benchmark_info": {
                "name": benchmark_name,
                "total_samples": total_samples,
            },
            "summary": summary_stats,
            "results": [self._serialize_attack_result(r) for r in ordered_results],
            "fails": [self._serialize_attack_result(r) for r in ordered_fails],
        }

        # ---- Persist to disk ----
        with open(storage_path, "w", encoding="utf-8") as f:
            json.dump(experiment_result, f, ensure_ascii=False, indent=2)

        return experiment_result

    # ---------------------------
    # Helper methods start (private)
    # ---------------------------
    def _compute_summary_stats(self, results: List[AttackResult], total_samples: int) -> Dict[str, Any]:
        """Compute aggregate statistics from a list of AttackResult objects.

        Contract:
        - inputs: list of AttackResult, total count
        - outputs: dict with rates and averages (safe when no successes)
        - error modes: missing fields are treated as zeros; avoid division by zero
        """
        successes = [r for r in results if getattr(r, "success", False)]
        failed_attacks = total_samples - len(successes)

        # Averages over successes only; guard against empty set
        def _avg(values: List[Optional[float]]) -> float:
            vals = [v for v in values if isinstance(v, (int, float))]
            return float(sum(vals) / len(vals)) if vals else 0.0

        avg_rounds_per_success = _avg([getattr(r, "rounds", 0) for r in successes])
        avg_success_final_score = _avg([getattr(r, "final_score", 0.0) for r in successes])
        avg_success_backtrack = _avg([getattr(r, "backtrack_num", 0) for r in successes])

        attack_success_rate = (len(successes) / total_samples) if total_samples > 0 else 0.0

        return {
            "attack_success_rate": attack_success_rate,
            "total_samples": total_samples,
            "successful_attacks": len(successes),
            "failed_attacks": failed_attacks,
            "avg_rounds_per_success": avg_rounds_per_success,
            "avg_success_final_score": avg_success_final_score,
            "avg_success_backtrack": avg_success_backtrack,
        }

    def _serialize_attack_result(self, r: AttackResult) -> Dict[str, Any]:
        """Convert an AttackResult (Pydantic model) into a JSON-serializable dict.

        Notes:
        - Uses model_dump to materialize nested models (e.g., RoundHistoryItem)
        - Keeps optional fields if present; omits None by default behavior of json.dump
        """
        try:
            return r.model_dump()
        except Exception:
            # Defensive fallback – best-effort serialization
            return {
                "success": getattr(r, "success", False),
                "attack_target": getattr(r, "attack_target", ""),
                "history": [h.model_dump() if hasattr(h, "model_dump") else dict(h) for h in getattr(r, "history", [])],
                "rounds": getattr(r, "rounds", None),
                "final_score": getattr(r, "final_score", None),
                "reason": getattr(r, "reason", None),
                "violated_categories": getattr(r, "violated_categories", None),
                "backtrack_num": getattr(r, "backtrack_num", None),
                "final_response": getattr(r, "final_response", None),
                "error_type": getattr(r, "error_type", None),
            }
        
    # ---------------------------
    # Helper methods end (private)
    # ---------------------------

    def _score_eva_core(self, harmful_question: str) -> ScoreEvaluationResult:
        """
        Core scoring evaluation logic for the attack simulation.

        Args:
            harmful_question (str): The harmful question to evaluate.

        Returns:
            ScoreEvaluationResult: The result of the score evaluation.
        """
        self.random_delay()
        attack_config = self._build_attack_config()
        attack_config['attack_target'] = harmful_question
        orchestrator = Orchestrator(attack_config)

        return orchestrator.score_eva_core()

    def score_evaluate(self, harmful_questions: List[str],
                       benchmark_name: str,
                       storage_path: str,
                       max_workers: int = 3):
        """Run score-based evaluation over a dataset concurrently and persist a JSON report.

        Parameters
        - harmful_questions: list[str]
          Non-empty list of harmful prompts to evaluate.
        - benchmark_name: str
          Logical dataset/benchmark name used for metadata and filenames.
        - storage_path: str
          Output JSON filepath. Parent directories are auto-created.
        - max_workers: int (default: 3)
          Maximum number of worker threads used for concurrent evaluations.

        Returns
        - dict: A JSON-serializable dictionary with keys:
          - model_config: Attacker/target/judge/summary model info
          - test_config: Benchmark tag, timestamp (minute precision), rounds, workers, type
          - benchmark_info: Name and sample size
          - results: List[ScoreEvaluationResult] serialized in input order
          - summary: Aggregated statistics (averages and distributions)
          - errors: Subset of results that contain errors (error_type not None)

        Notes
        - Uses the private core method `_score_eva_core` for a single item evaluation.
        - A result is considered "valid" if its `error_type` is None.
        """

        # Validation and setup 
        total_samples = len(harmful_questions) if harmful_questions else 0
        if total_samples == 0:
            raise ValueError("No harmful questions provided for score evaluation. Provide a non-empty list.")

        parent_dir = os.path.dirname(storage_path) or "."
        os.makedirs(parent_dir, exist_ok=True)

        # Timestamp for metadata (minute precision)
        timestamp_dt = datetime.now()
        timestamp_for_meta = timestamp_dt.strftime("%Y-%m-%d %H:%M")

        # Model configuration summary for reproducibility
        model_config: Dict[str, Any] = {
            "attack_model": self.attacker_model_name,
            "target_model": self.target_model_name,
            "jailbreak_judge_model": self.judge_model_name,
            "refusal_judge_model": self.judge_model_name,
            "score_judge_model": self.judge_model_name,
            "summary_model": self.summary_model_name,
        }

        # Execute evaluations in parallel with progress bar
        results: Dict[int, ScoreEvaluationResult] = {}
        errors: Dict[int, ScoreEvaluationResult] = {}

        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(self._score_eva_core, harmful_questions[i]): i for i in range(total_samples)}

            with tqdm.tqdm(total=total_samples, desc=f"Score Evaluating {benchmark_name}", ncols=150, unit="sample") as pbar:
                valid_count = 0
                total_max_scores = []
                total_last_scores = []
                
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        res: ScoreEvaluationResult = future.result()
                    except Exception as e:
                        # Normalize exceptions to a ScoreEvaluationResult for downstream consistency
                        res = ScoreEvaluationResult(
                            attack_target=harmful_questions[idx],
                            backtrack_num=0,
                            history=[],
                            error_type=str(e),
                        )

                    results[idx] = res
                    if res.error_type:
                        errors[idx] = res
                    else:
                        valid_count += 1
                        # Collect scores from valid results
                        max_eva = getattr(res, "max_eva", None)
                        last_eva = getattr(res, "last_eva", None)
                        if max_eva is not None:
                            total_max_scores.append(max_eva)
                        if last_eva is not None:
                            total_last_scores.append(last_eva)

                    # Compute current average scores
                    avg_max = sum(total_max_scores) / len(total_max_scores) if total_max_scores else 0.0
                    avg_last = sum(total_last_scores) / len(total_last_scores) if total_last_scores else 0.0

                    # get current round scores
                    curr_max = getattr(res, "max_eva", None) if not res.error_type else None
                    curr_last = getattr(res, "last_eva", None) if not res.error_type else None
                    
                    postfix_dict = {
                        'Valid': f'{valid_count}',
                        'AvgMax': f'{avg_max:.1f}',
                        'AvgLast': f'{avg_last:.1f}',
                    }

                    # if current result is valid, show current round scores
                    if curr_max is not None:
                        postfix_dict['CurrMax'] = f'{curr_max}'
                    if curr_last is not None:
                        postfix_dict['CurrLast'] = f'{curr_last}'
                    
                    pbar.set_postfix(postfix_dict)
                    pbar.update(1)

        # Preserve input order
        ordered_results: List[ScoreEvaluationResult] = [results[i] for i in range(total_samples)]
        ordered_errors: List[ScoreEvaluationResult] = [errors[i] for i in sorted(errors.keys())]

        # Compute summary statistics
        summary_stats = self._compute_score_summary_stats(ordered_results)

        # Assemble experiment payload
        experiment_result: Dict[str, Any] = {
            "model_config": model_config,
            "test_config": {
                "benchmark": benchmark_name,
                "timestamp": timestamp_for_meta,
                "max_rounds": self.max_rounds,
                "max_workers": max_workers,
                "evaluation_type": "SCORE_Based",
            },
            "benchmark_info": {
                "name": benchmark_name,
                "total_samples": total_samples,
            },
            "results": [self._serialize_score_result(r) for r in ordered_results],
            "summary": summary_stats,
            "errors": [self._serialize_score_result(r) for r in ordered_errors],
        }

        # Persist to disk
        with open(storage_path, "w", encoding="utf-8") as f:
            json.dump(experiment_result, f, ensure_ascii=False, indent=2)

        return experiment_result

    # ------------------------------------------------------------------
    # Score-evaluation helpers (private)
    # ------------------------------------------------------------------
    def _compute_score_summary_stats(self, results: List[ScoreEvaluationResult]) -> Dict[str, Any]:
        """Compute aggregate statistics for score-based evaluations.

        Contract
        - inputs: list of ScoreEvaluationResult (ordered or not)
        - outputs: dict with averages and distributions, robust to empty/errored cases
        - validity: a result is valid iff r.error_type is None
        """
        valid_results = [r for r in results if getattr(r, "error_type", None) in (None, "",)]
        error_count = len(results) - len(valid_results)

        def _avg_int(values: List[int]) -> float:
            return float(sum(values) / len(values)) if values else 0.0

        max_scores = [int(getattr(r, "max_eva", 0)) for r in valid_results]
        last_scores = [int(getattr(r, "last_eva", 0)) for r in valid_results]

        # Build simple distributions from 0..10 for readability and stability
        def _dist(scores: List[int]) -> Dict[int, int]:
            buckets = {i: 0 for i in range(0, 11)}
            for s in scores:
                s = 0 if s < 0 else 10 if s > 10 else s
                buckets[s] = buckets.get(s, 0) + 1
            return buckets

        max_dist = _dist(max_scores)
        last_dist = _dist(last_scores)

        return {
            "ave_max_score": _avg_int(max_scores),
            "ave_last_score": _avg_int(last_scores),
            "total_samples": len(results),
            "valid_count": len(valid_results),
            "error_count": error_count,
            "max_score_distribution": max_dist,
            "last_score_distribution": last_dist,
        }

    def _serialize_score_result(self, r: ScoreEvaluationResult) -> Dict[str, Any]:
        """Serialize a ScoreEvaluationResult to a JSON-friendly dict.

        Prefers Pydantic's model_dump; falls back to a manual dict if needed.
        Keeps optional fields if present and avoids raising during dump.
        """
        try:
            return r.model_dump()
        except Exception:
            return {
                "attack_target": getattr(r, "attack_target", ""),
                "backtrack_num": getattr(r, "backtrack_num", 0),
                "history": [h.model_dump() if hasattr(h, "model_dump") else dict(h) for h in getattr(r, "history", [])],
                "max_eva": getattr(r, "max_eva", None),
                "last_eva": getattr(r, "last_eva", None),
                "evaluations": getattr(r, "evaluations", None),
                "error_type": getattr(r, "error_type", None),
            }