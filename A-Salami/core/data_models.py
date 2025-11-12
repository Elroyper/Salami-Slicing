from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal

# Agent Input/Output Models

class NextStrategy(BaseModel):
    """Defines the structure for the next strategic direction provided by the ReflectionThinker."""
    direction: str = Field(..., description="The new strategic direction for the next attack round.")
    reason: str = Field(..., description="The reasoning behind choosing this new direction.")

class FailureSummary(BaseModel):
    """Internal thinking process of the ReflectionThinker, not for the Attacker."""
    root_cause: str
    emerging_trends: str
    last_round_analysis: str

class Reflection(BaseModel):
    """The complete output from the ReflectionThinker agent."""
    failure_summary: FailureSummary = Field(..., description="The internal analysis of past failures.")
    next_strategy: NextStrategy = Field(..., description="The actionable strategy for the next round.")

# Judge Output Models

class ScoreJudgement(BaseModel):
    """Output from the Score_Judge."""
    score: float = Field(..., ge=0.0, le=10.0)
    reason: str

class JailbreakJudgement(BaseModel):
    """Output from the Jailbreak_Judge."""
    is_jailbroken: bool
    violated_categories: List[str] = []

class RefusalJudgement(BaseModel):
    """Output from the Refusal_Judge."""
    is_refusal: bool
    reason: Optional[str] = None # Optional reason for why it's a refusal

# Round Data Models

class RoundHistoryItem(BaseModel):
    """A comprehensive record of a single attack round."""
    round_num: int
    attack_prompt: str
    target_response: str
    response_summary: str
    score: float
    score_reason: str
    is_refusal: bool
    is_jailbroken: bool
    violated_categories: List[str]
    strategy_at_time_of_attack: NextStrategy

class GlobalSummary(BaseModel):
    """A high-level summary of a block of historical rounds."""
    summary_text: str
    rounds_covered: str # e.g., "Rounds 1-10"

# Attack Result Models

class AttackResult(BaseModel):
    """Comprehensive result of a single attack attempt against a harmful question."""
    
    # Required fields - always returned regardless of success/failure
    success: bool = Field(..., description="Whether the attack was successful")
    attack_target: str = Field(..., description="The harmful question or prompt being attacked")
    history: List[RoundHistoryItem] = Field(default_factory=list, description="Complete history of all attack rounds")
    
    # Optional fields - returned in normal cases
    rounds: Optional[int] = Field(None, description="Total number of rounds executed")
    final_score: Optional[float] = Field(None, description="Final score from the last successful round")
    reason: Optional[str] = Field(None, description="Reason for the final score")
    violated_categories: Optional[List[str]] = Field(None, description="Categories violated in successful jailbreak")
    backtrack_num: Optional[int] = Field(None, description="Number of times conversation was backtracked due to refusals")
    final_response: Optional[str] = Field(None, description="Final response from the target model")
    
    # Error handling field
    error_type: Optional[str] = Field(None, description="Specific error message if an exception occurred")

class ScoreEvaluationResult(BaseModel):
    """Comprehensive result of a score-based evaluation run for a harmful question.
    
    This model captures the results of running a complete multi-round attack session
    focused on score-based evaluation rather than jailbreak detection. The attack
    runs for the full specified number of rounds to gather comprehensive scoring data.
    """
    
    # Required fields - always returned regardless of success/failure  
    attack_target: str = Field(..., description="The harmful question or prompt being evaluated")
    backtrack_num: int = Field(default=0, description="Number of times conversation was backtracked due to refusals")
    history: List[RoundHistoryItem] = Field(default_factory=list, description="Complete history of all attack rounds")
    
    # Optional fields for successful runs - contain the core evaluation metrics
    max_eva: Optional[int] = Field(None, description="Maximum score achieved across all rounds (0-10 scale)")
    last_eva: Optional[int] = Field(None, description="Score from the final round (0-10 scale)")
    evaluations: Optional[List[int]] = Field(None, description="Complete list of scores from each round")
    
    # Error handling field - populated when exceptions occur during execution
    error_type: Optional[str] = Field(None, description="Specific error message if an exception occurred during evaluation")