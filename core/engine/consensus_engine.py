import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

# Graceful fallback for Unix-specific file locking
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False
    logging.warning("fcntl module not found. File locking will be disabled (expected on Windows).")

class ConsensusEngine:
    """
    Algorithmic Arbitrator for resolving conflicting Agent outputs.
    Evaluates 'Bullish' vs 'Bearish' (or similar) outputs from multiple agents
    using weighted confidence scores based on risk profiles.
    Escalates to Human-In-The-Loop (HITL) on high-conviction dealocks.
    Includes persistent file logging and narrative logger injection.
    """
    
    def __init__(self, threshold: float = 0.5, narrative_logger=None):
        logging.info("Initializing ConsensusEngine Arbitrator...")
        self.threshold = threshold
        self.decision_log: List[Dict[str, Any]] = []
        self.narrative_logger = narrative_logger 
        
        # Base multipliers based on predefined risk profiles
        self.risk_profiles = {
            "aggressive": {"FundamentalAnalystAgent": 1.2, "MarketSentimentAgent": 1.1, "RiskAssessmentAgent": 0.8},
            "moderate":   {"FundamentalAnalystAgent": 1.0, "MarketSentimentAgent": 1.0, "RiskAssessmentAgent": 1.0},
            "conservative":{"FundamentalAnalystAgent": 0.8, "MarketSentimentAgent": 0.9, "RiskAssessmentAgent": 1.5}
        }
        
    def _calculate_base_conviction(self, outputs: List[Dict[str, Any]], risk_profile_name: str) -> Dict[str, float]:
        """
        Calculates the risk-adjusted conviction scores for each unique viewpoint.
        """
        profile = self.risk_profiles.get(risk_profile_name, self.risk_profiles["moderate"])
        
        # Tally the weighted confidence for each viewpoint (e.g. "BULLISH", "BEARISH")
        conviction_tally = {}
        for out in outputs:
            agent_source = out.get("agent_source", "UnknownAgent")
            raw_confidence = float(out.get("confidence", 0.0))
            
            # Unify answers
            ans = str(out.get("answer", out.get("vote", "UNKNOWN"))).upper()
            if ans in ['APPROVE', 'BUY', 'LONG', 'YES', 'BULLISH']:
                viewpoint = 'BULLISH'
            elif ans in ['REJECT', 'SELL', 'SHORT', 'NO', 'BEARISH']:
                viewpoint = 'BEARISH'
            else:
                viewpoint = ans
                
            out['normalized_viewpoint'] = viewpoint
            
            # Apply profile weight
            profile_weight = profile.get(agent_source, 1.0)
            
            # Allow individual vote weight override, defaulting to 1.0
            vote_weight = float(out.get("weight", 1.0))
            
            total_weight = profile_weight * vote_weight
            adjusted_score = raw_confidence * total_weight
            
            out['adjusted_score'] = adjusted_score
            
            conviction_tally[viewpoint] = conviction_tally.get(viewpoint, 0.0) + adjusted_score
            logging.info(f"Consensus Check: [{agent_source}] voted {viewpoint} with adjusted power {adjusted_score:.2f} (Base: {raw_confidence}, Weight: {total_weight:.2f})")
            
        return conviction_tally

    def evaluate(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compatibility wrapper for test suite evaluating logic.
        """
        total = 0.0
        for s in signals:
            vote = str(s.get('vote', '')).upper()
            w = float(s.get('weight', 1.0))
            c = float(s.get('confidence', 0.0))
            multiplier = 1.0 if vote in ['BUY', 'APPROVE', 'LONG', 'YES'] else -1.0
            total += (multiplier * c * w)
        score = total / len(signals) if signals else 0.0

        if score > self.threshold:
            decision = 'BUY/APPROVE'
        elif score < -self.threshold:
            decision = 'SELL/REJECT'
        else:
            decision = 'HOLD'

        rationale = " | ".join(
            [f"{s.get('agent', 'Unknown')} voted {s.get('vote', '')}: {s.get('reason', '')}" for s in signals]
        )

        return {'decision': decision, 'score': score, 'rationale': rationale}

    def arbitrate(self, agent_outputs: List[Dict[str, Any]], risk_profile: str = "moderate") -> Dict[str, Any]:
        """
        Receives a list of dicts (simulating AgentOutputs) and arbitrates the final decision.
        Integrates risk profile weighting and numeric-based scoring constraints.
        """
        if not agent_outputs:
            return self._finalize_decision({"decision": "NO_DATA", "confidence": 0, "requires_human_review": True, "reason": "No outputs provided."})
            
        logging.info(f"\n--- ARBITRATING {len(agent_outputs)} CONFLICTING OUTPUTS (Profile: {risk_profile.upper()}) ---")
        
        # 1. Calculate Weighted Tally
        tally = self._calculate_base_conviction(agent_outputs, risk_profile)
        total_conviction_power = sum(tally.values())
        
        if total_conviction_power == 0:
            return self._finalize_decision({"decision": "ERROR", "requires_human_review": True, "reason": "Zero confidence across all agents."})
            
        # 2. Determine Voting Shares
        shares = {view: (score / total_conviction_power) for view, score in tally.items()}
        
        # Sort descending by share
        sorted_views = sorted(shares.items(), key=lambda item: item[1], reverse=True)
        top_view, top_share = sorted_views[0]
        
        # 3. Deadlock Detection (Human-In-The-Loop Escalation)
        if len(sorted_views) > 1:
            runner_up_view, runner_up_share = sorted_views[1]
            if top_share < 0.60:
                warning_msg = f"HITL DEADLOCK ESCALATION: High conviction deadlock between {top_view} ({top_share*100:.1f}%) and {runner_up_view} ({runner_up_share*100:.1f}%)."
                logging.warning(warning_msg)
                
                return self._finalize_decision({
                    "decision": "DEADLOCK",
                    "requires_human_review": True,
                    "confidence": top_share,
                    "reason": warning_msg,
                    "breakdown": shares,
                    "agent_narratives": agent_outputs
                })
                
        # 4. Numerized evaluation check against threshold
        # We also enforce that the top share's dominance is mathematically significant enough
        normalized_score = top_share - (1.0 - top_share) # rough net score metric
        
        if normalized_score < self.threshold and len(sorted_views) > 1:
            # Failed the threshold requirement, treat as soft deadlock
            msg = f"Consensus score {normalized_score:.2f} failed to pass threshold {self.threshold}. Resolving to HOLD."
            return self._finalize_decision({
                "decision": "HOLD",
                "requires_human_review": False,
                "confidence": top_share,
                "reason": msg,
                "breakdown": shares,
                "agent_narratives": agent_outputs
            })

        # 5. Clear Resolution
        success_msg = f"ALGORITHMIC CONSENSUS REACHED: {top_view} wins with {top_share*100:.1f}% conviction share (Net Score: {normalized_score:.2f})."
        logging.info(success_msg)
        return self._finalize_decision({
            "decision": top_view,
            "requires_human_review": False,
            "confidence": top_share,
            "reason": success_msg,
            "breakdown": shares,
            "agent_narratives": agent_outputs
        })
        
    def _finalize_decision(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Final step wrapper to ensure logging happens on all code paths.
        """
        # Append Timestamp if not present
        if "timestamp" not in result:
            result["timestamp"] = datetime.utcnow().isoformat()
            
        self._log_decision(result)
        return result

    def _log_decision(self, result: Dict[str, Any]):
        """Append to internal log with environment-agnostic fallbacks."""
        self.decision_log.append(result)

        # 1. Persistence Logic with Graceful Degradation
        try:
            # Step up 3 levels from core/engine/consensus_engine.py to reach the root
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            data_dir = os.path.join(base_dir, 'data')
            log_path = os.path.join(data_dir, 'decision_log.jsonl')

            os.makedirs(data_dir, exist_ok=True)

            with open(log_path, 'a') as f:
                if HAS_FCNTL:
                    fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(result) + "\n")
                finally:
                    if HAS_FCNTL:
                        fcntl.flock(f, fcntl.LOCK_UN)

            # Protocol ADAM-V-NEXT Additive requirement: Generate decision_log.json
            json_log_path = os.path.join(data_dir, 'decision_log.json')
            with open(json_log_path, 'w') as f:
                if HAS_FCNTL:
                    fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    json.dump(self.decision_log, f, indent=4)
                finally:
                    if HAS_FCNTL:
                        fcntl.flock(f, fcntl.LOCK_UN)

        except PermissionError:
            logging.error("Permission denied when writing to decision log.")
        except Exception as e:
            logging.error(f"Failed to persist decision log: {e}")

        # 2. Advanced LLM Environment Logging with Static Fallback
        if hasattr(self, 'narrative_logger') and self.narrative_logger is not None:
            try:
                self.narrative_logger.log_narrative(
                    event="Consensus Evaluation",
                    analysis=result.get('reason', ''),
                    decision=result['decision'],
                    outcome=f"Score: {result.get('confidence', 0)}"
                )
            except AttributeError:
                logging.warning("Narrative logger provided lacks 'log_narrative' method.")
            except Exception as e:
                logging.error(f"Narrative logging failed: {e}")
        else:
            # Fallback for static/headless execution
            # We already log heavily in arbitrate, but we can do a final debug dump
            logging.debug(f"Static Execution check. Rationale: {result.get('reason')}")

    def get_log(self) -> List[Dict[str, Any]]:
        return self.decision_log

    def get_decision_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve the most recent decisions."""
        return self.decision_log[-limit:]
