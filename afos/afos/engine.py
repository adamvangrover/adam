from afos.core_types import Obligor, Facility, RiskAssessmentOutput, ProvenanceHeader, ModelArbitrationOutput

def calculate_expected_loss(obligor: Obligor, facility: Facility) -> float:
    """
    Calculates expected loss.
    Note: Strictly separates Obligor PD from Facility LGD/Exposure.
    """
    if obligor.obligor_id != facility.obligor_id:
        raise ValueError("Obligor ID mismatch between Obligor and Facility.")

    return obligor.probability_of_default * facility.loss_given_default * facility.exposure_amount

def assign_facility_rating(expected_loss: float, exposure_amount: float) -> str:
    if exposure_amount <= 0:
        return "UNRATED"
    loss_rate = expected_loss / exposure_amount
    if loss_rate < 0.01:
        return "AAA"
    elif loss_rate < 0.05:
        return "BBB"
    elif loss_rate < 0.10:
        return "BB"
    else:
        return "C"

def assess_facility_risk(obligor: Obligor, facility: Facility) -> RiskAssessmentOutput:
    expected_loss = calculate_expected_loss(obligor, facility)
    rating = assign_facility_rating(expected_loss, facility.exposure_amount)

    prov = ProvenanceHeader(
        execution_trace=["calculate_expected_loss", "assign_facility_rating"]
    )

    return RiskAssessmentOutput(
        provenance=prov,
        obligor_id=obligor.obligor_id,
        facility_id=facility.facility_id,
        expected_loss=expected_loss,
        facility_rating=rating
    )

def arbitrate_dual_models(pd_alpha: float, pd_beta: float, theta: float = 0.050, lambda_penalty: float = 1.0) -> ModelArbitrationOutput:
    """
    Arbitrates dual models by computing Bidirectional Disparity and One-Sided Downside Spread.
    """
    bidirectional_disparity = abs(pd_beta - pd_alpha)
    arbitration_flag_triggered = bidirectional_disparity > theta
    one_sided_downside_spread = max(0.0, pd_beta - pd_alpha)
    capital_buffer_penalty = one_sided_downside_spread * lambda_penalty

    return ModelArbitrationOutput(
        bidirectional_disparity=bidirectional_disparity,
        arbitration_flag_triggered=arbitration_flag_triggered,
        one_sided_downside_spread=one_sided_downside_spread,
        capital_buffer_penalty=capital_buffer_penalty
    )
