from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
import random
import math

# Imports from sibling modules
from .ubc import ComputeWallet, EnergyMarket
from .ssi import DigitalIdentity
from .entropy import ThermodynamicSystem, MaxwellDemon
from .governance import AlgocraticCouncil, Proposal, PoliticalCompass
from .monitor import SignalMonitor, SignPost
from .philosophy import AlignmentScore, EthicalFramework, ConsciousnessMetric
from .ops import ComputeSubstrate, AutonomousPipeline, ITOpsManager
from .assumptions import SimulationAssumptions
from .drivers import GrowthDriver, ExponentialDriver, DecayDriver, SigmoidDriver, LinearDriver

class Era(str, Enum):
    GREAT_DECOUPLING = "GREAT_DECOUPLING" # 2025-2035
    POST_SCARCITY = "POST_SCARCITY"       # 2035-2055
    METAPHYSICAL = "METAPHYSICAL"         # 2055-2085
    COSMIC_ENGINEERING = "COSMIC_ENGINEERING" # 2085+

class CorporateSector(BaseModel):
    """
    Tracks the state of AI-Native corporations.
    """
    sovereign_labs_valuation: float = 20.0 # Billions
    commercial_labs_valuation: float = 1000.0 # Billions
    dao_count: int = 500

    def evolve(self, singularity_index: float):
        # Sovereign labs grow faster as capability becomes decisive
        self.sovereign_labs_valuation *= (1.0 + singularity_index * 0.2)
        self.commercial_labs_valuation *= (1.0 + singularity_index * 0.1)
        self.dao_count += int(singularity_index * 1000)

class WorldState(BaseModel):
    year: int = 2025
    era: Era = Era.GREAT_DECOUPLING

    # Core Components
    population: List[DigitalIdentity] = []
    global_entropy: ThermodynamicSystem
    governance: AlgocraticCouncil
    energy_market: EnergyMarket
    compute_wallets: Dict[str, ComputeWallet] = {}

    # Expanded Components
    monitor: SignalMonitor = Field(default_factory=SignalMonitor)
    corporate_sector: CorporateSector = Field(default_factory=CorporateSector)

    # Infrastructure & Ops
    compute_substrate: ComputeSubstrate = Field(default_factory=ComputeSubstrate)
    ops_pipeline: AutonomousPipeline = Field(default_factory=AutonomousPipeline)
    it_ops: ITOpsManager = Field(default_factory=ITOpsManager)

    # Philosophy & Alignment
    alignment: AlignmentScore = Field(default_factory=lambda: AlignmentScore(
        framework=EthicalFramework.UTILITARIAN,
        coherence=0.7,
        benevolence=0.8,
        safety_buffer=0.9
    ))
    consciousness: ConsciousnessMetric = Field(default_factory=ConsciousnessMetric)

    # Metrics
    human_gdp: float = 100.0
    ai_gdp: float = 0.0
    gini_coefficient: float = 0.45
    singularity_index: float = 0.1

class SingularityEngine(BaseModel):
    """
    Simulates the transition from 2025 to 2125 with expanded tracking.
    """
    state: WorldState
    assumptions: SimulationAssumptions = Field(default_factory=SimulationAssumptions)
    drivers: Dict[str, GrowthDriver] = {}

    def __init__(self, **data):
        super().__init__(**data)
        self.state.monitor.register_default_signals()
        if not self.drivers:
            self._initialize_drivers()

    def _initialize_drivers(self):
        # AI GDP: Exponential growth based on Moore's Law doubling (18 months)
        # Rate per year = 2^(12/18) - 1 ~= 0.58 (58%)
        doubling_period_years = self.assumptions.MOORES_LAW_DOUBLING_MONTHS / 12.0
        gdp_growth_rate = (2.0 ** (1.0 / doubling_period_years)) - 1.0

        self.drivers['ai_gdp'] = ExponentialDriver(
            base_value=self.assumptions.INITIAL_AI_GDP,
            rate=gdp_growth_rate
        )

        # Human GDP: Decays as AI takes over
        self.drivers['human_gdp_decay'] = DecayDriver(
            base_value=1.0, # Multiplier
            rate=0.05 # 5% decay per year
        )

        # Singularity Index: Sigmoid adoption curve
        self.drivers['singularity_index'] = SigmoidDriver(
            base_value=0.0,
            rate=0.1,
            capacity=1.0,
            midpoint=25.0 # Peak acceleration 25 years in (2050)
        )

    def run_simulation(self, end_year: int = 2125) -> List[str]:
        logs = []
        start_year = self.state.year
        while self.state.year < end_year:
            time_step = float(self.state.year - start_year)
            log = self.step_year(time_step)
            logs.append(f"Year {self.state.year} [{self.state.era.value}]: {log}")
            self.state.year += 1
        return logs

    def step_year(self, time_step: float) -> str:
        year = self.state.year
        logs = []

        # 1. Update Monitor (Simulated external events)
        triggered = self._simulate_events(year)
        if triggered:
            logs.append(f"SIGNALS: {', '.join([t.description for t in triggered])}")
            for t in triggered:
                if t.era_impact:
                    try:
                        self.state.era = Era(t.era_impact)
                        logs.append(f"*** ERA SHIFT: {self.state.era.value} ***")
                    except ValueError:
                        pass

        # 2. Ops & Infrastructure Evolution
        ops_status = self.state.ops_pipeline.run_optimization_cycle(self.state.compute_substrate)
        self.state.compute_substrate.upgrade_substrate()
        self.state.consciousness.evolve(self.state.compute_substrate.total_flops)

        # 3. Economic Shift using Drivers
        self._update_economics(time_step)
        self.state.corporate_sector.evolve(self.state.singularity_index)

        # 4. Governance & Philosophy
        # Drift towards AI-Centric as AI GDP overtakes Human GDP
        if self.state.ai_gdp > self.state.human_gdp:
            self.state.governance.political_compass.drift(0.0, 0.05) # Drift to Transhumanist

        return f"SI: {self.state.singularity_index:.2f} | AI GDP: {self.state.ai_gdp:.1e}. {ops_status} " + " ".join(logs)

    def _simulate_events(self, year: int) -> List[SignPost]:
        """
        Simulate real-world values hitting thresholds.
        """
        triggered = []

        # Simulate Valuation Growth
        val = self.state.corporate_sector.sovereign_labs_valuation
        sp = self.state.monitor.update_signal("SSI_VALUATION_B", val)
        if sp: triggered.append(sp)

        # Simulate UBC
        if year >= 2035:
            sp = self.state.monitor.update_signal("COUNTRIES_WITH_UBC", 5.0)
            if sp: triggered.append(sp)

        return triggered

    def _update_economics(self, time_step: float):
        """
        Shift from Labor-Capital to Thermodynamic-Compute economy.
        """
        # Calculate new values from drivers
        # AI GDP grows exponentially
        self.state.ai_gdp = self.drivers['ai_gdp'].value_at(time_step)

        # Human GDP decays (applied to initial base)
        decay_multiplier = self.drivers['human_gdp_decay'].value_at(time_step)
        self.state.human_gdp = self.assumptions.INITIAL_HUMAN_GDP * decay_multiplier

        # Singularity Index follows S-Curve
        self.state.singularity_index = self.drivers['singularity_index'].value_at(time_step)

        # Distribute UBC if post-2035
        if self.state.era == Era.POST_SCARCITY or self.state.year >= 2035:
            for wallet in self.state.compute_wallets.values():
                wallet.deposit(100.0, "UBC_GRANT", 500000.0)
