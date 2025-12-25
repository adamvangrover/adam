import unittest
from core.future_state.ubc import ComputeWallet, ComputeCredit
from core.future_state.ssi import DigitalIdentity, VerifiableCredential
from core.future_state.entropy import ThermodynamicSystem, MaxwellDemon
from core.future_state.governance import QuadraticVoting, Proposal, PoliticalCompass
from core.future_state.monitor import SignalMonitor, SignPost
from core.future_state.philosophy import AlignmentScore, EthicalFramework, ConsciousnessMetric
from core.future_state.ops import ComputeSubstrate, AutonomousPipeline
from core.future_state.drivers import ExponentialDriver, DecayDriver
from core.future_state.assumptions import SimulationAssumptions
from core.future_state.engine import SingularityEngine, WorldState, EnergyMarket, AlgocraticCouncil

class TestFutureState(unittest.TestCase):

    def test_ubc_wallet(self):
        wallet = ComputeWallet(owner_id="citizen_1", balance=0.0)
        wallet.deposit(100.0, "UBI", 1000.0)
        self.assertEqual(wallet.balance, 100.0)
        self.assertTrue(wallet.spend(50.0, "API_CALL"))
        self.assertEqual(wallet.balance, 50.0)

    def test_ssi_identity(self):
        did = DigitalIdentity()
        cred = VerifiableCredential(
            issuer="gov",
            subject_id=did.did,
            claims={"is_human": "true"}
        )
        did.add_credential(cred)
        self.assertTrue(did.is_verified_human)

    def test_entropy_system(self):
        system = ThermodynamicSystem(total_energy_joules=1000.0)
        demon = MaxwellDemon()
        demon.sort_resources(system, energy_input=10.0)
        self.assertGreater(system.organized_information_bits, 0.0)

    def test_quadratic_voting(self):
        qv = QuadraticVoting(voice_credits_balance=100)
        proposal = Proposal(id="p1", description="Test")
        self.assertTrue(qv.vote(proposal, 3, "for"))
        self.assertEqual(qv.voice_credits_balance, 91)

    def test_signal_monitor(self):
        monitor = SignalMonitor()
        sp = SignPost(id="test", description="Test", metric="VAL", threshold=100.0)
        monitor.active_sign_posts.append(sp)

        triggered = monitor.update_signal("VAL", 50.0)
        self.assertIsNone(triggered)

        triggered = monitor.update_signal("VAL", 100.0)
        self.assertIsNotNone(triggered)
        self.assertTrue(sp.is_triggered)

    def test_drivers(self):
        # Test Exponential
        driver = ExponentialDriver(base_value=100.0, rate=0.1)
        val = driver.value_at(1.0)
        self.assertAlmostEqual(val, 110.0)

        # Test Decay
        driver = DecayDriver(base_value=100.0, rate=0.1)
        val = driver.value_at(1.0)
        self.assertAlmostEqual(val, 90.0)

    def test_engine_initialization(self):
        thermo = ThermodynamicSystem(total_energy_joules=100.0)
        world = WorldState(global_entropy=thermo, governance=AlgocraticCouncil(), energy_market=EnergyMarket())
        engine = SingularityEngine(state=world)
        self.assertIn("ai_gdp", engine.drivers)
        self.assertIn("singularity_index", engine.drivers)

if __name__ == "__main__":
    unittest.main()
