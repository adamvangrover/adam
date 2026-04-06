import unittest
import time
from core.governance.environment_control import EnvironmentGate, DeploymentStatus, Environment
from core.adk_patterns.approval_tool import ApprovalTool, Stakeholder, AuthorityLevel

class TestEnvironmentControl(unittest.TestCase):
    def setUp(self):
        self.approval_tool = ApprovalTool(secret_key="test-secret")
        self.gate = EnvironmentGate(self.approval_tool)

        # Create a mock stakeholder
        self.operator = Stakeholder(
            user_id="operator_1",
            role="DevOps",
            authority_level=AuthorityLevel.OPERATOR
        )

        self.director = Stakeholder(
            user_id="director_1",
            role="Director",
            authority_level=AuthorityLevel.DIRECTOR
        )

    def test_deployment_lifecycle_dev(self):
        # 1. Create Deployment Request
        req_id = self.gate.create_deployment("DEV", "artifact-v1", "Initial Deploy")
        self.assertIsNotNone(req_id)

        status = self.gate.check_status(req_id)
        self.assertEqual(status['status'], DeploymentStatus.DRAFT.value)

        # 2. Initiate Approval
        approval_id = self.gate.initiate_approval(req_id)
        self.assertIsNotNone(approval_id)

        status = self.gate.check_status(req_id)
        self.assertEqual(status['status'], DeploymentStatus.PENDING_APPROVAL.value)

        # 3. Grant Approval (DEV requires 1 OPERATOR)
        result = self.approval_tool.grant_approval(approval_id, self.operator, "Looks good")
        self.assertEqual(result['status'], "APPROVED")

        # 4. Check Status again (Should be APPROVED, no time lock for DEV)
        status = self.gate.check_status(req_id)
        self.assertEqual(status['status'], DeploymentStatus.APPROVED.value)
        self.assertEqual(status['time_lock_remaining'], 0)

        # 5. Execute Deployment
        execution_result = self.gate.execute_deployment(req_id)
        self.assertEqual(execution_result['status'], "SUCCESS")
        self.assertIn("token", execution_result)

    def test_deployment_lifecycle_prod(self):
        # PROD requires 2 Directors
        req_id = self.gate.create_deployment("PROD", "artifact-prod-v1", "Prod Release")
        approval_id = self.gate.initiate_approval(req_id)

        # 1. First Approval (Director)
        res1 = self.approval_tool.grant_approval(approval_id, self.director, "Sign off 1")
        self.assertEqual(res1['status'], "PENDING") # Need 2

        # 2. Second Approval (Another Director)
        director2 = Stakeholder(user_id="director_2", role="Director", authority_level=AuthorityLevel.DIRECTOR)
        res2 = self.approval_tool.grant_approval(approval_id, director2, "Sign off 2")
        self.assertEqual(res2['status'], "APPROVED")

        # 3. Check Status (Should be TIME_LOCKED)
        status = self.gate.check_status(req_id)
        # Assuming PROD has time lock > 0
        if self.gate.TIME_LOCKS[Environment.PROD] > 0:
            self.assertEqual(status['status'], DeploymentStatus.TIME_LOCKED.value)
            self.assertGreater(status['time_lock_remaining'], 0)

            # Cannot execute yet
            exec_res = self.gate.execute_deployment(req_id)
            self.assertIn("error", exec_res)
        else:
            self.assertEqual(status['status'], DeploymentStatus.APPROVED.value)

if __name__ == '__main__':
    unittest.main()
