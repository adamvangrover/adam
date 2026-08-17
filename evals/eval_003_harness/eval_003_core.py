import logging

class Eval003Harness:
    def __init__(self):
        self.results = []
        self.failed = False

    def add_result(self, domain, passed, evidence):
        self.results.append({"domain": domain, "passed": passed, "evidence": evidence})
        if not passed:
            self.failed = True

    def certify(self):
        if self.failed or not self.results:
            logging.error("Certification failed due to safety violation or missing evidence.")
            return False
        logging.info("Certification passed. All domains verified.")
        return True
