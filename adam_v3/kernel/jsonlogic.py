import json
from json_logic import jsonLogic


class JsonLogicEvaluator:

    def evaluate(self, rule, data):
        return jsonLogic(rule, data)
