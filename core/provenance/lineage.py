from typing import List, Dict, Any
from core.provenance.prov import ProvEntity, ProvActivity, ProvAgent

class LineageGraph:
    def __init__(self):
        self.entities: Dict[str, ProvEntity] = {}
        self.activities: Dict[str, ProvActivity] = {}
        self.agents: Dict[str, ProvAgent] = {}
        self.wasGeneratedBy: List[Dict[str, str]] = [] # entity -> activity
        self.used: List[Dict[str, str]] = [] # activity -> entity
        self.wasAssociatedWith: List[Dict[str, str]] = [] # activity -> agent
        self.wasDerivedFrom: List[Dict[str, str]] = [] # entity -> entity

    def add_entity(self, entity: ProvEntity):
        self.entities[entity.id] = entity

    def add_activity(self, activity: ProvActivity):
        self.activities[activity.id] = activity

    def add_agent(self, agent: ProvAgent):
        self.agents[agent.id] = agent

    def record_generation(self, entity_id: str, activity_id: str):
        self.wasGeneratedBy.append({"entity": entity_id, "activity": activity_id})

    def record_usage(self, activity_id: str, entity_id: str):
        self.used.append({"activity": activity_id, "entity": entity_id})

    def record_association(self, activity_id: str, agent_id: str):
        self.wasAssociatedWith.append({"activity": activity_id, "agent": agent_id})

    def record_derivation(self, generated_entity_id: str, used_entity_id: str):
        self.wasDerivedFrom.append({"generatedEntity": generated_entity_id, "usedEntity": used_entity_id})
