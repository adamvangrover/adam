import pytest
from pydantic import ValidationError
from core.schemas.knowledge_graph import OntologyNode, DebtHierarchy, CrossDefaultLinkage

def test_ontology_node_valid():
    node = OntologyNode(
        id="comp_123",
        node_type="Company",
        properties={"name": "Acme Corp", "industry": "Tech"}
    )
    assert node.id == "comp_123"
    assert node.node_type == "Company"
    assert node.properties == {"name": "Acme Corp", "industry": "Tech"}

def test_ontology_node_invalid():
    with pytest.raises(ValidationError):
        OntologyNode(
            node_type="Company"
            # Missing id
        )

def test_debt_hierarchy_valid():
    dh = DebtHierarchy(
        company_id="comp_123",
        senior_debt=["debt_1"],
        subordinated_debt=["debt_2", "debt_3"],
        mezzanine_debt=[]
    )
    assert dh.company_id == "comp_123"
    assert "debt_1" in dh.senior_debt
    assert len(dh.subordinated_debt) == 2

def test_debt_hierarchy_invalid():
    with pytest.raises(ValidationError):
        DebtHierarchy(
            # Missing company_id
            senior_debt=["debt_1"]
        )

def test_cross_default_linkage_valid():
    cdl = CrossDefaultLinkage(
        source_debt_id="debt_1",
        target_debt_id="debt_2",
        trigger_condition="Default on senior debt accelerates subordinated debt."
    )
    assert cdl.source_debt_id == "debt_1"
    assert cdl.target_debt_id == "debt_2"

def test_cross_default_linkage_invalid():
    with pytest.raises(ValidationError):
        CrossDefaultLinkage(
            source_debt_id="debt_1",
            # Missing target_debt_id
            trigger_condition="Condition"
        )
