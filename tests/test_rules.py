from panvizplus.rules import load_ruleset


def test_ruleset_has_required_provenance():
    rules = load_ruleset()
    meta = rules["metadata"]

    assert meta["id"] == "panvizplus_v1"
    assert meta["status"] == "draft"
    assert meta["exact_biovia_reproduction"] is False


def test_foundation_rules_present():
    interactions = load_ruleset()["interactions"]

    expected = {
        "conventional_hbond",
        "carbon_hbond",
        "attractive_charge",
        "pi_cation",
        "pi_pi",
        "alkyl",
        "pi_alkyl",
        "unfavorable_vdw_bump",
    }
    assert expected.issubset(interactions)
