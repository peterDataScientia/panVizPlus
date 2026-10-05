from panvizplus.rules import load_ruleset


def test_ruleset_has_required_provenance():
    rules = load_ruleset()
    meta = rules["metadata"]

    assert meta["id"] == "panvizplus_v1"
    assert meta["status"] == "alpha"
    assert meta["exact_biovia_reproduction"] is False


def test_native_rule_families_present():
    interactions = load_ruleset()["interactions"]
    expected = {
        "conventional_hbond",
        "hydrophobic_contact",
        "salt_bridge",
        "pi_cation",
        "pi_pi",
        "halogen_bond",
        "metal_coordination",
        "water_bridge",
        "unfavorable_vdw_bump",
    }
    assert expected.issubset(interactions)
