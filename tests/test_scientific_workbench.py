import io
import zipfile

from rdkit import Chem

from panvizplus.audit import build_analysis_audit
from panvizplus.chemistry.models import Atom, NormalizedStructure
from panvizplus.chemistry.rdkit_layer import LigandChemistry
from panvizplus.interactions.models import CriterionResult, InteractionRecord
from panvizplus.reproducibility import build_manifest, build_publication_bundle
from panvizplus.rules import list_rulesets, load_ruleset


def atom(i, name, element, res, num, chain, xyz, record):
    return Atom(i, name, element, res, num, chain, *xyz, record)


def test_builtin_rule_profiles_are_versioned_and_distinct():
    profiles = list_rulesets()
    ids = [p["profile_id"] for p in profiles]
    assert ids == [
        "panvizplus_v1",
        "plip_style_2026_1",
        "prolif_style_2026_1",
    ]

    consensus = load_ruleset(profile="panvizplus_v1")
    plip = load_ruleset(profile="plip_style_2026_1")
    prolif = load_ruleset(profile="prolif_style_2026_1")

    assert consensus["metadata"]["display_name"] == "panVizPlus Consensus 2026.1"
    assert plip["metadata"]["exact_external_tool_reproduction"] is False
    assert prolif["metadata"]["exact_external_tool_reproduction"] is False
    assert (
        plip["interactions"]["conventional_hbond"]["geometry"]["donor_acceptor_distance_max"]
        != prolif["interactions"]["conventional_hbond"]["geometry"]["donor_acceptor_distance_max"]
    )


def test_audit_retains_rejected_hbond_reason():
    structure = NormalizedStructure(
        atoms=[
            atom(1, "CE", "C", "LYS", 20, "A", (1.4, 0.0, 0.0), "ATOM"),
            atom(2, "NZ", "N", "LYS", 20, "A", (0.0, 0.0, 0.0), "ATOM"),
            atom(3, "O1", "O", "LIG", 1, "Z", (2.9, 0.0, 0.0), "HETATM"),
            atom(4, "C1", "C", "LIG", 1, "Z", (4.0, 0.0, 0.0), "HETATM"),
        ],
        bonds={(1, 2), (3, 4)},
    )
    chemistry = LigandChemistry(
        mol=Chem.Mol(),
        selector="LIG:Z:1",
        rd_idx_to_atom_id={},
        atom_id_to_rd_idx={},
        acceptor_atom_ids={3},
        confidence="high",
        source="test",
        reconstruction_mode="test",
    )
    rules = load_ruleset()
    audit = build_analysis_audit(
        structure,
        "LIG:Z:1",
        [],
        rules,
        chemistry,
    )

    assert audit["hbond_audit"]["candidate_total"] == 1
    assert audit["hbond_audit"]["rejected"] == 1
    assert audit["hbond_audit"]["rejection_reasons"]["XDA_angle"] == 1
    assert audit["protein_residues_near_ligand"] >= 1


def test_publication_bundle_contains_reproducibility_assets():
    rules = load_ruleset()
    record = InteractionRecord(
        interaction_id="HB-0001",
        interaction_type="conventional_hbond",
        ligand_site="LIG:Z:1:O1",
        protein_site="SER:A:205:OG",
        residue_name="SER",
        residue_number=205,
        chain_id="A",
        criteria=[
            CriterionResult(
                "donor_acceptor_distance", 2.9, "<=", 3.4, "angstrom", True
            )
        ],
        measurements={"donor_acceptor_distance": 2.9},
        metadata={"chemistry_source": "wwPDB_CCD:ABC"},
    )
    chemistry = {
        "source": "wwPDB_CCD:ABC",
        "reconstruction_mode": "wwPDB_CCD",
        "confidence": "high",
        "net_charge": 0,
    }
    manifest = build_manifest(
        input_sha256="abc123",
        ligand_selector="ABC:A:1",
        ligand_net_charge=0,
        chemistry=chemistry,
        rules=rules,
        interaction_count=1,
    )
    audit = {
        "accepted_total": 1,
        "accepted_by_type": {"conventional_hbond": 1},
        "hbond_audit": {"records": [record], "accepted": 1, "rejected": 0},
    }

    bundle = build_publication_bundle(
        svg="<svg></svg>",
        records=[record],
        manifest=manifest,
        rules=rules,
        audit=audit,
    )

    with zipfile.ZipFile(io.BytesIO(bundle)) as zf:
        names = set(zf.namelist())

    assert {
        "figure.svg",
        "interactions.csv",
        "interactions.json",
        "analysis_manifest.json",
        "rules_used.yaml",
        "audit_summary.json",
        "methods.txt",
        "figure_caption.txt",
    } <= names
