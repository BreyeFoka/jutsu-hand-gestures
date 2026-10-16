from mudra_engine import JutsuComposer, MudraClassifier

# Finger states are (thumb, index, middle, ring, pinky): True means extended.


def test_classifies_each_known_hand_sign():
    c = MudraClassifier()
    assert c.classify((False, False, False, False, False)) == "TIGER"
    assert c.classify((False, True, False, False, False)) == "RAM"
    assert c.classify((False, True, True, False, False)) == "BIRD"
    assert c.classify((False, True, True, True, False)) == "HORSE"
    assert c.classify((True, True, False, False, False)) == "DOG"
    assert c.classify((True, True, True, True, True)) == "BOAR"


def test_unknown_hand_shapes_are_not_signs():
    assert MudraClassifier().classify((True, False, True, False, True)) is None


def test_detects_a_jutsu_when_its_signs_arrive_in_order():
    composer = JutsuComposer()
    results = [composer.add_mudra(m, i) for i, m in enumerate(["DOG", "TIGER", "BOAR", "RAM"])]
    assert results[:3] == [None, None, None]
    assert results[3].name == "Fireball (Demo)"


def test_finds_a_jutsu_after_unrelated_signs():
    composer = JutsuComposer()
    for i, m in enumerate(["HORSE", "TIGER", "RAM", "BIRD", "DOG"]):
        match = composer.add_mudra(m, i)
    assert match.name == "Shadow Clone (Demo)"


def test_wrong_order_does_not_count():
    composer = JutsuComposer()
    for i, m in enumerate(["RAM", "BOAR", "TIGER", "DOG"]):
        assert composer.add_mudra(m, i) is None


def test_history_stays_bounded():
    composer = JutsuComposer()
    for i in range(100):
        composer.add_mudra("TIGER", i)
    assert len(composer.events) == 30
    assert composer.recent_sequence(3) == ["TIGER", "TIGER", "TIGER"]
