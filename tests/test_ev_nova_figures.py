"""The committed SVG figures of the README are what `ev_nova/figures.py` draws, byte for byte."""

import pytest

from ev_nova import figures


@pytest.mark.parametrize("name", sorted(figures.FIGURES))
def test_committed_figure_is_current(name):
    committed = (figures.OUT / f"{name}.svg").read_bytes()
    assert figures.FIGURES[name]().encode("utf-8") == committed


def test_naev_envelope_ties_where_the_best_route_changes():
    # The kept envelope is checked against landing_envelope by `main` (half a minute); here,
    # each pair of neighbors ties exactly at the landing cost where the next one takes over.
    from fractions import Fraction

    parts = figures.naev_parts()
    for i in range(len(parts) - 1):
        (_, g1, j1, s1), (_, g2, j2, s2) = parts[i], parts[i + 1]
        x = figures.NAEV_ENVELOPE[i + 1][0]
        assert Fraction(g1, 1) / (j1 + x * s1) == Fraction(g2, 1) / (j2 + x * s2)
    assert [str(x) for x, _ in figures.NAEV_ENVELOPE[1:]] == ["29/168", "111/86"]


def test_naev_map_positions_and_paths(tmp_path):
    import xml.etree.ElementTree as ET

    from ev_nova import naev

    data = naev.load()
    assert all(len(s["pos"]) == 2 for s in data["systems"].values())
    assert figures.shortest_paths(data, "Scarlet", "Druss") == [["Scarlet", "Druss"]]
    paths = figures.shortest_paths(data, "Modus Manis", "An'ku")
    assert sorted(p[1] for p in paths) == ["Humdrum", "Tarmak"]
    assert all(len(p) == 3 for p in paths)
    # The reader takes the position from <pos>, and (0, 0) when a file has none.
    xml = '<ssys name="A"><general><radius>1</radius></general><pos x="-1.5" y="2"/></ssys>'
    (tmp_path / "a.xml").write_text(xml)
    assert naev.read_system(tmp_path / "a.xml").pos == (-1.5, 2.0)
    ET.ElementTree(ET.fromstring('<ssys name="B"><general/></ssys>')).write(tmp_path / "b.xml")
    assert naev.read_system(tmp_path / "b.xml").pos == (0.0, 0.0)
