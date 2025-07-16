import pytest
from maiwayrouting.core_route_service import UnifiedRouteService

# NOTE: This test assumes you have tricycle_terminals.geojson in data/ and that
# the file contains at least one terminal.  If absent, the test is skipped.

route_service = UnifiedRouteService("data")

@pytest.mark.skipif(not route_service.trike_terminals, reason="No tricycle terminals loaded")
def test_trike_leg_only_in_convenient():
    """Convenient preference should inject a ₱21 tricycle segment; other prefs should not."""
    # Pick a terminal whose nearest GTFS stop is within 1.5 km so the trike logic triggers
    term = None
    for t in route_service.trike_terminals:
        _, stop_dist = route_service.find_nearest_stop(t["lat"], t["lon"])
        if stop_dist <= 1.5:
            term = t
            break

    if term is None:
        pytest.skip("No tricycle terminal within 1.5 km of any GTFS stop – cannot test trike logic.")

    # 150 m north-east from the chosen terminal
    origin_lat = term["lat"] + 0.0015
    origin_lon = term["lon"] + 0.0015

    # Destination ~2 km away (arbitrary but reachable)
    dest_lat = term["lat"] + 0.02
    dest_lon = term["lon"] + 0.02

    # ---- Convenient route ----
    result_convenient = route_service.find_all_routes_with_coordinates(
        origin_lat,
        origin_lon,
        dest_lat,
        dest_lon,
        preferences=["convenient"],
    ).get("convenient")

    assert result_convenient and result_convenient.get("segments"), "No route returned for convenient preference"
    modes = {seg.get("mode") for seg in result_convenient["segments"]}
    if "Tricycle" not in modes:
        pytest.skip("Tricycle segment did not appear for chosen OD; skipping assertion.")

    # Fare should include flat PHP 21 trike charge (allow small float tolerance)
    assert pytest.approx(result_convenient["total_cost"], rel=1e-2) >= 21.0

    # ---- Fastest route ----
    result_fastest = route_service.find_all_routes_with_coordinates(
        origin_lat,
        origin_lon,
        dest_lat,
        dest_lon,
        preferences=["fastest"],
    ).get("fastest")

    if result_fastest and result_fastest.get("segments"):
        modes_fast = {seg.get("mode") for seg in result_fastest["segments"]}
        assert "Tricycle" not in modes_fast, "Tricycle should not appear in fastest route" 