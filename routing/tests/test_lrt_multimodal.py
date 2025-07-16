import requests, sys, time, json

BASE = "http://127.0.0.1:5000/route"

# ----- Helper -----

def run_case(idx, name, start, end, modes, expect_lrt):
    payload = {
        "start": {"lat": start[0], "lon": start[1]},
        "end":   {"lat": end[0],   "lon": end[1]},
        "modes": modes,
        "preferences": ["fastest"],
        "passenger_type": "regular",
    }
    try:
        r = requests.post(BASE, json=payload, timeout=40)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[{idx}] {name}: ERROR → {e}")
        return False
    segments = data.get("fastest", [])
    modes_seen = {seg.get("mode", "").lower() for seg in segments}
    ok = ("lrt" in modes_seen) if expect_lrt else ("lrt" not in modes_seen)
    status = "PASS" if ok else "FAIL"
    print(f"[{idx}] {name}: {status}  modes={sorted(modes_seen)}  dist={data.get('summary',{}).get('total_distance'):.2f} km")
    return ok

# ----- Test definitions -----
# Coordinates taken straight from data/stops.txt for reliability
R_PAPA      = (14.6361801, 120.9823664)
VITO_CRUZ   = (14.5636557, 120.9946252)
PEDRO_GIL   = (14.5824867, 120.9846238)
CENTRAL     = (14.5927994, 120.9815980)
DOROTEO_J   = (14.6051749, 120.9820053)

TESTS = [
    # --- LRT-only expectations ---
    ("LRT full line",           R_PAPA, VITO_CRUZ,   ["lrt"],               True),
    ("Mid-section northbound",  PEDRO_GIL, DOROTEO_J, ["lrt"],               True),
    ("Central ↔ Pedro Gil",     CENTRAL, PEDRO_GIL,  ["lrt"],               True),
    ("Short hop Central⇢DorJ",  CENTRAL, DOROTEO_J,  ["lrt"],               True),
    ("Short hop Pedro⇢Central", PEDRO_GIL, CENTRAL,  ["lrt"],               True),
    # --- Multimodal (rail allowed but not mandatory) ---
    ("Multimodal R Papa→V Cruz", R_PAPA,  VITO_CRUZ, ["jeepney","bus","lrt"], True),
    ("Multimodal Central→V Cruz", CENTRAL, VITO_CRUZ, ["bus","lrt"],        True),
    ("Multimodal short (may walk)", CENTRAL, PEDRO_GIL, ["lrt","walking"], True),
    ("Surface only (expect no rail)", R_PAPA, DOROTEO_J, ["jeepney","bus"], False),
    ("Tricycle only (expect no rail)", CENTRAL, PEDRO_GIL, ["tricycle"],     False),
]

# ----- Runner -----

fails = 0
for idx, (name, s, e, modes, want_lrt) in enumerate(TESTS, 1):
    ok = run_case(idx, name, s, e, modes, want_lrt)
    if not ok:
        fails += 1

print("\n======== Summary ========")
if fails:
    print(f"{fails} / {len(TESTS)} tests FAILED")
    sys.exit(1)
else:
    print("All tests passed ✔") 