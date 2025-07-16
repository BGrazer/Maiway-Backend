import requests, sys

BASE = "http://127.0.0.1:5000/route"

# Handy coordinate aliases (lat, lon)
R_PAPA  = (14.6361801, 120.9823664)
ABAD    = (14.6305540, 120.9814634)
BLUM    = (14.6226483, 120.9828683)
TAYUMAN = (14.6166975, 120.9827445)
BAMBANG = (14.6110290, 120.9824354)
UNAVE   = (14.6109554, 120.9824654)
DOR_J   = (14.6051749, 120.9820053)
CARRI   = (14.5990659, 120.9813079)
CENTRAL = (14.5927994, 120.9815980)
PEDRO   = (14.5824867, 120.9846239)
QUIRINO = (14.5765466, 120.9880546)
V_CRUZ  = (14.5636557, 120.9946252)

# Bus-only stop examples
BUS_BLUM  = (14.6066007, 120.9820723)   # stop_62_045
BUS_LAWTON= (14.5998187, 120.9813328)   # stop_4_103
BUS_PIER  = (14.5837233, 120.9838563)   # stop_62_103

CASES = [
    ("Full line north→south",          R_PAPA,  V_CRUZ,   True),
    ("Abad Santos → Central",          ABAD,    CENTRAL,  True),
    ("Blumentritt → Pedro Gil",        BLUM,    PEDRO,    True),
    ("Tayuman → UN Ave",               TAYUMAN, UNAVE,    True),
    ("Bambang → Quirino",              BAMBANG, QUIRINO,  True),
    ("Carriedo → Pedro Gil",           CARRI,   PEDRO,    True),
    ("Central → Quirino",              CENTRAL, QUIRINO,  True),
    ("Quirino → Abad Santos (reverse)",QUIRINO, ABAD,     True),
    ("Blumentritt bus → Vito Cruz",    BUS_BLUM,V_CRUZ,   True),
    ("R Papa → Lawton bus stop",       R_PAPA,  BUS_LAWTON,True),
    ("Lawton bus → Pedro Gil",         BUS_LAWTON, PEDRO, True),
    # --- 5 cases that should NOT pick rail (no rail allowed) ---
    ("Surface only short hop",         BUS_BLUM, BUS_LAWTON, False),
    ("Pier bus → Lawton bus",          BUS_PIER, BUS_LAWTON, False),
    ("Pier bus → Blumentritt bus",     BUS_PIER, BUS_BLUM,  False),
    ("Blumentritt bus → Pier bus",     BUS_BLUM, BUS_PIER,  False),
    ("Lawton bus → Tayuman bus",       BUS_LAWTON, TAYUMAN, False),
    # --- more mixed ---
    ("R Papa → Carriedo",              R_PAPA,  CARRI,    True),
    ("Central → Vito Cruz",            CENTRAL, V_CRUZ,   True),
    ("Abad → UN Ave",                  ABAD,    UNAVE,    True),
    ("Pedro Gil → R Papa",             PEDRO,   R_PAPA,   True),
]

FAILS = 0
for idx, (name, start, end, need_lrt) in enumerate(CASES, 1):
    modes = ["jeepney", "bus", "lrt"] if need_lrt else ["jeepney", "bus"]
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
        segs = data.get("fastest", [])
        seen = {s.get("mode", "").lower() for s in segs}
        ok = ("lrt" in seen) if need_lrt else ("lrt" not in seen)
        status = "PASS" if ok else "FAIL"
    except Exception as e:
        status = f"ERROR {e}"
        ok = False
        seen = set()
    print(f"[{idx:02}] {name:35} {status:5} modes={sorted(seen)}")
    if not ok:
        FAILS += 1

print("\n========================")
if FAILS:
    print(f"{FAILS}/{len(CASES)} tests FAILED")
    sys.exit(1)
print("All multimodal tests passed") 