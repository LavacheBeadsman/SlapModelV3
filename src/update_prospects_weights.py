"""
Update data/prospects_final.csv with official combine weights from combine_2026.csv.
Combine weigh-in data is more accurate than pre-combine estimates.
"""

import pandas as pd

pf = pd.read_csv("data/prospects_final.csv")
combine = pd.read_csv("data/combine_2026.csv")
rb_combine = combine[combine["position"] == "RB"]

def normalize(n):
    return str(n).strip().lower().replace(".", "").replace("'", "")

updated = 0
for _, cr in rb_combine.iterrows():
    if pd.isna(cr["weight"]):
        continue
    cn = normalize(cr["player_name"])
    for idx, pr in pf.iterrows():
        if pr["position"] != "RB":
            continue
        pn = normalize(pr["player_name"])
        # Match by last name + first name overlap
        cn_parts = cn.split()
        pn_parts = pn.split()
        if cn_parts[-1] == pn_parts[-1] and (cn_parts[0] in pn_parts[0] or pn_parts[0] in cn_parts[0]):
            old = pf.loc[idx, "weight"]
            new = cr["weight"]
            if pd.isna(old) or old != new:
                pf.loc[idx, "weight"] = new
                print(f"  {pr['player_name']}: {old} -> {new}")
                updated += 1
            break

print(f"\nUpdated {updated} weights")
pf.to_csv("data/prospects_final.csv", index=False)
print(f"Saved data/prospects_final.csv: {len(pf)} rows")
