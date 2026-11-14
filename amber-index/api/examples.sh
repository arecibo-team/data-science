#!/usr/bin/env bash
set -euo pipefail
BASE="${BASE:-http://127.0.0.1:8000}"
JSON='Content-Type: application/json'

step() { printf '\n\033[1m%s\033[0m\n$ %s\n' "$1" "$2"; }

step "1. Is the service up, and what was it trained on?" "curl $BASE/health"
curl -s "$BASE/health" | jq '{status, latest_year, training_years, cv_metrics: [.cv_metrics[] | {model, validation, MAE: (.MAE*1000|round/1000), Spearman: (.Spearman*1000|round/1000)}]}'

step "2. The exported model: features, pillars and weights" "curl $BASE/model"
curl -s "$BASE/model" | jq '{model, target, pillars: (.pillars | map_values(.w_final*1000|round/1000)), features: [.features[] | {feature, pillar, w_final: (.w_final*1000|round/1000)}]}'

step "3. Where is Europe's protection gap largest this year?" "curl $BASE/scores"
curl -s "$BASE/scores" | jq -r '.[] | "\(.rank)\t\(.iso2)\t\(.amber|round)\t\(.band)\tpower \(.power|round) (#\(.power_rank))\t\(.archetype)"' | head -10

step "4. Why is Greece red?" "curl $BASE/scores/GR"
curl -s "$BASE/scores/GR" | jq '{name, year, amber: (.amber|round), band, rank, rank_range: "\(.rank_lo)-\(.rank_hi)", power: (.power|round), reasons: [.reasons[].text], imputed: [.imputed | to_entries[] | select(.value) | .key]}'

step "5. Look back: the 2017 season" "curl '$BASE/scores?year=2017'"
curl -s "$BASE/scores?year=2017" | jq -r '.[:5][] | "\(.rank)\t\(.iso2)\t\(.amber|round)\t\(.band)"'

step "6. What if Romania doubled its firefighters? (raw indicators in, score out)" "curl -X POST $BASE/predict -d '{...}'"
for ff in 23481 46962; do
  curl -s -X POST "$BASE/predict" -H "$JSON" -d "{\"burned_ha\": 23000, \"area_km2\": 238398, \"population\": 19000000, \"firefighters\": $ff, \"spending_meur\": 900, \"stations\": 295, \"hazard_load\": 20, \"major_load\": 3}" \
    | jq -c "{firefighters: $ff, amber: .amber, band, power}"
done

step "7. Policy sliders: weigh danger twice as much as readiness" "curl -X POST $BASE/score/custom-weights -d '{\"danger\": 2, \"hazards\": 1, \"workforce\": 1, \"capability\": 1}'"
curl -s -X POST "$BASE/score/custom-weights" -H "$JSON" -d '{"danger": 2, "hazards": 1, "workforce": 1, "capability": 1}' \
  | jq -c '{weights, top5: [.scores[:5][] | "\(.iso2) \(.amber|round) \(.band)"]}'

step "8. Stress test: the same disaster hits the strongest (CZ) and the weakest (CY) country" "curl -X POST $BASE/simulate -d '{\"shocks\": [{\"iso2\": \"CZ\", \"preset\": \"multi_hazard\"}, {\"iso2\": \"CY\", \"preset\": \"multi_hazard\"}]}'"
curl -s "$BASE/simulate/presets" | jq -c '.multi_hazard'
curl -s -X POST "$BASE/simulate" -H "$JSON" -d '{"shocks": [{"iso2": "CZ", "preset": "multi_hazard"}, {"iso2": "CY", "preset": "multi_hazard"}]}' \
  | jq -r '.rows[] | select(.shocked) | "\(.iso2)\tpower \(.power_before|round) -> \(.power_after|round)\trank \(.power_rank_before) -> \(.power_rank_after)\tcrews tied up \(.committed_share*100|round)%\tunmet \(.unmet_demand)"'

step "9. Who should help Cyprus, and with how many firefighters?" "curl -X POST $BASE/optimize/assistance -d '{\"recipient\": \"CY\", \"shocks\": [{\"iso2\": \"CY\", \"preset\": \"multi_hazard\"}]}'"
curl -s -X POST "$BASE/optimize/assistance" -H "$JSON" -d '{"recipient": "CY", "shocks": [{"iso2": "CY", "preset": "multi_hazard"}]}' \
  | jq -r '"need \(.need), covered \(.covered), shortfall \(.shortfall), mean distance \(.mean_distance_km|round) km", (.donors[] | "  \(.iso2)\t\(.sent)\t\(.share_of_need*100|round)% of need\t\(.distance_km|round) km")'

step "10. Same request, but Poland and Czechia are fighting their own disasters" "curl -X POST $BASE/optimize/assistance -d '{... PL and CZ shocked ...}'"
curl -s -X POST "$BASE/optimize/assistance" -H "$JSON" -d '{"recipient": "CY", "shocks": [{"iso2": "CY", "preset": "multi_hazard"}, {"iso2": "PL", "preset": "multi_hazard"}, {"iso2": "CZ", "preset": "multi_hazard"}]}' \
  | jq -r '"need \(.need), covered \(.covered)", (.donors[] | "  \(.iso2)\t\(.sent)")'

step "11. A fire near Coimbra, Portugal: closest help first vs strongest help first" "curl -X POST $BASE/optimize/assistance -d '{\"recipient\": \"PT\", \"need\": 3000, \"incident_lat\": 40.2, \"incident_lon\": -8.4, \"weight_location\": 1}'"
for w in 1 0; do
  curl -s -X POST "$BASE/optimize/assistance" -H "$JSON" -d "{\"recipient\": \"PT\", \"need\": 3000, \"incident_lat\": 40.2, \"incident_lon\": -8.4, \"weight_location\": $w}" \
    | jq -c "{weight_location: $w, mean_km: (.mean_distance_km|round), donors: [.donors[] | \"\(.iso2) \(.sent)\"]}"
done
