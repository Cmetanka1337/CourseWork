# Berka Ingestion Report

- Rows read: **1056320**
- Rows normalized: **1056320**
- Unique users: **4500**
- Date range: **1993-01-01 -> 1998-12-31**
- Missing rates: `{"user_id": 0.0, "transaction_date": 0.0, "amount": 0.0, "category": 0.0, "flow_direction": 0.0}`

## Flow split
- Inflow rows: **405083**
- Outflow rows: **651237**

## Top categories
- `outflow:vyber`: 274675
- `inflow:urok`: 183114
- `inflow:vklad`: 156743
- `outflow:sluzby`: 155832
- `outflow:sipo`: 118065
- `outflow:prevod na ucet`: 60972
- `inflow:prevod z uctu`: 34888
- `inflow:duchod`: 30338
- `outflow:pojistne`: 18500
- `outflow:uver`: 13580
- `outflow:vyber kartou`: 8036
- `outflow:sankc. urok`: 1577

## Category rule
`category = f"{flow_direction}:{k_symbol or operation or type}"`
