# Підсумковий результат аналізу Dataset №1 (Stage 0-10)

## 1) Що було зроблено

Виконано повний validation pipeline до моделювання:

1. Stage 0: Sanity check
2. Stage 1: Category collapse
3. Stage 2: Побудова daily time series
4. Stage 3: Sparsity analysis
5. Stage 4: Distribution analysis
6. Stage 5: Autocorrelation analysis
7. Stage 6: Baseline test
8. Stage 7: Target validation
9. Stage 8: User variance
10. Stage 9: Category-level signal
11. Stage 10: Event-level analysis
12. Фінальне decision rule (CASE 1/2/3)

## 2) Дані та структура

- Джерело: `priyamchoksi/credit-card-transactions-dataset`
- Вхідний файл: `credit_card_transactions.csv`
- Використані поля:
  - user: `cc_num`
  - timestamp: `trans_date_trans_time`
  - amount: `amt`
  - category: `category`

Додаткова перевірка кардинальностей:

- `category_unique = 14`
- `merchant_unique = 693`
- `trans_num_unique = 1,296,675`

Висновок: "1M категорій" тут не підтвердилось для поля `category`; таке значення відповідає унікальному ID транзакції (`trans_num`), а не категорії витрат.

## 3) Stage-by-stage результати

## Stage 0 - Sanity

- Кількість рядків: **1,296,675**
- Кількість користувачів: **983**
- Діапазон дат: **2019-01-01 .. 2020-06-21**
- Середня кількість транзакцій на користувача: **1319.10**
- Медіана транзакцій на користувача: **1054**
- Користувачі з `>=200` транзакціями: **908 / 983**
- Частка користувачів з `>=200` транзакціями: **0.924**

Інтерпретація: за критерієм "<200 транзакцій = непридатний для personalization" більшість користувачів придатні.

## Stage 1 - Category collapse (manual)

- Унікальних сирих категорій: **14**
- Після collapse: **7**
- Режим: `manual`

Top raw categories:

| raw_category | count |
|---|---:|
| gas_transport | 131659 |
| grocery_pos | 123638 |
| home | 123115 |
| shopping_pos | 116672 |
| kids_pets | 113035 |
| shopping_net | 97543 |
| entertainment | 94014 |
| food_dining | 91461 |
| personal_care | 90758 |
| health_fitness | 85879 |

Top collapsed categories:

| category | count |
|---|---:|
| other | 555729 |
| shopping | 214215 |
| groceries | 169090 |
| fuel | 131659 |
| entertainment | 94014 |
| restaurants | 91461 |
| travel | 40507 |

Інтерпретація: категоризація вже відносно компактна; найбільший клас `other` (ризик змішування різних патернів).

## Stage 2 - Time series construction

- Побудовано щоденну рівномірну сітку для кожної пари `user_id x category`.
- Пропуски заповнені нулями.
- Подальші етапи виконані на щоденному spend-ряді.

## Stage 3 - Sparsity analysis

Загалом по `user x category`:

- Кількість пар: **6881**
- Median active days %: **18.22%**
- Частка пар `<20%`: **53.5%**
- Частка пар `20-50%`: **33.8%**
- Частка пар `>=50%`: **12.7%**

Медіанна активність за категоріями:

| category | median active_days_pct |
|---|---:|
| other | 57.43 |
| shopping | 30.11 |
| groceries | 24.91 |
| fuel | 17.10 |
| entertainment | 15.06 |
| restaurants | 14.68 |
| travel | 7.06 |

Інтерпретація: розрідженість висока, особливо для більшості тематичних категорій; стабільний сигнал обмежений.

## Stage 4 - Distribution analysis

- Частка нульових значень (daily grid): **0.760**
- Mean: **24.64**
- Median: **0.0**
- p95: **134.81**
- p99: **349.27**
- Skewness: **46.51**

Інтерпретація: дуже сильний heavy-tail і правостороння асиметрія; багато нулів + рідкі великі витрати.

## Stage 5 - Autocorrelation (найкритичніше)

Загалом:

- Median corr(t, t-1): **0.013**
- Median corr(t, t-7): **0.004**
- Median corr(t, t-30): **-0.007**
- Частка пар з corr(t, t-7) `<0.1`: **0.917**
- Частка `0.1..0.3`: **0.083**
- Частка `>=0.3`: **0.0006**

Median corr(t, t-7) за категоріями:

| category | median corr_lag7 |
|---|---:|
| other | 0.0328 |
| fuel | 0.0262 |
| groceries | 0.0258 |
| restaurants | 0.0041 |
| entertainment | 0.0018 |
| travel | -0.0031 |
| shopping | -0.0058 |

Інтерпретація: часовий сигнал майже відсутній за заданими критеріями (переважно <0.1).

## Stage 6 - Baseline test

Target: сума витрат на горизонті `t+1..t+7`.

Середні помилки baseline-моделей:

| baseline | MAE | RMSE |
|---|---:|---:|
| mean_predictor | 123.49 | 208.12 |
| rolling_mean_7 | 156.02 | 287.25 |
| last_value | 236.71 | 554.20 |

Інтерпретація: найкращий baseline тут — `mean_predictor`; сигнал динаміки (lag/rolling) слабкий.

## Stage 7 - Target validation

Кореляція між `target(t+1..t+7)` і `rolling_mean_7(t)`:

- Median corr: **0.0178**
- Частка corr `>=0.9`: **0.0**
- Квантилі corr: p10 **-0.0738**, p25 **-0.0278**, p50 **0.0178**, p75 **0.1001**, p90 **0.1842**

Інтерпретація: target не пояснюється просто rolling-mean (тобто не тривіальна копія середнього), але і сильного прогнозного сигналу немає.

## Stage 8 - User variance

- `cv_daily_mean_spend`: **0.642**
- `between_user_category_variance`: **476.30**
- Квантилі `daily_mean` по користувачах: p10 **56.14**, p25 **80.96**, p50 **164.52**, p75 **235.66**, p90 **320.57**

Інтерпретація: користувачі суттєво відрізняються за рівнем витрат; personalization концептуально має сенс, але слабкий часовий сигнал лімітує прогноз.

## Stage 9 - Category-level signal

Фокусні категорії з вимоги:

| category | median active_days_pct | median corr_lag7 |
|---|---:|---:|
| groceries | 24.91 | 0.0258 |
| fuel | 17.10 | 0.0262 |

`utilities` у поточному collapse-результаті як окремий клас відсутній (переважно поглинутий `other`).

Інтерпретація: навіть у "кращих" категоріях lag-7 кореляція дуже низька.

## 4) Stage 10 - Event-level analysis

На цьому етапі аналізовано бінарну серію:

- `y_t = 1`, якщо `amount > 0`
- `y_t = 0`, якщо витрата відсутня

### Результати

- Кількість проаналізованих пар `user x category`: **6881**
- Median event rate: **0.182**
- Median corr(t, t-1): **0.020**
- Median corr(t, t-7): **0.026**
- Median P(event_t=1 | event_t-1=0): **0.178**
- Median P(event_t=1 | event_t-1=1): **0.222**

### Baseline для event prediction

| baseline | accuracy | balanced accuracy |
|---|---:|---:|
| always-0 | 0.760 | — |
| frequency | 0.800 | 0.500 |

### Інтерпретація

Це важливий результат: **amount regression** справді виглядає майже безсигнальною задачею, але **event-level prediction** показує слабкий, хоча й не сильний, структурний патерн.

Отже:

- для регресії суми витрат сигнал слабкий;
- для передбачення факту витрати сигнал є, але він дуже обмежений;
- hybrid підхід або classification-first постановка виглядає більш обґрунтовано, ніж чиста regression.

## 5) Фінальний висновок (Decision logic)

**Рішення пайплайна: `CASE 1: no_signal` для regression по `amount`**

Причини:

1. Висока розрідженість (`53.5%` пар мають <20% активних днів).
2. Дуже слабка автокореляція для `amount` (median lag7 = `0.004`, `91.7%` пар мають lag7 < 0.1).
3. Розподіл з великою масою нулів і heavy-tail, що ускладнює стабільний short-term forecast.
4. Для `event` сигнал слабкий, але не нульовий: спостерігається невелика стабільність переходів і кореляція на рівні `~0.02-0.03`.

Формулювання для курсової:

> Прогнозування персональних витрат на цьому датасеті в поточній постановці обмежене через слабкий інформаційний часовий сигнал та високу розрідженість рядів. Однак для event-level постановки наявний слабкий, але вимірюваний сигнал, тому classification-first або hybrid formulation має більше сенсу, ніж чиста regression.

## 5) Що з цього практично випливає

- Переходити до "сильної" ML-моделі без зміни постановки задачі недоцільно.
- Доцільні кроки для наступної ітерації:
  1. Перегрупувати категорії (наприклад, `tfidf` або жорстке бізнес-мапування з окремим `utilities`).
  2. Підняти рівень агрегації (тиждень замість дня) для зменшення sparsity.
  3. Формулювати задачу як propensity / event prediction (ймовірність транзакції), а не точний amount forecast.
  4. Для курсової зафіксувати: regression не дає сигналу, але event-level постановка має слабкий потенціал.
  5. Якщо рухатись далі — будувати hybrid pipeline: classification for occurrence + regression for amount conditional on event.

## 6) Де лежать артефакти

- Головний машинний звіт: `outputs/report.json`
- Короткий автоматичний звіт: `outputs/report.md`
- Stage 1 таблиці: `outputs/tables/stage1_top10_raw.csv`, `outputs/tables/stage1_top10_collapsed.csv`
- Stage 3: `outputs/tables/stage3_sparsity.csv`
- Stage 5: `outputs/tables/stage5_autocorrelation.csv`
- Stage 6: `outputs/tables/stage6_baselines_summary.csv`
- Stage 7: `outputs/tables/stage7_target_validation.csv`
- Stage 8: `outputs/tables/stage8_user_variance.csv`
- Stage 9: `outputs/tables/stage9_category_signal.csv`
- Stage 10: `outputs/tables/stage10_event_level_analysis.csv`
- Графіки розподілу: `outputs/figures/stage4_hist_daily_spend.png`, `outputs/figures/stage4_hist_log_daily_spend.png`

