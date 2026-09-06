# Global Learnability Gate stress test

The 40-factor library was frozen before the additional countries were downloaded. No new-country return is used to choose the factor set.

Eligible comparisons: **44** across **18 datasets**.

## Aggregate selector results

```
          comparisons  datasets  avg_gainQ  median_gainQ  avg_sr  avg_delta_sr    avg_C  avg_cancel  positive_year_frac
selector                                                                                                               
stack3             44        18     0.0726        0.0469  1.1205        0.5621  13.8987      1.8698              0.6311
minimax            44        18     0.0722        0.0471  1.0914        0.5331  11.9121      1.8040              0.6019
mean3              44        18     0.0722        0.0455  1.1191        0.5608  13.9855      1.8833              0.6714
stack2             44        18     0.0702        0.0451  1.1086        0.5503  13.5072      1.8860              0.6221
gate2              44        18     0.0693        0.0423  1.0803        0.5220  12.4934      1.8557              0.5970
allpast            44        18     0.0676        0.0395  1.1085        0.5502  13.8547      1.8938              0.6175
median3            44        18     0.0675        0.0395  1.1076        0.5493  13.8388      1.8940              0.6175
recent             44        18     0.0605        0.0277  1.0966        0.5382  13.0907      1.8208              0.5939
current            44        18     0.0508        0.0213  1.0568        0.4985  15.0109      1.9353              0.6023
```

## Wins and losses versus the original selector

```
          q_wins  q_losses  sr_wins  sr_losses
selector                                      
allpast       30        12       27         15
current        0         0        0          0
gate2         29        11       18         22
mean3         34        10       33         11
median3       30        12       27         15
minimax       33        11       26         18
recent        28        16       26         18
stack2        31        11       27         15
stack3        36         8       33         11
```

## Selectors

- current: first five-year tuning block;
- gate2: min(first-five-year choice, all-past choice);
- allpast: all ten predeployment years;
- recent: most recent five years;
- median3: median complexity among old, recent and all-past choices;
- minimax: minimizes the worse loss across the two five-year blocks;
- mean3: equal-weight portfolio blend of old/recent/all-past choices;
- stack2: response-one stacking of Gate 2.0 and all-past;
- stack3: nonnegative response-one stacking of distinct old/recent/all-past policies.

## Skipped / insufficient-history datasets

```
            dataset                             reason
emerging_vwcap_T240 ValueError('insufficient history')
     frontier_vwcap           missing 2 frozen factors
     aus_vwcap_T120 ValueError('insufficient history')
     aus_vwcap_T240 ValueError('insufficient history')
     che_vwcap_T240 ValueError('insufficient history')
     hkg_vwcap_T240 ValueError('insufficient history')
     nld_vwcap_T120 ValueError('insufficient history')
     nld_vwcap_T240 ValueError('insufficient history')
     swe_vwcap_T240 ValueError('insufficient history')
          ita_vwcap      usable suffix only 224 months
          esp_vwcap      usable suffix only 224 months
          sgp_vwcap      usable suffix only 101 months
          kor_vwcap      usable suffix only 176 months
          bra_vwcap      usable suffix only 212 months
     ind_vwcap_T120 ValueError('insufficient history')
     ind_vwcap_T240 ValueError('insufficient history')
          chn_vwcap      usable suffix only 188 months
          zaf_vwcap      usable suffix only 230 months
```

## Selector disagreement

Mean max-minus-min grid-index disagreement across annual old/recent/all choices: 1.49.
Stack2 mean weight on conservative Gate 2.0: 0.840.
