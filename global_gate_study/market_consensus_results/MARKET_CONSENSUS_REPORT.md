# Market-inclusive conservative-to-consensus Gate stress

Eligible comparisons: **38** across **16 datasets**.

Each baseline contains the official local JKP market plus size, book-to-market and 12-1 momentum; the residual library is frozen. `cons_mean` and `cons_stack` never expand beyond the median of old/recent/all predeployment complexity choices, so additional complexity requires support from at least two temporal views.

## Aggregate

```
            comparisons  datasets  avg_gainQ  median_gainQ  avg_sr  avg_delta_sr    avg_C  avg_cancel  positive_year_frac
selector                                                                                                                 
cons_mean            38        16     0.0773        0.0486  1.2297        0.5031  13.8117      2.1914              0.6666
gate2                38        16     0.0768        0.0490  1.2179        0.4913  13.2769      2.1613              0.6599
cons_stack           38        16     0.0766        0.0466  1.2342        0.5077  14.0632      2.2043              0.6631
stack3               38        16     0.0765        0.0467  1.2394        0.5129  14.5341      2.1826              0.6588
median3              38        16     0.0746        0.0440  1.2329        0.5063  14.3465      2.2215              0.6540
current              38        16     0.0627        0.0242  1.1976        0.4710  15.7027      2.2590              0.6267
```

## Wins/losses vs current

```
            q_wins  q_losses  sr_wins  sr_losses
selector                                        
cons_mean       28         8       21         15
cons_stack      28         8       24         12
current          0         0        0          0
gate2           27         8       18         17
median3         28         8       23         13
stack3          30         8       26         12
```

## Annual diagnostics

Mean Gate2-to-consensus grid gap: 1.025.
Mean cons-stack weight on Gate2: 0.866.

## Skipped

```
                   dataset                             reason
emerging_market_vwcap_T240 ValueError('insufficient history')
                  frontier           missing 2 frozen factors
     aus_market_vwcap_T120 ValueError('insufficient history')
     aus_market_vwcap_T240 ValueError('insufficient history')
     che_market_vwcap_T240 ValueError('insufficient history')
     hkg_market_vwcap_T240 ValueError('insufficient history')
     nld_market_vwcap_T120 ValueError('insufficient history')
     nld_market_vwcap_T240 ValueError('insufficient history')
     swe_market_vwcap_T240 ValueError('insufficient history')
                       ita           usable suffix 224 months
                       esp           usable suffix 224 months
                       sgp           usable suffix 101 months
                       kor           usable suffix 176 months
                       bra           usable suffix 212 months
     ind_market_vwcap_T120 ValueError('insufficient history')
     ind_market_vwcap_T240 ValueError('insufficient history')
                       chn           usable suffix 188 months
                       zaf           usable suffix 230 months
```