# Global market-inclusive Gate stress

Eligible comparisons: **38** across **16 datasets**.

Each baseline contains the official local JKP market return plus size, book-to-market, and 12-1 momentum. The same frozen 37 residual extensions are then added.

## Aggregate

```
          comparisons  datasets  avg_gainQ  median_gainQ  avg_sr  avg_delta_sr    avg_C  positive_year_frac
selector                                                                                                   
gate2              38        16     0.0768        0.0490  1.2179        0.4913  13.2769              0.6599
stack2             38        16     0.0766        0.0466  1.2342        0.5077  14.0632              0.6631
mean3              38        16     0.0765        0.0473  1.2379        0.5113  14.5906              0.6706
stack3             38        16     0.0765        0.0467  1.2394        0.5129  14.5341              0.6588
allpast            38        16     0.0746        0.0440  1.2329        0.5063  14.3465              0.6540
median3            38        16     0.0746        0.0440  1.2329        0.5063  14.3465              0.6540
minimax            38        16     0.0745        0.0277  1.2263        0.4997  13.2762              0.5996
current            38        16     0.0627        0.0242  1.1976        0.4710  15.7027              0.6267
recent             38        16     0.0623        0.0235  1.2093        0.4827  13.7227              0.5754
```

## Wins/losses vs current

```
          q_wins  q_losses  sr_wins  sr_losses
selector                                      
allpast       28         8       23         13
current        0         0        0          0
gate2         27         8       18         17
mean3         33         5       27         11
median3       28         8       23         13
minimax       28        10       24         14
recent        22        16       21         17
stack2        28         8       24         12
stack3        30         8       26         12
```

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