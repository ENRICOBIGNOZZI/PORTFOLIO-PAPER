# Conservative-to-consensus Gate stress

Eligible comparisons: **44** across **18 datasets**.

The factor library is the previously frozen 40-factor set. `cons_mean` and `cons_stack` never expand beyond the median of old/recent/all predeployment complexity choices, so additional complexity requires support from at least two temporal views.

## Aggregate

```
            comparisons  datasets  avg_gainQ  median_gainQ  avg_sr  avg_delta_sr    avg_C  avg_cancel  positive_year_frac
selector                                                                                                                 
stack3               44        18     0.0726        0.0469  1.1205        0.5621  13.8987      1.8698              0.6311
cons_mean            44        18     0.0706        0.0439  1.1002        0.5419  13.1661      1.8748              0.6260
cons_stack           44        18     0.0701        0.0451  1.1082        0.5499  13.5005      1.8869              0.6221
gate2                44        18     0.0693        0.0423  1.0803        0.5220  12.4934      1.8557              0.5970
median3              44        18     0.0675        0.0395  1.1076        0.5493  13.8388      1.8940              0.6175
current              44        18     0.0508        0.0213  1.0568        0.4985  15.0109      1.9353              0.6023
```

## Wins/losses vs current

```
            q_wins  q_losses  sr_wins  sr_losses
selector                                        
cons_mean       31        11       25         17
cons_stack      31        11       27         15
current          0         0        0          0
gate2           29        11       18         22
median3         30        12       27         15
stack3          36         8       33         11
```

## Annual diagnostics

Mean Gate2-to-consensus grid gap: 1.403.
Mean cons-stack weight on Gate2: 0.840.

## Skipped

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