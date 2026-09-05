"""Render actual CSV evidence; no simulated observations or fitted oracle frontier."""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak

LABELS={'matern_full':'Matérn completo','constant_full':'Statico regolarizzato','linear_full':'Lineare','rbf_full':'Gaussiano','equal_allocation':'Allocazione uniforme','matern_loss_selected':'Matérn: selezione sulla loss','compact_top8':'Top 8','compact_top16':'Top 16','compact_top32':'Top 32','compact_top64':'Top 64','standalone_top16':'Top 16 per Sharpe individuale','published_by2004':'Pubblicate entro il 2004','finance_prior':'Biblioteca finanziaria predefinita','matern_ew':'Matérn: equal-weighted','matern_vw':'Matérn: value-weighted'}

def build(out: Path) -> None:
    read=lambda name:pd.read_csv(out/(name+'.csv'))
    models=read('selected_models').set_index('model');curves=read('complexity_curves');main=curves.query("model=='matern_full'")
    ranking=read('feature_ranking_VALIDATION_ONLY');groups=read('group_ablation');state=read('state_ablation')
    hist=read('history_comparison');nominal=read('nominal_size_comparison');cost=read('cost_PROXY_sensitivity')
    meta=json.loads((out/'manifest.json').read_text());figs=out/'figures';figs.mkdir(exist_ok=True)
    def save(name):
        plt.tight_layout();plt.savefig(figs/(name+'.png'),dpi=190,bbox_inches='tight');plt.savefig(figs/(name+'.pdf'),bbox_inches='tight');plt.close()
    x=read('primary_path_OOS').iloc[:,1:].to_numpy();rng=np.random.default_rng(414);boot=[]
    for _ in range(1500):
        starts=rng.integers(0,len(x),size=int(np.ceil(len(x)/12)))
        idx=((starts[:,None]+np.arange(12))%len(x)).ravel()[:len(x)]
        y=x[idx];boot.append(np.sqrt(12)*y.mean(0)/y.std(0,ddof=1))
    low,high=np.quantile(boot,[.025,.975],axis=0)
    pd.DataFrame({'C':main.mean_C,'test_sr':main.test_sharpe,'pointwise_95_low':low,'pointwise_95_high':high}).to_csv(out/'primary_curve_bands.csv',index=False)
    plt.figure(figsize=(7.3,3.45));plt.plot(main.mean_C,main.test_sharpe,label='Sharpe test 2005-2025')
    plt.fill_between(main.mean_C,low,high,alpha=.18,label='95% puntuale: block bootstrap')
    chosen=main.loc[main.selected].iloc[0]
    plt.scatter([chosen.mean_C],[chosen.test_sharpe],marker='D',s=50,label=f'Scelta su validation: C={chosen.mean_C:g}')
    plt.xlabel('Complessità C: effective dimension dei managed payoffs');plt.ylabel('Sharpe annualizzato');plt.legend(fontsize=8);save('01_complexity_oos')
    plt.figure(figsize=(7.3,3.05));plt.plot(main.mean_C,main.mean_IS_Sharpe,label='IS del nucleo response-one (mediana)')
    plt.plot(main.mean_C,main.test_sharpe,label='OOS della policy con controllo del rischio')
    plt.yscale('log');plt.xlabel('Complessità C');plt.ylabel('Sharpe (scala log)');plt.legend(fontsize=8);save('02_fit_vs_learning')
    plt.figure(figsize=(7.3,3.6))
    for name in ['constant_full','matern_full','linear_full','rbf_full']:
        r=models.loc[name];v=np.linspace(0,r.volatility,30)
        plt.plot(v,v*r.sharpe,label=LABELS[name]);plt.scatter([r.volatility],[r['mean']],s=22)
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1));plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.xlabel('Volatilità annualizzata');plt.ylabel('Rendimento medio annualizzato');plt.legend(fontsize=8);save('03_attained_risk_return')
    plt.figure(figsize=(7.3,3.5));plt.plot(hist.window,hist.selected_C,marker='o')
    plt.xlabel('Storia usata per la stima (mesi)');plt.ylabel('C selezionata sulla validation');plt.xticks(hist.window);save('04_history_complexity')
    yy=np.arange(len(groups));plt.figure(figsize=(7.3,3.8))
    plt.barh(yy-.17,groups.test_delta_sr,height=.32,label='C riselezionata su validation')
    plt.barh(yy+.17,groups.matched_C_test_delta_sr,height=.32,label='C invariata al valore del modello completo')
    plt.yticks(yy,groups.group.str.replace('_',' '));plt.axvline(0,linewidth=.8)
    plt.xlabel('Sharpe completo meno Sharpe senza il gruppo');plt.legend(fontsize=8);save('05_group_ablation')
    names=['compact_top8','compact_top16','compact_top32','compact_top64','matern_full'];ix=np.arange(len(names))
    plt.figure(figsize=(7.3,3.6));plt.bar(ix-.18,models.loc[names,'validation_sharpe'],width=.34,label='Validation 1995-2004')
    plt.bar(ix+.18,models.loc[names,'sharpe'],width=.34,label='Test 2005-2025')
    plt.xticks(ix,['8','16','32','64','153']);plt.xlabel('Numero di caratteristiche, NON complessità C');plt.ylabel('Sharpe');plt.legend();save('06_feature_selection')
    plt.figure(figsize=(7.3,4.0));yy=np.arange(len(state))
    delta=state.test_delta_sr.to_numpy();lo=state.delta_sr_simultaneous_lo.to_numpy();hi=state.delta_sr_simultaneous_hi.to_numpy()
    plt.errorbar(delta,yy,xerr=np.vstack([delta-lo,hi-delta]),fmt='o',capsize=3)
    plt.yticks(yy,state.removed_state.str.replace('_',' '));plt.axvline(0,linewidth=.8)
    plt.xlabel('Sharpe completo meno Sharpe senza lo stato; bande simultanee 95%');save('07_state_ablation')
    plt.figure(figsize=(7.3,3.6));plt.plot(cost.sleeve_charge_bps,cost.sharpe,marker='o')
    plt.xlabel('Addebito ipotetico per riallocazione fra fattori (bps)');plt.ylabel('Sharpe dopo il solo addebito proxy');save('08_cost_proxy')
    plt.figure(figsize=(7.3,3.6));plt.plot(nominal.nominal_policy_parameters,nominal.selected_C,marker='o',label='Approssimazione RFF')
    plt.axhline(models.loc['matern_full','selected_budget'],linestyle='--',label='Kernel esatto')
    plt.xscale('log');plt.xlabel('Numero nominale di parametri della policy');plt.ylabel('C selezionata');plt.legend();save('09_nominal_representation')
    spec=read('managed_payoff_spectrum');cont=read('oos_spectral_contributions').iloc[:,1:]
    plt.figure(figsize=(7.3,3.1));plt.plot(spec['rank'],spec.mean_share_of_training_trace.cumsum())
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1));plt.xlabel('Rango della direzione stimata');plt.ylabel('Quota cumulata della traccia');save('10_spectral_variance')
    plt.figure(figsize=(7.3,3.1));plt.bar(['1-5','6-15','16-40','41-120'],cont.mean()*12*100)
    plt.axhline(0,linewidth=.8);plt.xlabel('Bande di rango: base stimata prima di ciascun rendimento');plt.ylabel('Contributo al rendimento annuo (punti %)');save('11_oos_spectral_contribution')
    W=A4[0]-96
    body=ParagraphStyle('Body',fontName='Helvetica',fontSize=10.5,leading=15,spaceAfter=9)
    title=ParagraphStyle('Title',fontName='Helvetica-Bold',fontSize=20,leading=24,spaceAfter=15)
    small=ParagraphStyle('Small',parent=body,fontSize=8.8,leading=12,spaceAfter=7)
    cell=ParagraphStyle('Cell',parent=body,fontSize=8.5,leading=11,spaceAfter=0)
    story=[];md=[]
    def p(text,style=body):story.append(Paragraph(text,style));md.append(re.sub('<[^>]*>','',text)+'\n')
    def page(text):
        if story:story.append(PageBreak())
        story.append(Paragraph(text,title));md.append('\n## '+text+'\n')
    def table(headers,rows,widths=None):
        values=[[Paragraph(html.escape(str(v)),cell) for v in row] for row in [headers]+rows]
        t=Table(values,colWidths=widths or [W/len(headers)]*len(headers),repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('LINEABOVE',(0,0),(-1,0),.7,colors.black),('LINEBELOW',(0,0),(-1,0),.5,colors.black),('LINEBELOW',(0,-1),(-1,-1),.5,colors.black),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
        story.extend([t,Spacer(1,12)])
        md.extend(['| '+' | '.join(map(str,headers))+' |','|'+'---|'*len(headers)]+['| '+' | '.join(map(str,r))+' |' for r in rows]+[''])
    def figure(name,height=228):
        im=Image(str(figs/(name+'.png')));ratio=im.imageHeight/im.imageWidth;h=min(height,W*ratio);w=h/ratio
        story.append(Image(str(figs/(name+'.png')),width=w,height=h));story.append(Spacer(1,9));md.append('![](figures/'+name+'.png)\n')
    def metric_table(names):
        rows=[]
        for name in names:
            r=models.loc[name];rows.append([LABELS.get(name,name),f'{r.selected_budget:g}',f'{r.validation_sharpe:.3f}',f'{r.sharpe:.3f}',f'{100*r["mean"]:.2f}%',f'{100*r.volatility:.2f}%'])
        table(['Policy','C','SR val.','SR test','Media ann.','Vol. ann.'],rows,[W*.33,W*.09,W*.13,W*.13,W*.16,W*.16])
    page('Dalla teoria alle decisioni di portafoglio')
    p('<b>Esperimento pubblico Jensen-Kelly-Pedersen | 5 settembre 2026</b>')
    p('La conclusione utile non è scegliere sempre il modello più flessibile. È distinguere quale rappresentazione funziona, quanta complessità attivare e quanto sia stabile il valore attribuito alle singole caratteristiche.')
    p(f'<b>Analisi eseguita:</b> {meta["n_factors"]} portafogli caratteristici USA, {meta["n_selected_models"]} specificazioni selezionate, {meta["n_path_points"]} punti sulle traiettorie di regolarizzazione e 153 rimozioni individuali sulla sola validation. Test separato: {meta["test_months"]} mesi, gennaio 2005-dicembre 2025.')
    metric_table(['constant_full','matern_full','compact_top8'])
    p('<b>Tre decisioni suggerite dall\'evidenza.</b> Mantenere un benchmark statico regolarizzato serio. Non eliminare caratteristiche solo per comprimere la libreria. Non usare il migliore Sharpe di test come regola di scelta: la biblioteca top-8 vince sulla validation ma non nel test successivo.')
    p('<b>Perimetro:</b> questi sono rendimenti pubblici di portafogli long-short costruiti sulle caratteristiche JKP. Non sono il pannello azionario CTF, non sono una submission al leaderboard e non permettono di ricostruire costi o posizioni sui singoli titoli. I risultati precedenti nei PDF del progetto non sono usati come dati di questo esperimento. [1,2]')
    p('La parola complessità indica soltanto C, la effective dimension dei managed payoffs. Il numero di caratteristiche e la dimensione nominale della rappresentazione restano oggetti distinti. [3,4]',small)
    page('1. Che cosa è stato effettivamente stimato')
    table(['Fase','Periodo e decisione'],[['Dati','Serie USA ufficiali JKP; variante principale capped-value-weighted; valori già orientati long-short dagli autori.'],['Calibrazione','Gennaio 1973-dicembre 1994: copertura della biblioteca, scala degli stati e lunghezza del kernel.'],['Validation','Gennaio 1995-dicembre 2004: scelta di C, ranking delle caratteristiche e numero di caratteristiche.'],['Test','Gennaio 2005-dicembre 2025: solo valutazione delle scelte bloccate.'],['Ristima','Mensile, ultimi 120 mesi; sensitività 60 e 240 mesi sulle stesse date di valutazione.']],[W*.20,W*.80])
    p('Gli stati sono le medie e volatilità mobili a 12 mesi dei rendimenti di sette gruppi bibliografici, ritardate di un mese: 14 coordinate. Non sono le 13 categorie empiriche stimate da JKP. Nessun rendimento del mese corrente entra nella decisione di quel mese.')
    p('<b>Nucleo statistico:</b> G_ts = K(Z_t,Z_s) R_t\'R_s / T; C(lambda) = tr[G(G + lambda I)^(-1)]. Il livello lambda si trova usando soltanto gli autovalori del training, per raggiungere un budget C prefissato. È una procedura empirica data-dependent, non uno stimatore dei parametri strutturali b e r.')
    p('Controllo del rischio: obiettivo ex ante 10% annuo, covarianza con 10% di shrinkage verso la diagonale e somma dei pesi assoluti sui fattori non superiore a tre. Il vincolo può impedire di raggiungere il 10%. La loss nativa è calcolata prima di questo scaling; C descrive il nucleo stimato, non ogni proprietà della policy finale.')
    p('<b>Vincolo storico:</b> la libreria completa è retrospettiva e contiene anche lavori pubblicati dopo il 2004. La sensitività su 57 caratteristiche pubblicate entro il 2004 limita questo problema, ma non ricostruisce una vintage storica dei dati. [1]')
    page('2. Quale rappresentazione regge fuori campione?')
    metric_table(['constant_full','matern_full','linear_full','rbf_full','equal_allocation'])
    figure('03_attained_risk_return',240)
    p('Il kernel costante corrisponde a una combinazione di fattori senza timing nonlineare degli stati, ma con pesi ristimati ogni mese. Qui offre Sharpe maggiore del Matérn. Non lo domina in ogni unità economica: il Matérn produce media e certainty equivalent realizzati maggiori, con maggiore volatilità.')
    a=models.loc['matern_full'];b=models.loc['constant_full']
    p(f'CE annuale mean-variance, gamma=5: Matérn {100*a.ce_gamma5:.2f}%, statico {100*b.ce_gamma5:.2f}%. I segmenti del grafico arrivano al rischio effettivamente osservato; non estrapolano il portafoglio statico al 10% violando il vincolo sui pesi.')
    inference=read('paired_inference');ci=inference.query("comparison=='matern_full minus constant_full' and block==12").iloc[0]
    p(f'La differenza Sharpe Matérn meno statico è {ci.delta_sr:.3f}. Il paired block bootstrap a 12 mesi produce un intervallo 95% [{ci.lo:.3f}; {ci.hi:.3f}]. È inferenza condizionata alle specificazioni selezionate, non una garanzia di dominanza futura. [File: paired_inference.csv]',small)
    page('3. Complessità: beneficio, limite e incertezza')
    figure('01_complexity_oos',255);figure('02_fit_vs_learning',218)
    best=main.loc[main.test_sharpe.idxmax()];last=main.iloc[-1]
    p(f'La validation sceglie C={chosen.mean_C:g}: Sharpe test {chosen.test_sharpe:.3f}. Il massimo descrittivo del test è C={best.mean_C:g}, Sharpe {best.test_sharpe:.3f}: non viene usato per cambiare la scelta. A C={last.mean_C:g} lo Sharpe test è {last.test_sharpe:.3f}, mentre lo Sharpe IS mediano del nucleo è {last.mean_IS_Sharpe:.1f}. Non emerge il collasso generalizzato suggerito dalle illustrazioni precedenti.')
    p('La banda è puntuale, al 95%, da blocchi circolari di 12 mesi. Non include nuova selezione del modello o ristima in ciascun bootstrap. Il confronto IS/OOS distingue il fit del nucleo response-one dalla performance della policy con il controllo causale del rischio. Non dimostra una forma universale della curva. [3,4,5]',small)
    page('4. Il valore della storia di mercato')
    table(['Training (mesi)','C scelta','SR validation','SR test','lambda medio'],[[int(r.window),f'{r.selected_C:.0f}',f'{r.validation_sr:.3f}',f'{r.test_sr:.3f}',f'{r.mean_lambda:.6f}'] for r in hist.itertuples()])
    figure('04_history_complexity',255)
    p('In questo esperimento, più storia sostiene una complessità selezionata maggiore: 32, 48, 64. Il Matérn con 240 mesi migliora lo Sharpe test rispetto alle finestre da 60 e 120 mesi, e riduce la riallocazione media mensile dei fattori da circa 0.874 a 0.480.')
    p('<b>Non tutto conferma la legge asintotica.</b> Il lambda medio cresce leggermente anziché diminuire. Qui la griglia è in C e lambda è invertito su spettri campionari diversi; b e r sono ignoti. Questi numeri non identificano gli esponenti teorici e non sono una validazione quantitativa della legge di regolarizzazione.')
    p('Decisione operativa da sottoporre a nuova verifica: scegliere congiuntamente finestra e C su validation temporale, conservando un test ulteriore. Non adottare automaticamente 240 mesi perché è il migliore dei tre nel test già osservato.')
    page('5. Quali gruppi di caratteristiche servono?')
    figure('05_group_ablation',275)
    p('Per ogni gruppo vengono eliminati i relativi portafogli, lasciando invariati gli stati. La differenza di Sharpe è misurata sia mantenendo C=48 sia scegliendo di nuovo C sulla validation. Positivo significa che la rimozione peggiora il risultato.')
    vg=groups.loc[groups.group=='value'].iloc[0]
    p(f'<b>Esempio decisivo: value.</b> Con C riselezionata la rimozione costa {vg.test_delta_sr:.3f} punti di Sharpe nel test. A C invariata, la differenza è {vg.matched_C_test_delta_sr:.3f}. Non sarebbe corretto attribuire tutta la prima differenza al contenuto economico di value: cambia anche la regolarizzazione scelta.')
    p('Il gruppo intangibles mostra contributi positivi in entrambi i confronti nei point estimates, ma gli intervalli simultanei al 95% non escludono zero per nessuno dei gruppi. Le correlazioni fra caratteristiche impediscono di interpretare la rimozione come un effetto causale isolato.')
    p('Uso pratico: confrontare contributo marginale, sensibilità a C e incertezza prima di eliminare un gruppo. La sparsità della lista di caratteristiche non coincide con una piccola complessità del portafoglio, distinzione centrale anche in Kozak-Nagel-Santosh. [3]')
    page('6. Le caratteristiche selezionate prima del test')
    translations={'ret_12_7':'Momentum dei mesi 12-7','resff3_12_1':'Momentum residuo 12-1','ival_me':'Valore intrinseco / mercato','eq_dur':'Duration azionaria','fcf_me':'Free cash flow / prezzo','ebitda_mev':'EBITDA / enterprise value','rd_me':'R&D / valore di mercato','seas_2_5an':'Stagionalità annuale, anni 2-5'}
    rows=[]
    for r in ranking.head(8).itertuples():rows.append([int(r.validation_joint_rank),r.characteristic,translations.get(r.characteristic,r.description),int(r.publication_year),f'{r.validation_contribution_sr:.4f}'])
    table(['#','Codice JKP','Significato','Anno','Delta SR val.'],rows,[W*.05,W*.22,W*.40,W*.10,W*.23])
    p('Ranking ottenuto eliminando una caratteristica alla volta e riselezionando C sulla sola validation. Delta SR = Sharpe completo meno migliore Sharpe dopo la rimozione. Il file completo contiene anche il confronto a C invariata e il ranking basato sul solo Sharpe individuale.')
    p('<b>Questa è una shortlist di ricerca, non una lista dimostrata di alpha indispensabili.</b> Alcune pubblicazioni sono posteriori al 2004: non erano una selezione conoscibile all\'inizio del test. Inoltre, i controlli successivi sulle prime dieci caratteristiche non mostrano contributi individuali diversi da zero nelle bande simultanee al 95%.')
    p('La domanda qui è quali portafogli caratteristici includere nell\'allocazione. La domanda distinta, quali variabili grezze inserire nella funzione dei pesi azionari, richiede il pannello CTF. I due problemi non sono intercambiabili. [1,2]')
    page('7. La sparsità selezionata non vince il test')
    figure('06_feature_selection',260);metric_table(['compact_top8','compact_top16','compact_top32','compact_top64','matern_full'])
    selected_compact=meta['selected_compact_model'];compact=models.loc[selected_compact]
    p(f'La stessa validation sceglie la dimensione della libreria: vince {selected_compact} con Sharpe {compact.validation_sharpe:.3f}. Nel test arriva a {compact.sharpe:.3f}, contro {models.loc["matern_full","sharpe"]:.3f} della libreria completa regolarizzata. Scegliere un altro numero di caratteristiche dopo aver visto il test significherebbe usarlo per fare model selection.')
    p('Le librerie compatte ricostruiscono anche gli stati usando soltanto le caratteristiche trattenute. Il confronto misura quindi una scelta completa di rappresentazione, non il solo effetto meccanico del numero di fattori.')
    p('<b>Conseguenza:</b> non vedo evidenza sufficiente per sostituire automaticamente la biblioteca ampia con la shortlist. Meglio tenere entrambe come strategie concorrenti bloccate e confrontarle in un periodo non ancora utilizzato. La regolarizzazione e la selezione dura di caratteristiche risolvono problemi diversi. [3,4]')
    page('8. Quali informazioni inserire negli stati?')
    figure('07_state_ablation',290)
    p('Qui l\'universo investibile resta composto da tutti i 153 fattori. Si rimuovono invece le due coordinate di un gruppo dagli stati, oppure tutte le medie o tutte le volatilità. Il kernel mantiene la calibrazione pre-validation; C viene riselezionata.')
    p('Le bande simultanee non sostengono una conclusione forte su un singolo gruppo. Nei point estimates, eliminare le volatilità porta lo Sharpe test a circa 0.795, ma questo non autorizza a scegliere retroattivamente quella specificazione. Eliminare gli stati value porta invece a circa 0.462.')
    p('Il test utile per il paper non è soltanto una classifica di feature importance: è capire se un\'informazione serve come segnale di timing, come investimento da detenere o come copertura. Queste prove isolano almeno le prime due funzioni. Per distinguere premi, rischio e costi a livello azionario servono dati e controlli ulteriori. [3,6]')
    page('9. I costi possono cambiare la scelta, ma quali costi?')
    table(['Addebito (bps)','C scelta','SR test dopo proxy','CE gamma=5'],[[int(r.sleeve_charge_bps),f'{r.selected_budget:g}',f'{r.sharpe:.3f}',f'{100*r.ce_gamma5:.2f}%'] for r in cost.itertuples()])
    figure('08_cost_proxy',235)
    p('L\'addebito è la tariffa ipotetica moltiplicata per la somma delle variazioni assolute dei pesi fra portafogli caratteristici. Con 25 o 50 bps, la validation passa da C=48 a C=1. Questo mostra che il criterio netto può preferire un uso diverso della rappresentazione.')
    p('<b>Non è un backtest netto azionario.</b> Mancano il turnover interno dei fattori, i costi dei titoli, il netting delle operazioni fra strategie, il prestito titoli, lo spread e il price impact. Questa proxy non è né una stima completa né necessariamente un limite inferiore ai costi reali.')
    p('La complessità non misura automaticamente turnover o capacità. Il passaggio alla frontiera implementabile richiede holdings e trades reali, come nel problema di Jensen-Kelly-Malamud-Pedersen. La regressione quadratica di base non si estende a costi nonlineari semplicemente sostituendo i rendimenti con quelli netti. [7]')
    page('10. Dimensione nominale: il plateau non è provato')
    table(['Feature RFF P','Parametri 153P','C scelta','SR test'],[[int(r.state_features),int(r.nominal_policy_parameters),f'{r.selected_C:g}',f'{r.test_sr:.3f}'] for r in nominal.itertuples()])
    figure('09_nominal_representation',245)
    p('Le approssimazioni usano prefissi della stessa estrazione di frequenze. Aumentando P la complessità selezionata varia da 8 a 118 e lo Sharpe test non converge monotonicamente al kernel esatto. In questa griglia e con questo seed non emerge il plateau stabile anticipato nelle precedenti illustrazioni.')
    p('Il fatto che i parametri nominali siano molti più di C resta vero per definizione e per i valori osservati. Non basta, però, per certificare che l\'approssimazione abbia recuperato la stessa regola economica. Servono più accuratezza e più seed, da trattare come nuova robustness e non da selezionare sul test corrente. [4,5]')
    page('11. Dove sta il rendimento nel sistema spettrale?')
    figure('10_spectral_variance',210);figure('11_oos_spectral_contribution',210)
    shares=spec.mean_share_of_training_trace
    cc=cont.mean()*12*100
    p(f'I primi cinque ranghi spiegano il {100*shares.iloc[:5].sum():.1f}% della traccia media di training; i primi quindici il {100*shares.iloc[:15].sum():.1f}%. I contributi al rendimento OOS annuo delle quattro bande sono '+', '.join(f'{z:+.2f}' for z in cc)+' punti percentuali. La loro somma ricostruisce il rendimento della policy.')
    p('La base è stimata prima di ciascun payoff. Le bande raggruppano ranghi mobili, non necessariamente gli stessi fattori economici nel tempo. Questi contributi sono firmati e possono essere negativi: non sono quote del vero segnale, contributi separabili allo Sharpe o una stima di r. Mostrano perché sola varianza spiegata e solo conteggio delle caratteristiche non bastano. [3,4]')
    page('12. Robustezza, interpretazione e decisione finale')
    metric_table(['matern_full','matern_vw','matern_ew','published_by2004','finance_prior'])
    p('La variante equal-weighted ha Sharpe test circa 1.575, molto maggiore della capped-value-weighted principale. È una sensitività cruciale, non il nuovo headline da scegliere dopo il test. Potrebbe riflettere differenze nella composizione e negoziabilità, ma qui non abbiamo i titoli per identificarne la causa.')
    p('<b>Che cosa farei con questi risultati:</b> mantenere la libreria ampia con shrinkage e un benchmark statico regolarizzato; trattare la shortlist come concorrente da verificare, non come lista da adottare; scegliere finestra e C prima del test; valutare ogni esclusione sia a C invariata sia con nuova regolarizzazione.')
    p('<b>Che cosa non farei:</b> promettere un massimo universale di Sharpe, stimare b o r da queste poche finestre, presentare segmenti di rischio-rendimento stimati come distanze note dalla frontiera di popolazione, oppure chiamare netti i risultati con il solo addebito proxy.')
    p('L\'evidenza pratica più forte è che <b>la representation e la selezione delle caratteristiche contano quanto la regolarizzazione</b>. Questa analisi non incorona un kernel. Fornisce un protocollo e risultati controllabili per separare apprendimento, selezione, scala e costi.')
    page('13. Replica, limiti e collegamento con la letteratura')
    p('<b>Replica:</b> fetch_public.py scarica i file ufficiali con SHA256 e provenienza; engine.py costruisce stati e policy causali; run_study.py esegue le selezioni cronologiche; build_report.py produce questa nota dai CSV. I test verificano timing, troncamento, assenza di selezione sul test, algebra primal-dual, scaling, rango e determinismo.')
    p('Comandi: python -m unittest discover -s empirical_jkp_practical/tests -v; python empirical_jkp_practical/run_study.py --raw jkp_public_raw --out empirical_jkp_practical/results; python empirical_jkp_practical/build_report.py --out empirical_jkp_practical/results.',small)
    p('Il bootstrap ricampiona rendimenti delle policy già selezionate. Non ripete l\'intera ricerca di caratteristiche. Le bande simultanee riguardano la famiglia indicata, non ogni scelta provata nel progetto. La storia già vista non diventa nuovamente un test indipendente.',small)
    refs=[
        '[1] Jensen, Kelly e Pedersen (2023), Is There a Replication Crisis in Finance?, Journal of Finance 78, 2465-2518. Dataset ufficiale: https://jkpfactors.com/data. Vintage scaricata il 5 settembre 2026; ultimo mese usato dicembre 2025; licenza CC BY-NC 4.0, solo uso di ricerca non commerciale.',
        '[2] JKP Common Task Framework, Dataset Access: https://jkpfactors.com/ctf/dataset-access. Il pannello delle caratteristiche e rendimenti azionari richiede accesso autorizzato WRDS; non è il dataset pubblico di portafogli di questa nota.',
        '[3] Kozak, Nagel e Santosh (2020), Shrinking the Cross Section, Journal of Financial Economics 135, 271-292. Confronto pertinente: shrinkage spettrale e sparsità delle caratteristiche non sono la stessa cosa.',
        '[4] Kelly, Malamud e Zhou (2024), The Virtue of Complexity in Return Prediction, Journal of Finance 79, 459-503; Kelly e Malamud (2025), Understanding the Virtue of Complexity. Confronto pertinente: representation, shrinkage e alignment; stessa C non implica stessa performance fra kernel.',
        '[5] Nagel (2025), Seemingly Virtuous Complexity in Return Prediction. Confronto pertinente: interpretare la regola economica indotta dalla rappresentazione e non affidarsi al solo conteggio di parametri.',
        '[6] Brandt, Santa-Clara e Valkanov (2009), Parametric Portfolio Policies, Review of Financial Studies 22, 3411-3447; Simon, Weibels e Zimmermann (2026), Deep Parametric Portfolio Policies, Management Science, doi:10.1287/mnsc.2025.00721. Direct policy learning e ruolo delle preferenze.',
        '[7] Jensen, Kelly, Malamud e Pedersen, Machine Learning and the Implementable Efficient Frontier, versione di ricerca citata nel progetto. Confronto pertinente: holdings, trading dinamico e costi devono entrare nel problema di implementazione; non sono osservabili dalle sole serie di fattori.'
    ]
    for ref in refs:p(ref,small)
    p('Questa nota distingue risultati eseguiti, interpretazioni e limiti. Non certifica una strategia pronta per capitale reale e non convalida numeri stock-level presentati altrove nel progetto.',small)
    def footer(c,doc):
        c.saveState();c.setFont('Helvetica',8);c.drawString(48,24,'PORTFOLIO | Evidenza pubblica JKP | 05.09.2026');c.drawRightString(A4[0]-48,24,str(doc.page));c.restoreState()
    pdf=out/'JKP_Practical_Learnability.pdf'
    SimpleDocTemplate(str(pdf),pagesize=A4,leftMargin=48,rightMargin=48,topMargin=43,bottomMargin=42,title='JKP: conseguenze pratiche della portfolio learnability',author='PORTFOLIO research note').build(story,onFirstPage=footer,onLaterPages=footer)
    (out/'REPORT.md').write_text('\n'.join(md),encoding='utf-8');print('Wrote',pdf)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('empirical_jkp_practical/results'));a=p.parse_args();build(a.out)
