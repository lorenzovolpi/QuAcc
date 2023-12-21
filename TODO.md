- [x] aggiungere media tabelle
- [x] plot; 3 tipi (appunti + email + garg)
- [x] sistemare kfcv baseline
- [x] aggiungere metodo con CC oltre SLD
- [x] prendere classe più popolosa di rcv1, togliere negativi fino a raggiungere 50/50; poi fare subsampling con 9 training prvalences (da 0.1-0.9 a 0.9-0.1)
- [x] variare parametro recalibration in SLD


- [x] fix grafico diagonal
    - seaborn example gallery
- [x] varianti recalib: bcts, SLD (provare exact_train_prev=False)
- [x] vedere cosa usa garg di validation size
- [x] per model selection testare il parametro c del classificatore, si esplora in np.logscale(-3,3, 7) oppure np.logscale(-4, 4, 9), parametro class_weight si esplora in None oppure "balanced"; va usato qp.model_selection.GridSearchQ in funzione di mae come errore, UPP come protocollo
    - qp.train_test_split per avere v_train e v_val
    - GridSearchQ(
        model: BaseQuantifier,
        param_grid: {
            'classifier__C': np.logspace(-3,3,7),
            'classifier__class_weight': [None, 'balanced'],
            'recalib': [None, 'bcts']
        },
        protocol: UPP(V_val, repeats=1000),
        error = qp.error.mae,
        refit=True,
        timeout=-1,
        n_jobs=-2,
        verbose=True).fit(V_tr)
- [x] plot collettivo, con sulla x lo shift e prenda in considerazione tutti i training set, facendo la media sui 9 casi (ogni line è un metodo), risultati non ottimizzati e ottimizzati
- [x] salvare il best score ottenuto da ogni applicazione di GridSearchQ
    - nel caso di bin fare media dei due best score
- [x] import baselines

- [ ] importare mandoline
  - mandoline può essere importato, ma richiedere uno slicing delle features a priori che devere essere realizzato ad hoc
- [ ] sistemare vecchie iw baselines
  - non possono essere fixate perché dipendono da numpy
- [x] plot avg con train prevalence sull'asse x e media su test prevalecne
- [x] realizzare grid search per task specifico partendo da GridSearchQ
- [x] provare PACC come quantificatore
- [x] aggiungere etichette in shift plot
- [x] sistemare exact_train quapy
- [x] testare anche su imbd

- [x] aggiungere esecuzione remota via ssh
- [x] testare confidence con sia max_conf che exntropy
- [x] implementare mul3
- [ ] rivedere nuove baselines
- [ ] importare nuovi dataset

- [ ] testare kernel density estimation (alternativa sld)
- [ ] significatività statistica (lunedì ore 10.00)
- [ ] usare un metodo diverso di classificazione sia di partenza che dentro quantificatore per cifar10
- [ ] valutare altre possibili esplorazioni del caso binario

multiclass:
- [x] aggiungere classe per gestire risultato estimator (ExtendedPrev)
- [x] sistemare return in MCAE e BQAE estimate
- [x] modificare acc e f1 in error.py
- [x] modificare report.py in modo che l'index del dataframe sia una tupla di prevalence
- [x] modificare plot per adattarsi a modifiche report
- [x] aggiungere supporto a multiclass in dataset.py
- [x] aggiungere group_false in ExtensionPolicy
- [ ] modificare BQAE in modo che i quantifier si adattino alla casistica(binary/multi in base a group_false)
