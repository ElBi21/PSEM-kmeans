# PThreads

Step assegnazione: provare a fare il seguente: al posto di

    $\forall$ punti:
        $\forall$ centroidi:

fare invece

$\forall$ centroidi:
    // Fare array ausiliario per salvarsi le assegnazioni temporaneamente, inizializza con FLT_MAX
    punto_dist_min = 0;
    $\forall$ punti: // Solo punti assegnati a un thread, splitta i punti prima!
        d = || centroide, punto ||;
        if d < punto_dist_min:
            punto_dist_min = d
    class_map
    
    aggiorna_posizione()